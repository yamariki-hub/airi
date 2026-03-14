---
title: "会場別モデル戦略：全24場を均一に扱う間違いをやめる"
emoji: "🗾"
type: "tech"
topics: ["機械学習", "競艇", "Python", "データ分析", "会場別"]
published: false
price: 500
---

去年の秋、初めて競艇場に一人で行った。

下関競艇場。山口県の端っこにある、静かな競艇場だ。平日の午後だったから観客が少なく、スタンドに一人でビールを持って座れた。

データを見ながら生レースを観戦するのは、ノートパソコンの前で見るのとは全然違う。水面が実際にそこにある。モーターの音がする。1号艇が内側をきれいに回るとき、「ああ、これが大外有利な競艇場と内側有利な競艇場の違いか」と体で理解した。

その日の私の下関のペーパートレードROIは268%だった。

---

---

## 競艇場は全部違う

全国24の競艇場は、それぞれ全く異なる特性を持つ。

水面の広さ、風向き、潮の影響、スタンドの向き。これらが組み合わさって、各競艇場の「コース有利性」が決まる。

1号艇（最内コース）の場別勝率を見ると、その差は衝撃的だ：

- **大村**　1号艇勝率 70.8%　— 全国最強のイン有利。狭い湾内水路。
- **若松**　1号艇勝率 66.3%　— イン有利。九州の穏やかな水面。
- **唐津**　1号艇勝率 65.4%　— 海面、比較的穏やか。
- **尼崎**　1号艇勝率 64.2%　— 淡水、静水面。
- **下関**　1号艇勝率 62.1%　— 比較的イン有利。
- **平和島**　1号艇勝率 46.2%　— 首都圏の荒れ場。
- **戸田**　1号艇勝率 46.8%　— 狭い水路、荒れやすい。
- **江戸川**　1号艇勝率 43.5%　— 河川、潮・風の影響大。

**大村70.8% vs 江戸川43.5%。27ポイントの差。**

これは同じモデルで全会場を予測すること自体が間違いである、ということを示している。

---

## 競艇場の特性深掘り

### 大村（長崎）：1号艇の聖地

```
水面タイプ: 湾内
特徴:       三方を山に囲まれ風の影響が少ない。水路が狭く、
            外から差すスペースが物理的に制限される。
1号艇勝率: 70.8%（全国1位）
戦略:       1号艇以外にベットするには強い理由が必要。
            「モーター2連率が明確に劣る」など。
```

### 平和島・戸田・江戸川：荒れ場トリオ

これら3つは首都圏に集中する「難水面」だ。

```
平和島: 東京都大田区。東京湾の影響を受けた強い風。
戸田:   埼玉県。狭い水路でスタートが難しい。
江戸川: 千葉・東京の境界。河川なので潮と風の複合影響。
```

これらの競艇場では1号艇の勝率が45%前後まで下がる。外コースの艇が差し込んでくる頻度が高い。

「荒れる＝穴馬が来る＝高配当の機会」だが、予測が困難でもある。

---

## 会場別モデルのアーキテクチャ

一つのモデルで24会場を全部カバーするのではなく、会場ごとに別々のモデルを訓練する。

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
import sqlite3
from typing import Optional
import logging

logger = logging.getLogger(__name__)

STADIUM_NAMES = {
    "01": "桐生",  "02": "戸田",   "03": "江戸川",
    "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡",  "08": "常滑",   "09": "津",
    "10": "三国",  "11": "びわこ",  "12": "住之江",
    "13": "尼崎",  "14": "鳴門",   "15": "丸亀",
    "16": "児島",  "17": "宮島",   "18": "徳山",
    "19": "下関",  "20": "若松",   "21": "芦屋",
    "22": "福岡",  "23": "唐津",   "24": "大村",
}

# ブラックリスト（ペーパートレード実績に基づく）
BLACKLIST_STADIUMS = {"17"}  # 宮島（ROI 0%）

# インコース特化戦略を適用する会場（1号艇勝率65%以上）
INSIDE_DOMINANT = {"24", "20", "23"}  # 大村・若松・唐津


class StadiumModels:
    """
    会場別に個別モデルを管理するクラス。

    設計思想:
    1. 各会場に専用LightGBMモデルを持つ
    2. 会場特性に合わせた追加特徴量を生成
    3. 訓練データが少ない会場はグローバルモデルにフォールバック
    """

    def __init__(
        self,
        db_path:           str = "boatrace.db",
        min_train_races:   int = 300,  # 会場モデル訓練に必要な最低レース数
    ):
        self.db_path         = db_path
        self.min_train_races = min_train_races
        self.models:        dict[str, lgb.Booster] = {}
        self.global_model:  Optional[lgb.Booster]  = None
        self.feature_names: list[str]               = []

    def train_all(self, training_days: int = 90):
        """
        全会場のモデルを訓練する。

        データが少ない会場はスキップ（グローバルモデルで対応）。
        """
        logger.info("Training global model...")
        self.global_model = self._train_global(training_days)

        available = self._get_available_stadiums(training_days)

        for stadium_code, n_races in available.items():
            name = STADIUM_NAMES.get(stadium_code, stadium_code)

            if stadium_code in BLACKLIST_STADIUMS:
                logger.info(f"Skipping {name} (blacklisted)")
                continue

            if n_races < self.min_train_races:
                logger.info(
                    f"{name}: {n_races}レース < {self.min_train_races}"
                    f"（不足）→ グローバルモデル使用"
                )
                continue

            logger.info(f"Training {name} model ({n_races}レース)...")
            model = self._train_stadium(stadium_code, training_days)
            if model:
                self.models[stadium_code] = model

        logger.info(
            f"Complete: {len(self.models)}会場の専用モデル "
            f"+ グローバルモデル"
        )

    def predict(
        self,
        features:     pd.DataFrame,
        stadium_code: str,
    ) -> np.ndarray:
        """
        指定会場のモデルで予測する。

        専用モデルがあればそれを使い、なければグローバルモデルにフォールバック。
        """
        if stadium_code in BLACKLIST_STADIUMS:
            logger.warning(f"Stadium {stadium_code} is blacklisted.")
            return np.zeros(len(features))

        model = self.models.get(stadium_code, self.global_model)
        if model is None:
            raise RuntimeError("No model available. Call train_all() first.")

        X = features[self.feature_names]
        return model.predict(X)

    def _train_stadium(
        self,
        stadium_code: str,
        training_days: int,
    ) -> Optional[lgb.Booster]:
        """会場専用モデルを訓練する。"""
        df = self._load_training_data(training_days, stadium_filter=stadium_code)
        if df.empty:
            return None

        df = self._add_stadium_features(df, stadium_code)
        X  = df[self.feature_names]
        y  = (df["finish_position"] == 1).astype(int)

        params  = self._get_lgbm_params(stadium_code)
        dataset = lgb.Dataset(X, label=y)

        return lgb.train(
            params,
            dataset,
            num_boost_round=300,
            valid_sets=[dataset],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(100),
            ],
        )

    def _add_stadium_features(
        self,
        df:           pd.DataFrame,
        stadium_code: str,
    ) -> pd.DataFrame:
        """
        会場特性に応じた追加特徴量を生成する。

        大村などイン有利: インコースの有利性をモデルに明示
        江戸川など荒れ場: 外コースの差し込み特徴量を追加
        """
        if stadium_code in INSIDE_DOMINANT:
            # インコース有利度（1号艇=1.0, 2号艇=0.71, ...）
            df["inside_advantage"] = 1.0 / (df["boat_number"] ** 0.5)

        elif stadium_code in {"04", "02", "03"}:
            # 外コース（4〜6号艇）の展示タイム優位性
            df["outer_ex_advantage"] = np.where(
                df["boat_number"] >= 4,
                -df.get("ex_time_from_best", 0.0),
                0.0,
            )

        return df

    def _get_lgbm_params(self, stadium_code: str) -> dict:
        """
        会場によってハイパーパラメータを変える。

        荒れ場: より複雑なモデル（深い木）
        安定場: シンプルなモデル（浅い木）
        """
        base = {
            "objective":     "binary",
            "metric":        "auc",
            "learning_rate": 0.05,
            "verbose":       -1,
            "random_state":  42,
        }

        if stadium_code in {"04", "02", "03"}:
            # 荒れ場は複雑な相互作用を捉えるため木を深く
            base.update({
                "num_leaves": 63,
                "max_depth":   8,
                "min_child_samples": 10,
            })
        else:
            # 安定場はシンプルに
            base.update({
                "num_leaves": 31,
                "max_depth":   6,
                "min_child_samples": 20,
            })

        return base

    def _load_training_data(
        self,
        training_days:   int,
        stadium_filter:  Optional[str] = None,
    ) -> pd.DataFrame:
        """DBから訓練データを読み込む。"""
        conn  = sqlite3.connect(self.db_path)
        query = f"""
            SELECT r.*, c.national_wr, c.motor_2rate, c.boat_2rate,
                   e.ex_time_from_best, e.ex_time_rank
            FROM race_results r
            LEFT JOIN race_cards c
                ON r.race_id = c.race_id AND r.boat_number = c.boat_number
            LEFT JOIN exhibition_times e
                ON r.race_id = e.race_id AND r.boat_number = e.boat_number
            WHERE r.race_date >= date('now', '-{training_days} days')
        """
        if stadium_filter:
            query += f" AND substr(r.race_id, 1, 2) = '{stadium_filter}'"

        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def _get_available_stadiums(self, training_days: int) -> dict[str, int]:
        """訓練可能な会場とそのレース数を返す。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(f"""
            SELECT substr(race_id, 1, 2) as code,
                   COUNT(DISTINCT race_id) as n_races
            FROM race_results
            WHERE race_date >= date('now', '-{training_days} days')
            GROUP BY code
            ORDER BY n_races DESC
        """).fetchall()
        conn.close()
        return dict(rows)

    def _train_global(self, training_days: int) -> lgb.Booster:
        """全会場統合のグローバルモデルを訓練する。"""
        df = self._load_training_data(training_days)
        X  = df[self.feature_names] if self.feature_names else df.select_dtypes("number")
        y  = (df["finish_position"] == 1).astype(int)

        dataset = lgb.Dataset(X, label=y)
        model   = lgb.train(
            {"objective": "binary", "metric": "auc",
             "num_leaves": 31, "learning_rate": 0.05, "verbose": -1},
            dataset,
            num_boost_round=300,
        )
        self.feature_names = model.feature_name()
        return model
```

---

## ペーパートレード実績：場別ROI

現時点（16件）のデータ。サンプルが少ないのは承知の上で、傾向として記録しておく。

- **下関**: ベット数 4件　的中 3件　ROI 268%　— 優秀（データ少）
- **芦屋**: ベット数 3件　的中 2件　ROI 170%　— 良好（データ少）
- **蒲郡**: ベット数 2件　的中 1件　ROI 170%　— 良好（データ少）
- **戸田**: ベット数 1件　的中 0件　ROI 0%　— 不明（1件のみ）
- **びわこ**: ベット数 2件　的中 0件　ROI 29%　— BL候補（データ少）
- **宮島**: ベット数 3件　的中 0件　ROI 0%　— **BL済み**

---

## ナイター競艇の特殊性

住之江、びわこ、桐生、蒲郡などはナイター開催がある。夜のレースはデイレースとは別物だ。

```python
def add_night_race_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    ナイター・デイの区別を特徴量に追加する。

    ナイター競艇: 照明が水面に反射し視界が異なる。
    選手によってナイター得意・不得意がある。
    同じ会場でも昼と夜では別の予測が必要かもしれない。
    """
    NIGHTER_STADIUMS = {"12", "11", "01", "07"}  # 住之江, びわこ, 桐生, 蒲郡

    df["is_night_race"] = (
        df["stadium_code"].isin(NIGHTER_STADIUMS)
        & (df["race_time"].str[:2].astype(int) >= 15)
    ).astype(int)

    return df
```

ナイターの有無が予測精度に影響するかどうかは未検証だ。でも「同じ競艇場でも昼と夜では別物」という直感はある。

---

競艇場に実際に足を運ぶ価値は、データ収集だけじゃないと思う。水面の感触、風の強さ、選手の動き。それらは数値化できないが、モデルを作る上での「勘所」を養ってくれる。

下関の帰り、新幹線の中でビールを飲みながら、今日見たレースをメモした。あの1号艇の出足は良かった。展示タイムと実際の動きの対応関係。

データだけでは分からないことがある。競艇場に行くのはそれを確認するためだ。

---

*次回：v4への挑戦。強化学習・Transformer・アンサンブル——次世代アーキテクチャの全アイデアを公開する。*
