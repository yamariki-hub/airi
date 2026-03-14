---
title: "特徴量エンジニアリング：何をモデルに入力するか"
emoji: "🔧"
type: "tech"
topics: ["機械学習", "競艇", "Python", "データ分析", "特徴量"]
published: false
price: 300
---

モデルの精度は、特徴量で9割決まる。

アルゴリズムを変えるより、特徴量を改善する方が効果が大きい。これはどんな機械学習プロジェクトでも同じだが、競艇は特にそれが顕著だった。

私がそれを実感したのは、深夜2時にモデルを改善しようとしてLightGBMのパラメータを片っ端から変えていたときだ。`num_leaves` を31から127に増やした。学習率を0.05から0.01にした。`bagging_fraction` を0.7にした。何をやってもROIがほとんど変わらない。

諦めかけたとき、ふと「展示タイムの特徴量の計算方法を変えてみよう」と思いついた。レース内での相対順位を追加しただけで、ROIが2ポイント改善した。

**特徴量が全てを決める。** アルゴリズムは二の次だ。

---

---

## 特徴量の全体像：2種類に分けて考える

私が現在使っている特徴量は大きく2グループに分けている。

**なぜ2グループに分けるのか。** それは「出走表が取れない日」への対応だ。出走表スクレイピングが失敗した日や、過去の出走表データが揃っていない期間でも、モデルを動かせるようにしておきたい。だから「出走表なしでも動くベース特徴量」と「出走表があれば追加するカード特徴量」に分けて管理している。

### ベース特徴量（出走表なしでも使えるもの）

```python
# forecast_v3.py より抜粋
BASE_FEATURES = [
    # ---- 基本情報 ----
    "boat_number",       # 枠番（1〜6）。最重要。コース≒枠番が勝負を決める
    "stadium_num",       # 会場番号（01〜24）。会場ごとに1号艇勝率が異なる
    "race_number",       # レース番号（1〜12）。第1Rと最終Rでは特性が違う

    # ---- ローリング選手成績（直近30日）----
    "racer_win_rate_30d",    # 直近30レースでの1着率（当日除く）
    "racer_2rate_30d",       # 直近30レースでの2着以内率
    "racer_3rate_30d",       # 直近30レースでの3着以内率
    "racer_avg_st_30d",      # 直近30レースの平均スタートタイミング

    # ---- ローリング選手成績（直近90日）----
    "racer_win_rate_90d",
    "racer_2rate_90d",
    "racer_3rate_90d",
    "racer_avg_st_90d",      # STの安定性を見る主指標（90日が最良だった）

    # ---- ローリング選手成績（直近180日）----
    "racer_win_rate_180d",
    "racer_2rate_180d",
    "racer_3rate_180d",
    "racer_avg_st_180d",     # 長期トレンド（季節性のあるスランプ/好調を捉える）

    # ---- レース内相対特徴量 ----
    "rel_win_rate_90d",      # このレースの平均勝率との偏差（飛び抜けて強い/弱いを検出）
    "boat1_win_rate_90d",    # 1号艇選手の直近勝率（1号艇が弱い＝波乱の可能性）
    "field_max_min_gap",     # レース内の最高〜最低勝率の差（実力格差の度合い）

    # ---- 枠適性 ----
    "racer_boat_win_rate",   # この選手×この枠番での過去勝率（枠なれ度合い）
    "racer_st_std",          # STの標準偏差（小さいほど安定、大きいと穴の可能性）
]
```

### カード特徴量（出走表から取れるもの）

```python
CARD_FEATURES = [
    # ---- 選手情報（出走表記載の最新期成績）----
    "class_num",         # 級別数値化（A1=4, A2=3, B1=2, B2=1）
    "national_win_rate", # 全国勝率（直近期）※national_2rateはVIF=14で削除済み
    "local_win_rate",    # 当地勝率（その場での特性を反映）
    "weight",            # 体重(kg)。軽いほど有利という説があるが効果は微妙
    "card_avg_st",       # 出走表記載の平均ST（より公式なデータ）
    "flying_count",      # フライング回数（ペナルティ。多いと慎重すぎる傾向）

    # ---- 機材成績 ----
    "motor_2rate",       # モーター2連率（機材の当たり外れを表す重要指標）
                         # ※motor_3rateはVIF≈5で冗長 → 削除済み
    "boat_2rate",        # ボート2連率

    # ---- レース内相対特徴量（カード版）----
    "motor_2rate_rel",   # モーター2連率のレース内偏差（このモーターは良いか？）
    "national_wr_rel",   # 全国勝率のレース内偏差（この選手は場違いに強いか？）
    "class_rel",         # 級別のレース内偏差（格の差）
    "field_top_wr",      # レース内最高勝率（最強選手はどれほど強いか）
    "boat1_national_wr", # 1号艇の全国勝率（1号艇が弱ければ荒れる）
    "is_top_motor",      # そのレースで最良モーターを持っているか（0/1フラグ）

    # ---- 展示タイム特徴量 ----
    "exhibition_time",   # 生の展示タイム（秒）
    "ex_time_rel",       # レース内平均との差（そのレースでの相対位置）
    "ex_time_from_best", # レース内最速との差（重要度2.7%：ベスト差が主指標）
    "ex_time_rank",      # レース内順位（重要度2.5%：1が最速）
    "is_fastest_ex",     # 最速か否か（重要度0.2%：参考程度）
]
```

---

## 特徴量の重要度：何が本当に効いているか

LightGBMで計算した特徴量重要度（上位10個）：

1. **boat_number**　**39.8%**　— 枠番。圧倒的1位
2. **national_wr_rel**　10.4%　— 全国勝率偏差
3. **boat1_national_wr**　6.3%　— 1号艇の強さ
4. **racer_avg_st_90d**　4.4%　— スタート力
5. **motor_2rate_rel**　3.9%　— モーター偏差
6. **ex_time_from_best**　2.7%　— 展示タイム差
7. **ex_time_rank**　2.5%　— 展示タイム順位
8. **racer_boat_win_rate**　2.1%　— 枠適性
9. **class_num**　1.9%　— 選手級別
10. **stadium_num**　1.7%　— 会場

**驚くべきことに `boat_number`（枠番）が約40%を占める。**

これは競艇の本質を表している。コース（≒枠番）が勝負を決める競技なのだ。どんなに優れた選手でも、6号艇から1号艇のA1選手に勝つのは統計的に著しく難しい。モデルはこの現実を正確に学習している。

逆に言えば、「枠番の有利・不利」を差し引いた上で「どの選手が上振れするか」を見つけることがAIの仕事になる。

---

## 相対特徴量が重要な理由：絶対値より偏差

「全国勝率0.45」という数字単体は大して意味がない。そのレースで他の5艇と比べてどうか、が重要だ。

```python
def compute_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース内相対特徴量を計算する。
    これにより「このレースで飛び抜けて強い/弱い選手」を検出できる。
    """
    # レース内の平均勝率・2連率を計算
    race_avg_win = df.groupby("race_id")["racer_win_rate_90d"].transform("mean")
    race_avg_2rate = df.groupby("race_id")["racer_2rate_90d"].transform("mean")

    # 偏差（平均との差）を特徴量にする
    # 例: このレースの平均勝率が0.30のとき、0.45の選手は+0.15の偏差
    df["rel_win_rate_90d"] = df["racer_win_rate_90d"] - race_avg_win
    df["rel_2rate_90d"] = df["racer_2rate_90d"] - race_avg_2rate

    # 1号艇の勝率をレース全行に付与（全艇が「1号艇はどれほど強いか」を知る）
    boat1_stats = (
        df[df["boat_number"] == 1][["race_id", "racer_win_rate_90d"]]
        .rename(columns={"racer_win_rate_90d": "boat1_win_rate_90d"})
    )
    df = df.merge(boat1_stats, on="race_id", how="left")

    # 実力格差の指標（最高と最低の差が大きいほど実力差が明確）
    race_max = df.groupby("race_id")["racer_win_rate_90d"].transform("max")
    race_min = df.groupby("race_id")["racer_win_rate_90d"].transform("min")
    df["field_max_min_gap"] = race_max - race_min

    return df
```

この数行のコードを追加しただけで、モデルの精度が目に見えて改善した。

同様に、モーター2連率もレース内相対値にする。「モーター2連率45%」が良いかどうかは、対戦相手のモーターと比べないとわからない。全艇が45%なら差はゼロだが、他の5艇が30%台なら圧倒的に有利だ。

---

## 削除した特徴量とその理由：多重共線性との戦い

特徴量を増やせばいいというものではない。相関の高い特徴量を複数入れると、モデルが不安定になり過学習しやすくなる。これを**多重共線性**という。

### VIF（分散膨張因子）で多重共線性を定量的に検出する

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

def calc_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    VIF（分散膨張因子）を計算する。
    VIF > 10 は多重共線性あり。VIF > 20 は危険ゾーン。
    """
    # 欠損値を除外してからVIF計算
    X = df[features].dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(features))
    ]
    return vif_data.sort_values("VIF", ascending=False)
```

**削除した特徴量（VIF高すぎ）：**

- **`national_2rate`**（VIF **30.0**）: `national_win_rate` と強相関（どちらも選手の勝率）のため削除
- **`local_2rate`**（VIF 6.7）: `local_win_rate` と相関のため削除
- **`motor_3rate`**（VIF 5.0）: `motor_2rate` と相関のため削除

`national_2rate`（全国2連率）のVIF=30は相当高い。`national_win_rate`（全国勝率）と一緒に入れると、モデルが「どちらの特徴量を重視すべきか」で混乱する。片方を削除したら安定した。

---

## データリーク防止：shift(1)の意味

ローリング統計の計算で**絶対に守らなければならないルール**がある。

**当日のデータを当日の予測に使ってはいけない。**

```python
def compute_racer_rolling_stats(df: pd.DataFrame, windows: list[int] = [30, 90, 180]) -> pd.DataFrame:
    """
    選手ごとのローリング成績を計算する。
    重要: shift(1) を使って当日のデータを除外する（データリーク防止）
    """
    df = df.sort_values(["racer_number", "race_date", "race_number"])

    for w in windows:
        # 1着率: shift(1) で前日以前のデータのみを使用
        df[f"racer_win_rate_{w}d"] = (
            df.groupby("racer_number")["place"]
            .transform(
                # shift(1)で「1つ前」にずらしてから rolling を計算
                # これにより「当日の結果」が「当日の特徴量」に入ることを防ぐ
                lambda s: (s.shift(1) == 1).rolling(w, min_periods=5).mean()
            )
        )

        # 2連率
        df[f"racer_2rate_{w}d"] = (
            df.groupby("racer_number")["place"]
            .transform(lambda s: (s.shift(1) <= 2).rolling(w, min_periods=5).mean())
        )

        # 平均スタートタイミング
        df[f"racer_avg_st_{w}d"] = (
            df.groupby("racer_number")["start_timing"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=5).mean())
        )

    return df
```

`shift(1)` を忘れた私の最初のモデルは、ROI 148%という夢の数字を出した。

「答えを見ながら問題を解く」状態になっていたからだ。本番では答えは存在しない。実運用に移したら91%まで急落した。この話は記事06で詳しく書く。

---

## 枠適性の計算：選手×枠番の組み合わせ

「この選手は内枠が得意」「あの選手は外からでも突っ込んでくる」という傾向をデータ化する。

```python
def compute_racer_boat_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    選手×枠番の歴史的勝率を計算する（枠適性）。
    例: 登録番号4321の選手が2号艇で走った直近90レースでの勝率
    """
    df = df.sort_values(["racer_number", "race_date", "race_number"])

    df["_is_win"] = (df["place"] == 1).astype(int)

    df["racer_boat_win_rate"] = (
        df.groupby(["racer_number", "boat_number"])["_is_win"]
        .transform(
            # 選手×枠番の組み合わせごとに直近90レースの勝率を計算
            # min_periods=5: 最低5レース以上の実績がないとNaNを返す
            lambda s: s.shift(1).rolling(90, min_periods=5).mean()
        )
    )

    df = df.drop(columns=["_is_win"])
    return df
```

特徴量重要度では `racer_boat_win_rate` が8位（2.1%）だった。大きくはないが、安定して貢献している特徴量だ。

---

## 効かなかった特徴量

試して削除したもの：

- **天候・風速**（試した理由: 荒れた日は穴が出やすい？）: ROI変化なし。ノイズの方が大きい
- **波高**（試した理由: 荒水面での実力差？）: 効果なし
- **選手の年齢**（試した理由: 若手は外枠でも攻める？）: 効果なし
- **体重**（試した理由: 軽いほど有利？）: ほぼ効果なし
- **気温・水温**（試した理由: 機材コンディションへの影響？）: 効果なし

天候系の特徴量が全滅だったのは正直驚いた。「嵐の日は荒れる」という直感がモデルには全く通じなかった。これはたぶん、天候の影響が「全艇に均等にかかる」からだと思っている。均等に影響するものは、相対的な優劣を変えない。

---

## LightGBMに欠損値をそのまま渡す

一つ重要な実装のポイントがある。**LightGBMは欠損値（NaN）を自前で処理できる。**

```python
import lightgbm as lgb

# カテゴリ変数として扱う特徴量
cat_names = ["stadium_num"]

# 欠損値はNaNのまま渡す（中央値埋めはしない）
# LightGBMが「この特徴量が欠損している」という情報を活かして学習する
ds = lgb.Dataset(
    X,                           # 特徴量行列（NaN含んでOK）
    label=y,                     # ターゲット（1着=1, それ以外=0）
    feature_name=feature_cols,
    categorical_feature=cat_names,  # カテゴリ変数として扱う（ラベルエンコード不要）
)
```

欠損値を中央値で埋めると「データが欠けている」という情報が失われる。例えば出走表データがないレースは「出走表の特徴量が全部NaN」という状態だが、その「全部欠損」という事実自体がデータリーチの少ないレースである可能性を示している。LightGBMはこれを自動で学習してくれる。

---

## まとめ

特徴量エンジニアリングで最も効いたもの：

1. **レース内相対特徴量**（絶対値より「このレースでどれだけ違うか」）
2. **展示タイム関連**（`ex_time_from_best` が特に有効）
3. **枠適性**（選手×枠番の組み合わせ勝率）

逆に効かなかったもの：
- 天候・風速・波高（全艇に均等にかかるのでノイズになる）
- 体重・年齢（競艇での影響は統計的に微弱）

そして何より大事なのが **`shift(1)` によるデータリーク防止**だ。これを忘れると夢のようなバックテスト結果が出て、実運用で崩壊する。私が一度やらかした。

---

深夜に特徴量の相関行列を眺めながら飲む缶ビールは、なぜかひどく美味しい。

数字の中に構造が見える瞬間がある。「あ、これとこれは同じことを言っているんだ」と気づく瞬間。それが好きだ。ファンドにいたときも、競艇AIを作っている今も、それは変わらない。

---

*次回: LightGBM Binary + LambdaRank — なぜ2つのモデルを組み合わせるのか*
