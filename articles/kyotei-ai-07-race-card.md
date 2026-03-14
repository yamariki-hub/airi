---
title: "出走表データの威力：当日朝の情報がモデルを変える"
emoji: "📋"
type: "tech"
topics: ["機械学習", "競艇", "Python", "データ分析", "特徴量"]
published: false
price: 300
---

私が競艇AIの開発を始めてから、最初の壁にぶつかったのは「データが足りない」という単純な事実だった。

2025年の夏、v2モデルのROIは91%。聞こえはいいが、要するに100円賭けると91円しか返ってこない。9円の赤字だ。競艇の控除率は25%だから、還元率75%の市場でROI91%なら「まだマシ」という話ではある。でも私は負けている。それが全てだった。

深夜2時、ウィスキーのグラスを傾けながらJupyter Notebookを眺めていた。使っていた特徴量は、過去レースの成績集計だけ。選手の通算勝率、モーター勝率、コース別成績。そういったものを集めてLightGBMに食わせていた。

「これは過去の話しか見ていない。当日の情報がない」

当たり前のことに、3ヶ月かかって気づいた。

---

---

## 出走表という金鉱

競艇には「カード」と呼ばれるデータが存在する。正式には出走表。各レースの本番前日〜当日に公開される、いわゆる「今日の選手の状態」を示すデータだ。

中身を見ると、こんな情報が詰まっている：

**選手情報**
- 氏名・登録番号・支部・年齢・体重
- 級別（A1/A2/B1/B2）
- 全国勝率・全国2連率・全国3連率
- 当地勝率・当地2連率
- 直近スタートタイミング（ST）
- フライング・出遅れ回数（FL/L）

**機材情報**
- モーター番号・モーター2連率
- ボート番号・ボート2連率

これをスクレイピングで取れるようになった瞬間、私のモデルは変わった。

---

## スクレイピング実装：requests + BeautifulSoup

まずシンプルな実装から。カードデータはJavaScriptレンダリングが不要なページも多いので、requestsで十分なケースが多い。

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.boatrace.jp/owpc/pc/race/racelist"

def fetch_card_data(race_id: str, retries: int = 3) -> Optional[dict]:
    """
    出走表データを取得する。

    race_id: "202501011201" 形式（場所コード+日付+レース番号）
    retries: タイムアウト時のリトライ回数
    """
    # 場所・日付・レース番号を分解
    jcd = race_id[:2]   # 会場コード（01=桐生, 17=宮島など）
    hd  = race_id[2:10] # 日付（YYYYMMDD）
    rno = race_id[10:]  # レース番号（01〜12）

    params = {
        "jcd": jcd,
        "hd":  hd,
        "rno": rno,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(
                BASE_URL,
                params=params,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            return _parse_card_html(resp.text, race_id)

        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1} for {race_id}")
            time.sleep(2 ** attempt)  # 指数バックオフ（1秒, 2秒, 4秒）

        except Exception as e:
            logger.error(f"Error fetching {race_id}: {e}")
            return None

    return None


def _parse_card_html(html: str, race_id: str) -> dict:
    """HTMLをパースして選手・機材データを抽出する。"""
    soup = BeautifulSoup(html, "html.parser")

    records = []
    # 出走表テーブルの各行（6艇分）
    for row in soup.select("table.is-w748 tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        boat_num = int(cells[0].text.strip())

        record = {
            "race_id":        race_id,
            "boat_number":    boat_num,
            "racer_name":     cells[2].text.strip(),
            "racer_class":    cells[3].text.strip(),  # A1/A2/B1/B2
            "national_wr":    _to_float(cells[4].text),  # 全国勝率
            "national_2rate": _to_float(cells[5].text),  # 全国2連率
            "local_wr":       _to_float(cells[6].text),  # 当地勝率
            "local_2rate":    _to_float(cells[7].text),  # 当地2連率
            "motor_2rate":    _to_float(cells[8].text),  # モーター2連率
            "boat_2rate":     _to_float(cells[9].text),  # ボート2連率
            "avg_st":         _to_float(cells[10].text), # 平均スタートタイミング
        }
        records.append(record)

    return {"race_id": race_id, "entries": records}


def _to_float(text: str) -> Optional[float]:
    """テキストを浮動小数点数に変換。変換不可なら None を返す。"""
    try:
        return float(text.strip().replace("-", ""))
    except (ValueError, AttributeError):
        return None
```

---

## 並列取得：ThreadPoolExecutorで爆速スクレイピング

1レースずつ取得していたら日が暮れる。1日12場×12レース＝144レース分を取得するのに、シングルスレッドだと30分かかる。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3

def fetch_cards_bulk(race_ids: list[str], max_workers: int = 10) -> list[dict]:
    """
    複数レースのカードデータを並列取得する。

    max_workers=10 は boatrace.jp への過負荷を避けるための上限。
    それ以上増やすとIPブロックリスクがある（経験済み）。
    """
    results = []
    failed  = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(fetch_card_data, rid): rid
            for rid in race_ids
        }

        for future in as_completed(future_to_id):
            rid = future_to_id[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
                else:
                    failed.append(rid)
            except Exception as e:
                logger.error(f"Unexpected error for {rid}: {e}")
                failed.append(rid)

        time.sleep(0.1)  # リクエスト間隔を設けて礼儀正しく

    if failed:
        logger.warning(f"Failed to fetch {len(failed)} races: {failed[:5]}...")

    logger.info(f"Fetched {len(results)}/{len(race_ids)} races successfully")
    return results


def save_cards_to_db(records: list[dict], db_path: str = "boatrace.db"):
    """取得したカードデータをSQLiteに保存する。"""
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS race_cards (
            race_id        TEXT,
            boat_number    INTEGER,
            racer_name     TEXT,
            racer_class    TEXT,
            national_wr    REAL,
            national_2rate REAL,
            local_wr       REAL,
            local_2rate    REAL,
            motor_2rate    REAL,
            boat_2rate     REAL,
            avg_st         REAL,
            fetched_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (race_id, boat_number)
        )
    """)

    for card in records:
        for entry in card.get("entries", []):
            conn.execute("""
                INSERT OR IGNORE INTO race_cards
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, tuple(entry.values()))

    conn.commit()
    conn.close()
```

実際にこれを動かして4ヶ月分のデータを収集した。144,000件超の選手エントリーデータ。深夜3時にスクリプトを走らせて、朝起きたらデータベースが育っていた。地味に嬉しい瞬間だった。

---

## カード特徴量の設計：相対評価が鍵

「全国勝率6.50」という数字は、そのままでは弱い特徴量だ。なぜなら、そのレースに出てくる他の選手も全員それなりの勝率を持っているからだ。

重要なのは**レース内の相対評価**だ。

```python
import pandas as pd
import numpy as np

def engineer_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    カード特徴量を設計する。

    絶対値ではなく、レース内での相対値を使うことが鍵。
    「このレースで一番強い選手か？」を数値化する。
    """
    grp = df.groupby("race_id")

    # --- 選手能力の相対評価 ---
    # 全国勝率のレース内相対値（自分 - レース平均）
    df["nwr_rel"] = df["national_wr"] - grp["national_wr"].transform("mean")

    # 1号艇の全国勝率（コース有利を考慮した特徴量）
    boat1_nwr = grp.apply(
        lambda g: g.loc[g["boat_number"] == 1, "national_wr"].values[0]
        if (g["boat_number"] == 1).any() else np.nan
    )
    df["boat1_nwr"] = df["race_id"].map(boat1_nwr)

    # --- モーター・ボート機材の相対評価 ---
    # モーター2連率のレース内相対値
    # （抽選で割り当てられるため、レース内の相対差が本質的な情報）
    df["motor_2rate_rel"] = (
        df["motor_2rate"] - grp["motor_2rate"].transform("mean")
    )

    # ボート2連率の相対値
    df["boat_2rate_rel"] = (
        df["boat_2rate"] - grp["boat_2rate"].transform("mean")
    )

    # --- ST関連 ---
    df["st_rel"] = df["avg_st"] - grp["avg_st"].transform("mean")

    # --- 級別エンコーディング ---
    class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    df["racer_class_num"] = df["racer_class"].map(class_map).fillna(1)
    df["class_rel"] = (
        df["racer_class_num"] - grp["racer_class_num"].transform("mean")
    )

    return df
```

---

## 特徴量重要度の実際：何が効いているのか

4ヶ月のデータでLightGBMを訓練した結果、特徴量重要度（gainベース）はこうなった：

- **`boat_number`**　39.8%　— コース有利性（1号艇が圧倒的）
- **`nwr_rel`**　10.4%　— 全国勝率の相対値
- **`boat1_nwr`**　6.3%　— 1号艇選手の実力
- **`racer_st_90d`**　4.4%　— 直近90日のST平均
- **`motor_2rate_rel`**　3.9%　— モーター性能の相対値
- **`national_2rate`**　※削除　— VIF=30（後述）

`boat_number`が39.8%というのは衝撃的だった。モデルの4割近くが「どのコースか」だけで決まっている。競艇が「1号艇有利のスポーツ」というのは誰でも知っているが、数字で見るとここまでとは。

---

## 多重共線性との格闘：VIF分析

`national_wr`（全国勝率）と`national_2rate`（全国2連率）はほぼ同じ情報を持っている。両方入れるとモデルが混乱する。

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Variance Inflation Factor（分散膨張係数）で多重共線性を検出する。

    VIF > 10: 問題あり
    VIF > 30: 深刻（即削除を検討）
    """
    X = df[features].dropna()

    vif_data = pd.DataFrame({
        "feature": features,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(len(features))
        ]
    }).sort_values("VIF", ascending=False)

    return vif_data

# 実際の分析結果（抜粋）
# feature              VIF
# national_2rate       30.2   <- 削除対象
# national_wr          18.5   <- 要注意
# motor_2rate_rel       2.1   <- OK
# nwr_rel               1.8   <- OK
```

`national_2rate`のVIFが30.2。`national_wr`とほぼ完全に相関している。削除したら予測安定性が上がった。

---

## カバレッジ問題：過去データは後から取れない

カードデータは**当日〜前日のみ**公開される。2025年11月にスクレイピングを始めた時点で、それ以前のレースのカードデータは存在しない。

2025年7月〜10月の約4ヶ月分のレース結果はあるが、カード情報は欠損している。

「もっと早く始めていれば」と深夜に思った。でも、始めた日が一番早い。

---

## 結果：バックテストROIが8ポイント改善

カード特徴量を導入した後、バックテストでROIが91%から99.2%まで改善した。まだ赤字だ。でも方向は正しい。

居酒屋でビールを飲みながら、このグラフを見ていた。一人で。隣のサラリーマンが競馬の話をしているのを横目に、私は競艇のモーター2連率の分布を眺めていた。

それが私の休日だ。後悔はない。

---

*次回：展示タイムという「レース直前の最強シグナル」をどう使うか。Playwrightでの動的スクレイピングと、is_fastest_exが実はほぼ無意味だったという苦い発見について。*
