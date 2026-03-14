---
title: "展示タイムの使い方：レース直前の最強シグナル"
emoji: "⏱️"
type: "tech"
topics: ["機械学習", "競艇", "Python", "データ分析", "展示タイム"]
published: false
price: 500
---

「展示タイムが速い艇が勝つ」

競艇ファンなら誰でも知っている経験則だ。私はそれを信じて、AIにも組み込んだ。そして、半年後に打ち砕かれた。

---

---

## 展示航走とは何か

レース本番の約2時間前、選手たちは本コースで試走を行う。これが「展示航走」だ。

本番と同じ水面、同じモーター、同じボートで走るタイムが計測される。一般的なタイムは6.50〜7.20秒の範囲に収まる。0.1秒の差が、そのまま実力差になる世界だ。

選手はこのタイムを見てモーターの調子を確認し、整備を行う。観客もこのタイムを見て最終判断をする。そして私のAIも、このタイムを特徴量として取り込もうとした。

問題は「どう取り込むか」だった。

---

## Playwright必須：JavaScriptの壁

展示タイムはboatrace.jpの特定ページに掲載されるが、このページがJavaScriptで動的にレンダリングされる。requestsで取得しても、肝心のタイムデータが空だ。

```
<div id="exhibition-time" data-loaded="false">
  <!-- JavaScriptが実行されてから表示 -->
</div>
```

ここで`requests + BeautifulSoup`は無力だ。Playwrightが必要になる。

```python
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def fetch_exhibition_times(
    jcd: str,           # 会場コード（例: "01" = 桐生）
    hd:  str,           # 日付（例: "20251108"）
    rno: str,           # レース番号（例: "01"〜"12"）
    timeout_ms: int = 20000,  # タイムアウト20秒
) -> Optional[list[dict]]:
    """
    Playwrightを使って展示タイムを動的取得する。

    JavaScriptレンダリングが完了するまで待機してからスクレイピング。
    ヘッドレスモード（画面なし）で実行するためCIでも動く。
    """
    url = (
        f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
        f"?jcd={jcd}&hd={hd}&rno={rno}"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        try:
            page.goto(url, timeout=timeout_ms)

            # 展示タイムのテーブルが読み込まれるまで待機
            # boatrace.jp は非同期でデータを流し込む設計なので要待機
            page.wait_for_selector(
                "table.is-w495",
                timeout=timeout_ms
            )

            # DOM安定化を待つ
            page.wait_for_load_state("networkidle", timeout=timeout_ms)

            content = page.content()
            return _parse_exhibition_html(content, jcd, hd, rno)

        except PWTimeout:
            logger.warning(
                f"Timeout fetching exhibition times: "
                f"jcd={jcd}, hd={hd}, rno={rno}"
            )
            return None

        except Exception as e:
            logger.error(f"Error: {e}")
            return None

        finally:
            browser.close()


def _parse_exhibition_html(
    html: str,
    jcd:  str,
    hd:   str,
    rno:  str,
) -> list[dict]:
    """展示情報HTMLをパースして各艇のタイムを抽出する。"""
    from bs4 import BeautifulSoup

    soup    = BeautifulSoup(html, "html.parser")
    race_id = f"{jcd}{hd}{rno.zfill(2)}"

    records = []
    table   = soup.find("table", class_="is-w495")
    if not table:
        logger.warning(f"Exhibition table not found for {race_id}")
        return []

    for row in table.find_all("tr")[1:]:  # ヘッダー行をスキップ
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        try:
            boat_num = int(cells[0].text.strip())
            ex_time  = float(cells[1].text.strip())
            records.append({
                "race_id":     race_id,
                "boat_number": boat_num,
                "ex_time":     ex_time,
            })
        except (ValueError, IndexError):
            continue

    return records
```

---

## 特徴量設計：「最速フラグ」だけでは足りない

展示タイムが取れたところで、どう特徴量にするかが問題だ。最初に思いつくのは「最速フラグ」。

```python
import pandas as pd
import numpy as np

def engineer_exhibition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    展示タイム特徴量を設計する。

    複数の特徴量を用意して、後でモデルに何が効くか確認する。
    「最速フラグ」一本槍は危険、というのが後の教訓になる。
    """
    grp = df.groupby("race_id")

    # 特徴量1: レース内相対値（自分 - レース平均。小さいほど速い）
    df["ex_time_rel"] = df["ex_time"] - grp["ex_time"].transform("mean")

    # 特徴量2: 最速からの差（0なら最速、値が大きいほど遅い）
    df["ex_time_from_best"] = (
        df["ex_time"] - grp["ex_time"].transform("min")
    )

    # 特徴量3: 順位（1〜6。タイムが速い順）
    df["ex_time_rank"] = grp["ex_time"].rank(method="min", ascending=True)

    # 特徴量4: 最速フラグ（1=最速、0=それ以外）
    # ← これが後に問題になる
    df["is_fastest_ex"] = (df["ex_time_from_best"] == 0.0).astype(int)

    # 特徴量5: Zスコア（レース内標準化。負の値ほど速い）
    df["ex_time_zscore"] = (
        (df["ex_time"] - grp["ex_time"].transform("mean"))
        / grp["ex_time"].transform("std").clip(lower=0.001)
    )

    return df
```

4種類の特徴量を用意した。どれが効くかは、後でモデルに聞けばいい。それが機械学習というものだ。

---

## 衝撃の特徴量重要度：is_fastest_exは死んでいた

4ヶ月のデータでモデルを訓練し、LightGBMの特徴量重要度を確認した。

深夜1時、ウィスキーを飲みながら画面を見ていた。

- **`boat_number`**　39.8%　— 1位
- **`nwr_rel`**　10.4%　— 2位
- **`ex_time_from_best`**　2.7%　— 8位
- **`ex_time_rank`**　2.5%　— 9位
- **`ex_time_rel`**　1.8%　— 11位
- **`is_fastest_ex`**　**0.2%**　— **最下位**

**is_fastest_ex、重要度0.2%。**

私はしばらく画面を見つめた。グラスを置いた。また見た。

「……なんで」

最速フラグが最弱の特徴量だった。v2モデルではこれをメインフィルタとして使っていた。「最速展示タイムの艇にだけベットする」という戦略で。

完全な間違いだった。

---

## なぜ最速フラグは弱いのか

少し冷静になって考えた。

`is_fastest_ex`は二値変数だ。0か1しかない。しかも6艇のうち必ず1艇だけが1になる。情報量が極めて少ない。

一方、`ex_time_from_best`は連続値だ。最速から0.02秒差なのか、0.30秒差なのかが分かる。この「差の大きさ」こそが重要な情報だった。

競艇でよくあるケース：

```
1号艇の展示タイム: 6.80秒（最速）
2号艇の展示タイム: 6.82秒（最速から0.02秒差）
6号艇の展示タイム: 7.10秒（最速から0.30秒差）
```

最速フラグは1号艇だけに1を立てるが、2号艇との差はほぼゼロだ。一方、6号艇は明確に遅い。この差異を`is_fastest_ex`は全く表現できない。

情報量の問題だ。バイナリよりも連続値の方が、モデルに渡せる情報量が多い。当たり前のことを、データから学んだ。

---

## フィルタ条件の改善：v2 → v3

この発見を受けて、ベット条件を変えた。

```python
# ===== v2 の条件（間違っていた）=====
def should_bet_v2(forecast: dict) -> bool:
    """
    v2: 最速展示タイムの艇にだけベット。
    問題: is_fastest_ex はバイナリで情報量が極めて少ない。
    """
    return (
        forecast["conf_pctile"] >= 95       # 信頼度上位5%
        and forecast["is_fastest_ex"] == 1  # 最速フラグ（← これが問題）
    )


# ===== v3 の条件（改善後）=====
def should_bet_v3(forecast: dict) -> bool:
    """
    v3: 展示タイムが最速から0.10秒以内かつ順位2位以内の艇にベット。

    改善点:
    1. 最速フラグ（バイナリ）→ 連続値での条件に変更
    2. 0.10秒以内: 明確に遅い艇を除外しつつ、準最速も含める
    3. 順位2位以内: タイム差が小さくても3番手以降は除外
    """
    return (
        forecast["conf_pctile"] >= 95               # 信頼度上位5%
        and forecast["ex_time_from_best"] <= 0.10   # 最速から0.10秒以内
        and forecast["ex_time_rank"] <= 2            # 順位2位以内
    )
```

この変更でベット対象が増えた。v2では「1艇しかベットしない」ケースが多かったが、v3では「1〜2艇にベット」できるケースが増えた。バックテストでの予測精度は維持しつつ、機会損失が減った。

---

## 展示タイムの実際の分布

手元のデータから集計した展示タイムの統計：

- **平均**: 6.748秒
- **標準偏差**: 0.098秒
- **最速（min）**: 6.490秒
- **最遅（max）**: 7.350秒
- **最速から0.10秒以内の割合**: 35.2%（6艇中約2.1艇）

0.10秒という閾値は、全体の35%の艇が該当する。1レースあたり平均2.1艇。ちょうどいい絞り込みだ。

---

## タイミング問題：朝に全レース分は取れない

実運用上の根本問題がある。

展示航走は本番2時間前に行われる。私の現在のシステムは朝7時に当日の予測を生成する。展示タイムが出るのは各レースの2時間前だから、朝7時時点では1Rの展示タイムすら出ていない。

```python
# 現状の実装（問題あり）
def morning_batch():
    today = datetime.now().strftime("%Y%m%d")
    # 朝7時時点では展示タイムなし → ex_time_from_best が NaN になる
    # NaNのまま予測 → 展示タイムの恩恵が受けられない
    fetch_all_exhibition_times(today)

# 理想の実装（未実装）
# 各レースの開始2時間前に動的にfetchする常駐プロセスが必要
# 現在の朝バッチアーキテクチャとは設計が根本的に異なる
```

この問題を解決するには、常駐プロセスを立てて各レース開始の2時間前に動的にfetchする仕組みが必要だ。これはv4で取り組む予定の課題だ。

現状、展示タイムの取得率は12%程度。バックテストと実運用のROI乖離の主因の一つがここにある。

---

競艇場のスタンドで一人でビールを飲みながら、展示航走を生で見たことがある。モーターの音が違う。明らかに軽快な音の艇と、重そうな音の艇がある。その「音」を数値化できないかと、ふと思う。

無理な話だが、思ってしまう。それが私というものだ。

---

*次回：Value Bet理論。「確率 × オッズ > 1.0」という単純な式が、なぜこんなに実装が難しいのか。朝のオッズと締切直前のオッズが全く違う問題について。*
