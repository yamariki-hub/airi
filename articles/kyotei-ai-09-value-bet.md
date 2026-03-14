---
title: "Value Bet理論：勝率より「期待値」を追う"
emoji: "💰"
type: "tech"
topics: ["機械学習", "競艇", "Python", "期待値", "最適化"]
published: false
price: 500
---

外資系ヘッジファンドにいた頃、私はリスクプレミアムというものを毎日計算していた。市場が過小評価している資産を買い、過大評価している資産を売る。それだけだ。

競艇に置き換えると、同じロジックが使える。

---

---

## 期待値（EV）の基本式

```
EV = 予測勝利確率 × 払戻オッズ
```

EVが1.0を超えれば、長期的にプラスになる賭けだ。

具体例：

```
1号艇の予測勝利確率: 45%
1号艇の単勝オッズ:  2.5倍

EV = 0.45 × 2.5 = 1.125  → ベットすべき
```

逆に：

```
1号艇の予測勝利確率: 45%
1号艇の単勝オッズ:  1.8倍

EV = 0.45 × 1.8 = 0.81   → ベットしてはいけない
```

単純だ。でも実装は単純ではない。

---

## 競艇の控除率25%という壁

競艇の払戻金は、集めた賭け金の75%だ。残り25%が胴元の取り分になる。

これが意味するのは：**市場（他の観客全員）より33%以上賢くないと、長期的に勝てない。**

数学的に示すと：

```
期待オッズ = 1 / (予測確率) × (1 - 控除率)
           = 1 / 0.45 × 0.75
           = 1.667倍

→ 実際のオッズが1.667倍より高い → 買い
→ 実際のオッズが1.667倍より低い → 見送り
```

市場は集合知だ。何千人もの観客が「このレースはこの艇が強い」と判断してお金を投じる。その集合知に対して、私のモデルが優れた判断をできているか。

ROI 41.5%の現状を見ると、答えは「まだできていない」だ。

---

## EV計算の実装

```python
import sqlite3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# EV閾値の設定
# 1.0: プラス期待値のみ（理論値、実際は控除率で厳しい）
# 1.10: 10%のマージン
# 1.20: 20%のマージン（現在の設定）← 推奨
# 1.30: 厳しすぎてベット機会が激減
EV_THRESHOLD = 1.20


def calculate_ev(
    predicted_prob: float,
    current_odds:   float,
) -> float:
    """
    期待値を計算する。

    EV = 予測確率 × 払戻オッズ
    EV > 1.0: 理論的にプラス期待値
    EV > 1.20: 控除率と予測誤差を考慮したマージンあり
    """
    if predicted_prob <= 0 or current_odds <= 0:
        return 0.0
    return predicted_prob * current_odds


def evaluate_bet(
    race_id:        str,
    boat_number:    int,
    predicted_prob: float,
    db_path:        str = "boatrace.db",
) -> dict:
    """
    現在のオッズを取得してEVを計算し、ベット判断を返す。

    オッズが取得できない場合はEV=Noneを返す（ベット見送り）。
    オッズが30分以上古い場合も見送り（締切直前の動きに追従できない）。
    """
    conn = sqlite3.connect(db_path)
    row  = conn.execute("""
        SELECT win_odds, fetched_at
        FROM live_odds
        WHERE race_id = ? AND boat_number = ?
        ORDER BY fetched_at DESC
        LIMIT 1
    """, (race_id, boat_number)).fetchone()
    conn.close()

    if row is None or row[0] is None:
        logger.warning(
            f"No odds for race_id={race_id}, boat={boat_number}"
        )
        return {
            "race_id":        race_id,
            "boat_number":    boat_number,
            "predicted_prob": predicted_prob,
            "current_odds":   None,
            "ev":             None,
            "should_bet":     False,
            "reason":         "odds_unavailable",
        }

    current_odds = row[0]
    fetched_at   = row[1]
    ev           = calculate_ev(predicted_prob, current_odds)

    # オッズの新鮮さチェック（30分以上古いオッズは信頼しない）
    fetch_time  = datetime.fromisoformat(fetched_at)
    age_minutes = (datetime.now() - fetch_time).seconds / 60

    if age_minutes > 30:
        logger.warning(
            f"Stale odds ({age_minutes:.0f} min old) for {race_id}"
        )

    should_bet = (ev >= EV_THRESHOLD) and (age_minutes <= 30)

    return {
        "race_id":        race_id,
        "boat_number":    boat_number,
        "predicted_prob": predicted_prob,
        "current_odds":   current_odds,
        "ev":             ev,
        "should_bet":     should_bet,
        "odds_age_min":   age_minutes,
        "reason":         "ev_above_threshold" if should_bet else "ev_below_threshold",
    }
```

---

## オッズ取得の実装：watch_bets.py

オッズは時間とともに変動する。締切5分前と締切直前では全く違うオッズになることもある。だから5分ごとにオッズを再取得して、EVを再計算する監視スクリプトが必要だ。

```python
import time
from playwright.sync_api import sync_playwright

def fetch_live_odds(race_id: str) -> list[dict]:
    """
    リアルタイムオッズを取得する。

    boatrace.jp のオッズページはJavaScript動的レンダリングのため
    Playwright必須。requestsでは「---」しか表示されない。
    """
    jcd = race_id[:2]
    hd  = race_id[2:10]
    rno = race_id[10:]

    url = (
        f"https://www.boatrace.jp/owpc/pc/race/oddstf"
        f"?jcd={jcd}&hd={hd}&rno={rno}"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        try:
            page.goto(url, timeout=30000)
            page.wait_for_selector("table.is-w650", timeout=15000)
            html = page.content()
            return _parse_odds_html(html, race_id)

        except Exception as e:
            logger.error(f"Failed to fetch odds for {race_id}: {e}")
            return []

        finally:
            browser.close()


def watch_bets_loop(active_races: list[str], interval_sec: int = 300):
    """
    アクティブなレースのオッズを5分ごとに監視する。

    EVが閾値を超えたらログに記録（実際の発注は手動で判断）。

    active_races: 当日のベット候補レースID一覧
    interval_sec: 更新間隔（デフォルト5分=300秒）
    """
    logger.info(f"Starting odds watcher for {len(active_races)} races")

    while active_races:
        for race_id in active_races[:]:
            odds_list = fetch_live_odds(race_id)

            if not odds_list:
                continue

            save_odds_to_db(odds_list)

            for entry in odds_list:
                boat_num  = entry["boat_number"]
                predicted = get_predicted_prob(race_id, boat_num)

                if predicted is None:
                    continue

                ev = calculate_ev(predicted, entry["win_odds"])

                if ev >= EV_THRESHOLD:
                    logger.info(
                        f"[ALERT] EV={ev:.3f} | "
                        f"race={race_id} boat={boat_num} | "
                        f"prob={predicted:.3f} odds={entry['win_odds']}"
                    )

        time.sleep(interval_sec)
```

---

## EV閾値の比較：どこに設定すべきか

閾値を上げると「確実なバリューベットのみ」を狙えるが、機会が激減する。

- **EV閾値 1.00**: ベット数/日 12.3件　ROI 88.4%　— 控除率に負けている
- **EV閾値 1.10**: ベット数/日 7.8件　ROI 95.2%　— まだ赤字
- **EV閾値 1.20**: ベット数/日 4.1件　ROI 103.7%　— ギリギリプラス
- **EV閾値 1.30**: ベット数/日 1.8件　ROI 108.2%　— 機会が少なすぎる

EV_THRESHOLD=1.20はこの分析が根拠だ。バックテストで唯一プラスになる閾値であり、かつ1日4件程度のベット機会がある。

ただしこれはバックテストの話だ。実運用では別の話になる。

---

## Kelly基準との組み合わせ

いくらベットするか、という問題にKelly基準が答えを出してくれる。

```python
def kelly_bet_size(
    predicted_prob: float,
    odds:           float,
    bank_roll:      float,
    kelly_fraction: float = 0.25,  # Quarter Kelly（保守的設定）
) -> float:
    """
    Kelly基準でベット額を計算する。

    Kelly fraction = (p * b - q) / b
    p: 勝つ確率
    q: 負ける確率（= 1 - p）
    b: オッズ - 1（純利益率）

    全額Kelly（fraction=1.0）は理論上最適だが、
    モデルの予測誤差を考慮してQuarter Kellyを推奨。
    """
    p = predicted_prob
    q = 1 - p
    b = odds - 1.0  # 純利益率

    if b <= 0:
        return 0.0

    full_kelly = (p * b - q) / b

    if full_kelly <= 0:
        return 0.0  # 期待値がマイナス、賭けない

    # Quarter Kellyで保守的に
    bet_fraction = full_kelly * kelly_fraction

    # 最大ベット額は総資産の5%（リスク管理）
    max_bet = bank_roll * 0.05

    return min(bank_roll * bet_fraction, max_bet)


# 使用例
# EV=1.125, prob=0.45, odds=2.5, 資金10万円
# Kelly = (0.45 × 1.5 - 0.55) / 1.5 = 0.083
# Quarter Kelly = 0.021
# ベット額 = 100,000 × 0.021 = 2,083円
```

---

## 現状の問題：win_odds = None

2025年12月から実際にwatch_bets.pyを動かしているが、深刻な問題がある。

```
[WARNING] No odds for race_id=1712251201, boat=1 - reason: odds_unavailable
[WARNING] No odds for race_id=1712251201, boat=2 - reason: odds_unavailable
[WARNING] Timeout fetching odds from boatrace.jp (30s)
```

boatrace.jpへのアクセスが頻繁にタイムアウトする。5分ごとのリクエストを12場分送ると、サーバーに負荷がかかるのか、応答が遅くなる。

データが取れないのでEVが計算できない。EVが計算できないのでベット判断ができない。ベット判断ができないのでROI 41.5%の改善もできない。

「データが全て」という言葉が身に染みる。データが取れなければ何も始まらない。

居酒屋でハイボールを飲みながら、このログを眺めていた。同席していた友人に「何の暗号？」と聞かれた。「収支管理」と答えた。嘘はついていない。

---

*次回：ROI 41.5%の現実。ペーパートレードの自動化と、バックテストとの乖離がなぜ起きるのかの考察。*
