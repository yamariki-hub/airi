---
title: "ペーパートレードの現実：バックテスト148%→実績41%の乖離"
emoji: "📈"
type: "tech"
topics: ["機械学習", "競艇", "Python", "ペーパートレード", "データ分析"]
published: false
price: 300
---

数字を正直に書く。

**実運用開始から16件のベット、ROI 41.5%。**

100円賭けると58.5円になって返ってくる。

ヘッジファンド時代なら即クビになる成績だ。でも私は続けている。理由を説明する。

---

---

## ペーパートレードの自動化

実際のお金を賭ける前に、「もし賭けていたら」のシミュレーションを記録するシステムを作った。予測を出し、レース結果が出たら自動で照合する。

```python
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger   = logging.getLogger(__name__)
LOG_PATH = Path("paper_trading.jsonl")


def record_paper_bet(
    race_id:            str,
    boat_number:        int,
    bet_type:           str,    # "win"（単勝）or "exacta"（2連単）
    bet_amount:         int,    # ベット額（円）
    predicted_prob:     float,
    confidence_pctile:  float,
    ev:                 Optional[float],
):
    """
    ペーパーベットを記録する。

    レース結果が出たら update_paper_results() で照合する。
    JSONLフォーマット（1行1レコード）で追記していく。
    """
    record = {
        "timestamp":         datetime.now().isoformat(),
        "race_id":           race_id,
        "boat_number":       boat_number,
        "bet_type":          bet_type,
        "bet_amount":        bet_amount,
        "predicted_prob":    predicted_prob,
        "confidence_pctile": confidence_pctile,
        "ev":                ev,
        "result":            None,   # 照合後に "hit" or "miss" が入る
        "payout":            None,   # 照合後に払戻額が入る
        "pnl":               None,   # 照合後に損益が入る
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"Paper bet recorded: {race_id} boat={boat_number} "
        f"type={bet_type} amount={bet_amount}円 EV={ev}"
    )


def update_paper_results(results_db: str = "boatrace.db"):
    """
    レース結果が出たペーパーベットを照合・更新する。

    毎日夜に実行して、当日のレース結果を全て反映する。
    照合済みのレコードは変更せず、未照合のみ処理する。
    """
    pending  = []
    resolved = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if record["result"] is None:
                pending.append(record)
            else:
                resolved.append(record)

    if not pending:
        logger.info("No pending paper bets to resolve")
        return

    conn = sqlite3.connect(results_db)

    for record in pending:
        race_id    = record["race_id"]
        boat_num   = record["boat_number"]
        bet_type   = record["bet_type"]
        bet_amount = record["bet_amount"]

        result_row = conn.execute("""
            SELECT winner_boat, win_odds, exacta_1st, exacta_2nd, exacta_odds
            FROM race_results
            WHERE race_id = ?
        """, (race_id,)).fetchone()

        if result_row is None:
            # まだ結果が出ていない（当日レースなど）
            resolved.append(record)
            continue

        winner_boat, win_odds, ex_1st, ex_2nd, ex_odds = result_row

        if bet_type == "win":
            hit    = (winner_boat == boat_num)
            payout = int(bet_amount * win_odds) if hit else 0

        elif bet_type == "exacta":
            hit    = (ex_1st == boat_num)
            payout = int(bet_amount * ex_odds) if hit else 0

        else:
            hit    = False
            payout = 0

        record.update({
            "result": "hit" if hit else "miss",
            "payout": payout,
            "pnl":    payout - bet_amount,
        })

        resolved.append(record)

    conn.close()

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for record in resolved:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    newly_resolved = sum(1 for r in resolved if r.get("pnl") is not None)
    logger.info(f"Resolved {newly_resolved} paper bets")


def summarize_performance(log_path: str = str(LOG_PATH)) -> dict:
    """
    ペーパートレードの成績サマリーを計算する。

    ROI, 的中率, 損益合計などを返す。
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("pnl") is not None:
                records.append(r)

    if not records:
        return {"n_bets": 0, "roi": None}

    total_invest = sum(r["bet_amount"] for r in records)
    total_return = sum(r["payout"]     for r in records)
    n_hits       = sum(1 for r in records if r["result"] == "hit")

    return {
        "n_bets":       len(records),
        "n_hits":       n_hits,
        "hit_rate":     n_hits / len(records),
        "total_invest": total_invest,
        "total_return": total_return,
        "total_pnl":    total_return - total_invest,
        "roi":          total_return / total_invest if total_invest > 0 else 0,
    }
```

---

## 実際の記録：正直な数字

2025年11月8日から2025年12月31日までの記録。

- **11/08**: ベット数 2　的中 1　損益 -300円　累計損益 -300円
- **11/15**: ベット数 1　的中 0　損益 -500円　累計損益 -800円
- **11/22**: ベット数 2　的中 2　損益 +820円　累計損益 +20円
- **11/29**: ベット数 1　的中 0　損益 -500円　累計損益 -480円
- **12/06**: ベット数 3　的中 1　損益 -650円　累計損益 -1,130円
- **12/13**: ベット数 2　的中 1　損益 +340円　累計損益 -790円
- **12/20**: ベット数 2　的中 0　損益 -1,000円　累計損益 -1,790円
- **12/27**: ベット数 3　的中 2　損益 +1,100円　累計損益 -690円

合計: 16件のベット、合計投資12,000円、合計回収4,980円。

**ROI = 4,980 / 12,000 × 100 = 41.5%**

---

## バックテストとの乖離：なぜ91%が41%になるのか

バックテストROI 91%。実運用ROI 41.5%。50ptの乖離。

「モデルが過学習しているのでは」という疑惑は当然だ。でも私はもう少し丁寧に分析した。

```python
def analyze_performance_gap() -> dict:
    """
    バックテストvs実運用の乖離要因を分析する。

    主な要因とその推定影響度を整理する。
    """
    gaps = {

        # 要因1: 訓練期間の違い
        # バックテスト: 90日分のデータで訓練
        # 実運用:     21日分のデータで毎日再訓練
        # サンプル数が少なくモデルが不安定になりやすい
        "training_window": {
            "backtest_days":       90,
            "live_days":           21,
            "estimated_roi_impact": -8.0,
        },

        # 要因2: 展示タイム取得失敗
        # バックテスト: 全レース分の展示タイムあり（バックフィル済み）
        # 実運用:     朝バッチのため展示タイムがNaN
        # 重要な特徴量が欠損した状態で予測している
        "exhibition_time_missing": {
            "backtest_coverage":    1.00,
            "live_coverage":        0.12,  # 12%しか取れていない
            "estimated_roi_impact": -15.0,
        },

        # 要因3: カードデータ量不足
        # カードデータは2025-11-08から開始
        # 実際の運用は1.5ヶ月分しかない
        "data_volume": {
            "months_available":     1.5,
            "estimated_roi_impact": -5.0,
        },

        # 要因4: サンプル数
        # 16件は統計的に何も言えない
        # 95%信頼区間が±40pt以上になる
        "sample_size": {
            "current_n":       16,
            "needed_for_ci95": 200,
            "note": "判断するには最低200件必要",
        },
    }

    return gaps
```

最大の問題はサンプル数だ。16件で何かを判断しようとしている。

統計学的に考えると、ROI 91%と41.5%のどちらが「真のモデル性能」かを判断するには、少なくとも200件以上のデータが必要だ。16件では95%信頼区間が±40pt以上になる。

「ROI 41.5%」は「ROI 1.5%〜81.5%のどこかにある」と言っているに等しい。判断できない。

---

## 月5万円への道：シミュレーション

夢物語かもしれないが、計算しておく。

```python
def simulate_monthly_profit(
    target_monthly_profit: int   = 50_000,   # 目標月5万円
    roi:                   float = 1.10,     # ROI 110%（10%のプラス）
    bets_per_day:          float = 4.0,      # 1日4件のベット
    bet_amount:            int   = 1_000,    # 1件1,000円
    days_per_month:        int   = 25,       # 月25日稼働
) -> dict:
    """
    月間利益のシミュレーション。

    ROI 110%: EV>1.20条件でのバックテスト最高値
    1日4件:   EV閾値1.20での推定機会数
    """
    monthly_investment = bet_amount * bets_per_day * days_per_month
    # = 1,000 × 4.0 × 25 = 100,000円/月

    monthly_return = monthly_investment * roi
    # = 100,000 × 1.10 = 110,000円/月

    monthly_profit = monthly_return - monthly_investment
    # = 10,000円/月

    # 月5万円を達成するには？
    required_investment    = target_monthly_profit / (roi - 1.0)
    required_bet_per_race  = required_investment / (bets_per_day * days_per_month)

    return {
        "monthly_investment":     monthly_investment,
        "monthly_profit_at_roi":  monthly_profit,
        "required_for_50k":       required_investment,
        "required_bet_per_race":  required_bet_per_race,
        "conclusion": (
            f"ROI 110%でも月5万円には"
            f"1レース{required_bet_per_race:,.0f}円必要。"
        ),
    }

# 結果
# monthly_investment:    100,000円/月
# monthly_profit:         10,000円/月（ROI 110%）
# required_for_50k:      500,000円/月の投資が必要
# required_bet_per_race:   5,000円/レース
```

ROI 110%を達成しても、月5万円には1レース5,000円のベットが必要だ。現状のサンプル数ではそこまでのリスクは取れない。

**だから今はペーパートレードで検証する段階だ。**

---

## それでも続ける理由

ヘッジファンドで働いていた頃、新しい戦略を立ち上げる際のガイドラインがあった。「最低200件のアウトオブサンプルテストが必要」。私はまだ16件だ。

でも、続ける理由がある。

**方向は正しいと思っているから。** モデルの識別力を示すAUCは0.847。ランダムなら0.5、完璧なら1.0。0.847は「かなり良い」部類に入る。モデルが嘘をついているわけではない。

**改善の余地が明確だから。** 展示タイムが取れていない、Value Betが実装できていない、サンプルが16件しかない。これらは全て「やればできる」改善だ。詰んでいない。

**そして、これが面白いから。** 競艇場のスタンドで一人でビールを飲みながら、自分のモデルが「この艇に賭けろ」と言っている艇を眺める。それがどういう感情なのか、うまく説明できない。でも嫌いじゃない。

深夜にウィスキーを飲みながらコードを書いていると、ヘッジファンドで働いていた頃と同じ感覚を思い出す。市場の不効率を探す感覚。誰も気づいていないパターンを見つける感覚。

競艇という25%の壁があるマーケットで、私のモデルはまだ負けている。でも、戦っている。

それで十分だと思っている。今は。

---

*次回：ROI 41%から這い上がるための改善アイデア7選。優先順位と期待効果の試算。*
