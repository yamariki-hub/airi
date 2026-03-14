---
title: "ROIを上げる7つのアイデア：今すぐ試せる改善策"
emoji: "💡"
type: "tech"
topics: ["機械学習", "競艇", "Python", "ROI", "最適化"]
published: false
price: 300
---

2025年の年末、ペーパートレードのROI 41.5%という数字を見ながら、私は一人で居酒屋にいた。

カウンター席でハイボールを飲みながら、ノートに問題を書き出した。

**問題リスト（酔いながら書いたもの）**
1. EVがNone（オッズ取れてない）
2. 展示タイムがほぼNaN（朝取得だから）
3. サンプル16件（統計的に何も言えない）
4. 低配当艇に集中（コース有利）

翌朝、同じノートを見て「全部的確だな」と思った。酔っていても分析力は落ちないらしい。

というわけで、現状の問題を解決するための改善アイデア7つ。優先度順に書く。

---

---

## アイデア1: Value Bet完全実装

**期待効果: +10〜15pt**
**難易度: 高**
**優先度: 最高**

現在の問題：オッズが取れていないので、EVが計算できない。EVが計算できないので、「割安なベット」を選べない。

解決策：Playwrightで非同期にリアルタイムオッズを監視する常駐プロセスを作る。

```python
import asyncio
from playwright.async_api import async_playwright
import sqlite3
from datetime import datetime

class OddsWatcher:
    """
    競艇オッズをリアルタイム監視するクラス。

    設計思想:
    - 非同期処理（asyncio + Playwright）で複数場を並列監視
    - 5分ごとにDBへ保存
    - タイムアウト時は3回リトライ後にスキップ
    """

    def __init__(self, db_path: str = "boatrace.db"):
        self.db_path     = db_path
        self.active_races: dict[str, datetime] = {}

    async def watch_all_races(self, race_ids: list[str]):
        """全アクティブレースを並列監視する。"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            tasks = [
                self._watch_single_race(browser, rid)
                for rid in race_ids
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            await browser.close()

    async def _watch_single_race(self, browser, race_id: str):
        """単一レースのオッズを5分ごとに取得する。"""
        page = await browser.new_page()
        jcd  = race_id[:2]
        hd   = race_id[2:10]
        rno  = race_id[10:]
        url  = (
            f"https://www.boatrace.jp/owpc/pc/race/oddstf"
            f"?jcd={jcd}&hd={hd}&rno={rno}"
        )

        while race_id in self.active_races:
            try:
                await page.goto(url, timeout=20000)
                await page.wait_for_selector("table.is-w650", timeout=10000)

                odds = await self._parse_odds(page, race_id)
                if odds:
                    self._save_to_db(odds)

                    for entry in odds:
                        ev = self._calculate_ev(race_id, entry)
                        if ev and ev >= 1.20:
                            print(
                                f"[ALERT] EV={ev:.3f} | "
                                f"{race_id} 艇{entry['boat_number']} | "
                                f"オッズ{entry['win_odds']}倍"
                            )

            except Exception as e:
                print(f"[WARN] {race_id}: {e}")

            await asyncio.sleep(300)  # 5分待機

        await page.close()

    def _save_to_db(self, odds: list[dict]):
        """オッズをSQLiteに保存する。"""
        conn = sqlite3.connect(self.db_path)
        for entry in odds:
            conn.execute("""
                INSERT OR IGNORE INTO live_odds
                (race_id, boat_number, win_odds, fetched_at)
                VALUES (?, ?, ?, ?)
            """, (
                entry["race_id"],
                entry["boat_number"],
                entry["win_odds"],
                datetime.now().isoformat(),
            ))
        conn.commit()
        conn.close()

    def _calculate_ev(self, race_id: str, odds_entry: dict) -> float:
        """予測確率×オッズでEVを計算する。"""
        conn = sqlite3.connect(self.db_path)
        row  = conn.execute("""
            SELECT predicted_prob FROM forecasts
            WHERE race_id = ? AND boat_number = ?
            ORDER BY created_at DESC LIMIT 1
        """, (race_id, odds_entry["boat_number"])).fetchone()
        conn.close()

        if row is None:
            return None

        return row[0] * odds_entry["win_odds"]
```

このシステムが完成すれば、EVが実際に計算できるようになる。理論上、ROIが10〜15pt改善する可能性がある。

「可能性」と書いたのは、まだ実装途中だからだ。正直に書く。

---

## アイデア2: 会場ブラックリストの精緻化

**期待効果: +3〜5pt**
**難易度: 低**
**優先度: 高**

宮島競艇場はすでにブラックリスト入りさせた。ペーパートレードでROI 0%だったからだ。でも「ROI 0%だから除外」という基準が統計的に正しいかどうかを確認する必要がある。

```python
import pandas as pd
import scipy.stats as stats
import json

def analyze_stadium_performance(
    log_path:              str   = "paper_trading.jsonl",
    min_bets:              int   = 10,    # 最低10件ないと統計的判断不可
    blacklist_roi_threshold: float = 0.60,  # ROI 60%以下でBL候補
) -> pd.DataFrame:
    """
    会場別のROIを分析してブラックリスト候補を特定する。

    二項検定で「このROIが偶然かどうか」も確認する。
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("pnl") is not None:
                records.append(r)

    df = pd.DataFrame(records)
    df["stadium_code"] = df["race_id"].str[:2]

    stadium_names = {
        "01": "桐生",  "02": "戸田",   "03": "江戸川",
        "04": "平和島", "05": "多摩川", "06": "浜名湖",
        "07": "蒲郡",  "08": "常滑",   "09": "津",
        "10": "三国",  "11": "びわこ",  "12": "住之江",
        "13": "尼崎",  "14": "鳴門",   "15": "丸亀",
        "16": "児島",  "17": "宮島",   "18": "徳山",
        "19": "下関",  "20": "若松",   "21": "芦屋",
        "22": "福岡",  "23": "唐津",   "24": "大村",
    }
    df["stadium_name"] = df["stadium_code"].map(stadium_names)

    results = []
    for code, grp in df.groupby("stadium_code"):
        n = len(grp)
        if n < min_bets:
            results.append({
                "stadium": stadium_names.get(code, code),
                "n_bets": n,
                "status": "データ不足（判断保留）",
            })
            continue

        total_invest = grp["bet_amount"].sum()
        total_return = grp["payout"].sum()
        roi          = total_return / total_invest if total_invest > 0 else 0
        hit_rate     = (grp["result"] == "hit").mean()

        # 二項検定: 的中率が期待値（16.7%）より有意に高いか
        n_hits  = int((grp["result"] == "hit").sum())
        binom_p = stats.binomtest(n_hits, n, p=0.167, alternative="greater").pvalue

        status = "BL候補" if roi < blacklist_roi_threshold else "維持"

        results.append({
            "stadium":  stadium_names.get(code, code),
            "n_bets":   n,
            "hit_rate": f"{hit_rate:.1%}",
            "roi":      f"{roi:.1%}",
            "status":   status,
            "p_value":  f"{binom_p:.3f}",
        })

    return pd.DataFrame(results).sort_values("roi")
```

現時点ではデータが少なすぎて統計的に有意な判断ができない。でも方向性は見えている。

---

## アイデア3: 信頼度TOP1%への絞り込み

**期待効果: +5〜8pt**
**難易度: 低**
**優先度: 高**

現在は信頼度上位5%（conf_pctile >= 95）の予測にベットしている。これをTOP1%に絞る。

```python
# 現在の設定
CONF_THRESHOLD = 95  # 上位5%

# 変更後の設定
CONF_THRESHOLD = 99  # 上位1%

# バックテスト比較（推定）
# conf >= 95: 1日平均12.3件、ROI 99.2%
# conf >= 99: 1日平均 2.4件、ROI 104.7%
# conf >= 99.5: 1日平均0.8件（少なすぎる）
```

ただし、1日2.4件になるとサンプルが溜まるスピードが5分の1になる。200件集めるのに83日から415日になる。

「より正確に、でも遅く」か「少し荒削りでも早く検証」か。今は後者を選んでいる。

---

## アイデア4: 3号艇フィルタの再評価

**期待効果: 不明（要検証）**
**難易度: 中**
**優先度: 中**

v3では「3号艇へのベットを禁止」しているが、これが正しいかどうかわからない。

3号艇は「アウトコース」でありながら「インコースほどの有利性はない」という中途半端なポジションだ。一般的に勝率が低い。でもそれが全ての場で成立するわけではない。

```python
def evaluate_boat_number_filter(
    results_df:   pd.DataFrame,
    min_samples:  int = 100,
) -> pd.DataFrame:
    """
    艇番ごとのROIを評価して、フィルタの妥当性を確認する。

    100件以上溜まった段階で再評価することを推奨。
    現時点（16件）ではこの分析は無意味に近い。
    """
    analysis = results_df.groupby("boat_number").agg(
        n_bets       = ("bet_amount", "count"),
        total_invest = ("bet_amount", "sum"),
        total_return = ("payout",     "sum"),
    ).assign(
        roi = lambda x: x["total_return"] / x["total_invest"]
    )

    for boat_num, row in analysis.iterrows():
        if row["n_bets"] < min_samples:
            print(
                f"艇{boat_num}: {row['n_bets']}件 "
                f"→ 統計的判断には{min_samples}件必要"
            )

    return analysis
```

100件溜まったら再評価する。現時点では保留。

---

## アイデア5: Kelly基準でベット額を動的に決める

**期待効果: リスク調整済みリターンの改善**
**難易度: 低**
**優先度: 中**

現在は全ベット一律1,000円。でもEVが1.5の予測と1.2の予測では、ベットすべき金額が違う。

```python
def dynamic_bet_sizing(
    forecasts:      list[dict],
    bank_roll:      float,
    kelly_fraction: float = 0.25,  # Quarter Kelly（保守的）
    min_bet:        int   = 500,
    max_bet:        int   = 3_000,
) -> list[dict]:
    """
    EVに基づいてベット額を動的に決定する。

    フルKelly（fraction=1.0）は理論上最適だが、
    モデルの予測誤差があるためQuarter Kellyを推奨。
    上限3,000円で大負けを防ぐ。
    """
    result = []

    for forecast in forecasts:
        ev   = forecast.get("ev")
        prob = forecast["predicted_prob"]
        odds = forecast.get("current_odds")

        if ev is None or ev < 1.20 or odds is None:
            continue

        b = odds - 1.0   # 純利益率
        p = prob
        q = 1.0 - p

        kelly = (p * b - q) / b  # フルKelly
        if kelly <= 0:
            continue

        bet_fraction = kelly * kelly_fraction
        raw_bet      = int(bank_roll * bet_fraction)
        final_bet    = max(min_bet, min(max_bet, raw_bet))

        forecast["bet_amount"] = final_bet
        forecast["kelly_raw"]  = kelly
        result.append(forecast)

    return result
```

---

## アイデア6: 時間帯フィルタ（4R〜10R）

**期待効果: +2〜4pt**
**難易度: 低**
**優先度: 低**

競艇の1Rと12Rは特殊だという仮説がある。1Rは「開幕戦」でデータが少なく、12Rは「最終レース」で荒れやすい。

```python
def is_normal_race(race_id: str) -> bool:
    """
    「普通」のレース番号かどうかを判定する。

    4R〜10R が最も標準的なレース。
    1-3R: 朝一、選手の状態が不安定な場合がある
    11-12R: 最終レース付近、荒れやすい傾向

    ※ 仮説検証中。現時点では根拠が薄い。
    """
    race_num = int(race_id[-2:])
    return 4 <= race_num <= 10
```

---

## アイデア7: 2連単専用モデルの追加

**期待効果: 高配当狙いで収益構造改善**
**難易度: 高**
**優先度: 中**

現在のモデルは「1着を予測する」設計だ。2連単（1着と2着の組み合わせ）は別の問題だ。ペーパートレードでは2連単が全収益の66%を占めているが、それは「たまたまヒットした」だけで、モデルが意図したものではない。

```python
from itertools import permutations

def prepare_exacta_training_data(
    race_data: pd.DataFrame,
    results:   pd.DataFrame,
) -> pd.DataFrame:
    """
    2連単（Exacta）モデルの訓練データを作成する。

    各レースの全1着・2着組み合わせ（6P2=30通り）を生成し、
    実際の組み合わせをラベル1とする多クラス分類問題。
    """
    all_pairs = []

    for race_id, race_grp in race_data.groupby("race_id"):
        boats = race_grp["boat_number"].tolist()

        result_row = results[results["race_id"] == race_id]
        if result_row.empty:
            continue

        true_1st = result_row["winner_boat"].iloc[0]
        true_2nd = result_row["second_boat"].iloc[0]

        for first, second in permutations(boats, 2):
            first_feat  = race_grp[race_grp["boat_number"] == first].iloc[0]
            second_feat = race_grp[race_grp["boat_number"] == second].iloc[0]

            all_pairs.append({
                "race_id":              race_id,
                "first_boat":           first,
                "second_boat":          second,
                "first_nwr":            first_feat["national_wr"],
                "first_motor_2r":       first_feat["motor_2rate"],
                "first_ex_from_best":   first_feat.get("ex_time_from_best", np.nan),
                "second_nwr":           second_feat["national_wr"],
                "second_motor_2r":      second_feat["motor_2rate"],
                "second_ex_from_best":  second_feat.get("ex_time_from_best", np.nan),
                "label": int(first == true_1st and second == true_2nd),
            })

    return pd.DataFrame(all_pairs)
```

---

## 優先順位まとめ

1. **Value Bet完全実装**　期待効果 +10〜15pt　難易度 高　— 実装中
2. **会場BL精緻化**　期待効果 +3〜5pt　難易度 低　— データ待ち
3. **TOP1%絞り込み**　期待効果 +5〜8pt　難易度 低　— すぐ試せる
4. **3号艇フィルタ見直し**　期待効果 不明　難易度 中　— 100件後
5. **Kelly基準ベット額**　期待効果 リスク改善　難易度 低　— すぐ試せる
6. **時間帯フィルタ**　期待効果 +2〜4pt　難易度 低　— 仮説検証中
7. **2連単専用モデル**　期待効果 高配当狙い　難易度 高　— 設計中

---

居酒屋で書いたリストが、ちゃんとロードマップになった。

次にやることは決まっている。Value Betの実装だ。それだけで10〜15pt改善するなら、他のことを後回しにしても価値がある。

ハイボールをもう一杯頼みながら、そう思った。

---

*次回：会場別モデルで精度向上。大村70.8%と平和島46.2%の違いが、なぜそこまで生まれるのか。会場特化戦略の設計。*
