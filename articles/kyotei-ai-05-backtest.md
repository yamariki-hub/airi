---
title: "バックテスト設計：「過去で勝てた」を「未来で勝てる」に変えるには"
emoji: "📊"
type: "tech"
topics: ["機械学習", "競艇", "Python", "バックテスト", "データ分析"]
published: false
price: 300
---

バックテストは「未来の自分を騙すための作業」になりがちだ。

ROIが高いバックテスト結果を出すのは難しくない。過去のデータを使ってモデルを作り、同じ過去のデータで評価する。パラメータをいじり続ける。「ROI 120%」という数字が出る。

難しいのは、**実運用でも同じROIが出るバックテスト**を設計することだ。

私は何度かこの罠にはまった。バックテストで105%が出た戦略を実運用に移したら80%だった。バックテストで99%が出た戦略のペーパートレードが、現在41.5%だ。

この記事では「なぜバックテストと実運用が乖離するか」と「それを少しでも減らす設計方法」を書く。

---

---

## なぜ普通の検証は使えないか

一番シンプルな検証方法はこうだ：

```python
from sklearn.model_selection import train_test_split
import numpy as np

# NG: 時系列を無視したランダム分割
X = df[feature_cols].values
y = (df["place"] == 1).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,  # ← これが問題
)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

これは競艇AIでは**絶対にやってはいけない**。

なぜか。`shuffle=True` でランダムに分割すると、2026年1月のデータが訓練セットに入り、2025年6月のデータが検証セットに入ることが起きる。つまり**未来のデータで過去を予測している**状態になる。

現実の競艇予測では、「今日のレース」を「今日より前のデータ」でしか予測できない。未来は存在しない。

---

## Walk-Forward法：時系列データの正しい評価

時系列データに対する正しい評価方法はこれだ：

```
期間1: [訓練 1〜3月] → [検証 4月]
期間2: [訓練 2〜4月] → [検証 5月]
期間3: [訓練 3〜5月] → [検証 6月]
...（スライドしながら繰り返す）
```

訓練データは常に検証データより過去。未来の情報は絶対に混入しない。

```python
from datetime import date, timedelta
import pandas as pd

def walk_forward_backtest(
    df: pd.DataFrame,
    train_days: int = 90,     # 訓練期間（日数）
    test_days: int = 30,      # 検証期間（日数）
    strategy: str = "CONF95", # 使用する戦略
) -> list[dict]:
    """
    Walk-Forward バックテストを実行する。

    train_days日分で訓練 → test_days日分で評価 → 1期間スライド
    を繰り返す。各期間の結果をリストで返す。
    """
    results = []

    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    # 訓練期間分を確保できる最初の日から開始
    test_start = min_date + timedelta(days=train_days)

    fold_num = 0
    while test_start + timedelta(days=test_days) <= max_date:
        fold_num += 1
        test_end = test_start + timedelta(days=test_days)
        train_start = test_start - timedelta(days=train_days)

        # 訓練データ（test_startより前）
        train_df = df[
            (df["race_date"] >= str(train_start)) &
            (df["race_date"] < str(test_start))
        ].copy()

        # 検証データ（test_start〜test_end）
        test_df = df[
            (df["race_date"] >= str(test_start)) &
            (df["race_date"] < str(test_end))
        ].copy()

        # データが少なすぎる場合はスキップ
        if len(train_df) < 500 or len(test_df) < 100:
            test_start += timedelta(days=test_days)
            continue

        print(f"\n=== Fold {fold_num}: 訓練 {train_start}〜{test_start}, 検証 {test_start}〜{test_end} ===")

        # モデル訓練
        model_bin, model_rank, fcols = train_models(train_df)

        # 検証期間で評価
        fold_result = evaluate_fold(
            test_df, model_bin, model_rank, fcols,
            strategy=strategy,
        )
        fold_result.update({
            "fold": fold_num,
            "train_start": str(train_start),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        })
        results.append(fold_result)

        # 1期間スライド
        test_start += timedelta(days=test_days)

    return results
```

---

## 評価指標：ROIだけ見ていると騙される

バックテストで確認すべき指標は複数ある。ROIだけを最適化すると過学習する。

### 1. ROI（回収率）：最基本の指標

```python
def calc_roi(bets: list[dict]) -> float:
    """
    ROI = 総払い戻し / 総ベット額 × 100
    100% = 損益ゼロ
    目標: 安定して105%以上
    """
    if not bets:
        return 0.0
    total_bet = sum(b["bet_amount"] for b in bets)
    total_payout = sum(b["payout"] for b in bets)
    if total_bet == 0:
        return 0.0
    return total_payout / total_bet * 100
```

### 2. 黒字fold数：安定性の指標

12期間中何期間が黒字か。ROI 100%超えのfold数で安定性を評価する。

「平均ROI 105%だが、1期間が200%で他は全部90%」という状態と「12期間すべてが103〜107%」という状態は全く違う。後者の方が実運用に近い。

```python
def count_profitable_folds(results: list[dict]) -> tuple[int, int]:
    """黒字期間数を数える"""
    profitable = sum(1 for r in results if r.get("roi", 0) >= 100)
    total = len(results)
    return profitable, total
```

### 3. 最大ドローダウン：精神的な耐久力の指標

```python
def calc_max_drawdown(daily_profits: list[float]) -> float:
    """
    最大ドローダウン（連続損失の最大額）を計算する。
    これが大きいと精神的に続けられなくなる。
    """
    cumulative = pd.Series(daily_profits).cumsum()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    return float(drawdown.min())  # 負の値（損失額）
```

連続で負け続けた場合の最大損失。これが大きいと、たとえ長期的にROI 100%超えの戦略でも心が折れてやめてしまう。特に実際のお金を賭けているなら、この指標は重要だ。

### 4. シャープレシオ：リターンのリスク調整済み評価

```python
def calc_sharpe(daily_rois: list[float]) -> float:
    """
    シャープレシオ = 平均リターン / 標準偏差 × sqrt(取引日数)
    高いほど、ボラティリティが低い安定した戦略
    """
    returns = pd.Series(daily_rois)
    if returns.std() == 0:
        return 0.0
    # 年換算（競艇は年300日程度）
    return returns.mean() / returns.std() * (300 ** 0.5)
```

---

## 私のバックテスト結果（12期間Walk-Forward）

実際に回した結果：

- **全レース（RANK_any）**: ROI 89.8%　黒字fold 4/12　最大連敗 7　— 全レースに均等ベット
- **CONF75**: ROI 92.7%　黒字fold 5/12　最大連敗 5　— 確信度上位25%に絞る
- **CONF95+EX**: ROI **99.2%**　黒字fold **8/12**　最大連敗 4　— 確信度TOP5%+展示タイム条件

CONF95+EXは「確信度TOP5% かつ 展示タイム条件をクリア（ex_time_from_best ≤ 0.10秒 かつ ex_time_rank ≤ 2）」を満たすレースのみに賭ける戦略だ。

ROI 99.2%、黒字fold 8/12。

これが今の私の最良の戦略だ。でも実際のペーパートレードは41.5%。この乖離については後述する。

---

## バックテストのよくある落とし穴

### 落とし穴1: データリーク（最も多い失敗）

```python
# NG: その日の結果が特徴量に混入している
df["win_rate"] = df.groupby("racer")["place"].transform("mean")
# transform("mean") は過去・現在・未来の全データの平均を使う

# OK: shift(1)で当日を除外
df["win_rate"] = (
    df.groupby("racer")["place"]
    .transform(lambda s: (s.shift(1) == 1).rolling(30, min_periods=5).mean())
)
```

私の最初のモデルはこれで ROI 148% という数字を出した。実運用で90%以下に急落した。次の記事で詳しく書く。

### 落とし穴2: 取引コストの無視

競艇の控除率25%は「市場の構造的なコスト」としてモデルの外に存在する。

バックテストでは「100円賭けて110円返ってきた = ROI 110%」と計算できるが、これは個々のレースの実績値。ランダムに賭け続けると長期ROIは75%に収束する（100 × 0.75 = 75円）。

**モデルがしなければならないのは、このランダム75%のベースラインを超えること**だ。ROI 80%のモデルは「ランダムより5ポイント悪い」という意味で、ただの損失要因だ。

### 落とし穴3: DQ（失格）の不正確な処理

競艇では転覆・妨害・フライングで失格になる艇がある。データ上では `place = 7` や `place = 0` で入ってくることがある。

```python
def fix_dq(df: pd.DataFrame) -> pd.DataFrame:
    """
    失格（DQ）の正しい処理。
    place > 6 または place = 0 はNaNに変換する。
    全艇がDQのレースは中止レースとして除外。
    """
    # DQをNaNに変換（ローリング統計への悪影響を防ぐ）
    mask_dq = (
        (df["place"] > 6) |
        (df["place"] == 0) |
        df["place"].isna()
    )
    df.loc[mask_dq, "place"] = np.nan
    df.loc[mask_dq, "start_timing"] = np.nan  # STタイムも無効化

    # 全艇DQのレース（実質的な中止）を除外
    valid = df.groupby("race_id")["place"].transform(lambda x: x.notna().any())
    df = df[valid].reset_index(drop=True)

    print(f"DQ処理後: {len(df)} rows（除外: {mask_dq.sum()}行）")
    return df
```

DQを `place = 7` のまま使うと、ローリング勝率の計算が狂う。7着として「負け」にカウントするのと「無効」にするのでは、選手の成績評価が大きく変わる。

### 落とし穴4: 過剰なパラメータ最適化（カーブフィッティング）

バックテストで「このパラメータだとROI 108%」を発見して喜ぶのは危険だ。

なぜなら、そのパラメータは**そのバックテスト期間のデータに特有のパターン**に過学習している可能性がある。

対策：
- パラメータ探索はバックテスト期間の**前半**で行う
- 発見したパラメータを**後半の独立したテスト期間**で最終確認する
- 「なぜこのパラメータが良いのか」を論理的に説明できなければ信用しない

私はこれを「2重のWalk-Forward」と呼んでいる。外側のWalk-Forwardが最終評価、内側のWalk-Forwardがパラメータ探索だ。実装は複雑になるが、再現性が格段に上がる。

### 落とし穴5: 非開催日の扱い

競艇は基本毎日開催だが、悪天候・整備日で中止になることがある。この日のデータは「ベット数ゼロ」として処理する必要がある。

```python
def fill_non_race_days(
    daily_results: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    非開催日を0ベット・0収益として埋める。
    日次の時系列を正確に作るために必要。
    """
    # 全日付のインデックスを作成
    all_dates = pd.date_range(start_date, end_date, freq="D")
    full_df = pd.DataFrame({"race_date": all_dates})

    # 実際のデータとマージ（開催日のみデータあり）
    result = full_df.merge(daily_results, on="race_date", how="left")
    result = result.fillna({
        "bet_count": 0,
        "total_bet": 0,
        "total_payout": 0,
        "daily_profit": 0,
    })
    return result
```

---

## バックテストと実運用の乖離：正直に話す

バックテストROI 99.2%。ペーパートレードROI 41.5%。

この差は何か。

**1. 分布の外れ（アウトオブサンプル問題）**
バックテスト期間と実運用期間では、競艇の「環境」が変わっている可能性がある。特定の季節・特定の選手コンディションに偏ったデータでモデルを作ると、それ以外の状況では精度が落ちる。

**2. サンプル数の問題**
CONF95戦略では1日あたり0〜3件しか賭けない。月30件程度のサンプルでROIを評価するのは、統計的に相当不安定だ。「本当に41.5%なのか」「運が悪かっただけなのか」が判断できない水準だ。

**3. データリークの残滓**
完全にリークを取り除けていない可能性がある。バックテストで使ったデータの一部が訓練に「染み出している」パターンはまだ完全には排除できていない。

**4. 展示タイム取得のタイミングずれ**
バックテストでは「その日の展示タイム」を使っているが、実際には展示タイムはレース数時間前にしか取得できない。バックテスト時に使った展示タイムと、実運用時に取れた展示タイムに差がある可能性がある。

バックテストは「どの方向に改善するかを判断するための道具」だ。「これで実際に儲かる」という証明にはならない。この現実を受け入れることが、競艇AI開発の第一歩だと思っている。

---

## バックテストの自動化：毎日回す仕組み

```python
import subprocess
from datetime import date

def run_daily_backtest(today: date):
    """
    毎朝、前日までのデータで最新のバックテストを実行する。
    結果をJSONに保存して、トレンドを監視する。
    """
    print(f"=== Daily Backtest: {today} ===")

    # 最新12期間Walk-Forwardを実行
    df = load_all_features()  # 全期間の特徴量を読み込み
    results = walk_forward_backtest(df, train_days=21, test_days=7)

    # 結果をJSONに保存
    import json
    output_path = f"backtest_results/{today}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # サマリーを出力
    avg_roi = sum(r["roi"] for r in results) / len(results)
    profitable = sum(1 for r in results if r["roi"] >= 100)
    print(f"平均ROI: {avg_roi:.1f}%")
    print(f"黒字期間: {profitable}/{len(results)}")

    return results
```

毎日バックテストを回すことで「モデルの劣化」を早期に検知できる。1週間前は99%だったのに今週は88%になっている、という変化に気づけば、何かが変わったことがわかる。

---

今日のペーパートレード：2件賭けて1件的中。ROI 60%。

負けた日は居酒屋で一人でビールを飲む。ノートにバックテストの結果を書き留めながら「なんで外れたんだ」と考える。隣のサラリーマンが楽しそうに話しているのを聞きながら、競艇のオッズを眺める。

それでもまた明日、6時に起きてスクリプトを走らせる。

データが増えた。それだけでいい。

---

*次回: データリークという罠 — v1モデルでROI 148%が幻だった理由*
