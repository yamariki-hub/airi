---
title: "LightGBM Binary + LambdaRank：なぜ2つのモデルを組み合わせるのか"
emoji: "🤖"
type: "tech"
topics: ["LightGBM", "機械学習", "競艇", "Python", "ランキング学習"]
published: false
price: 500
---

「なんで1つのモデルじゃダメなの？」

よく聞かれる。正直に言うと、最初は1つで作った。Binaryモデルだけで「1着確率を出して、一番高い艇に賭ける」という単純な実装をした。

それなりに動いた。AUCも悪くなかった。

でも「このモデルが本当に着順を理解しているか？」という疑問が頭から離れなかった。

深夜、居酒屋から帰ってきてラップトップを開き、「LambdaRank 競艇」で検索した。検索エンジンのランキング最適化に使われているアルゴリズムが、着順予測に転用できるという記事を見つけた。缶チューハイを飲みながら論文を読んだ。

**これだ、と思った。**

---

---

## 2つの予測タスク：Binaryは「確率」、LambdaRankは「順位」

競艇で「1着を当てる」には、実は2種類の判断が必要だ。

- **勝利確率の絶対値**: この艇が1着になる確率は何%か？　→ Binary分類
- **艇間の順位関係**: このレースで最も強い艇はどれか？　→ LambdaRank

Binaryモデル単体の弱点：各艇を「1着か否か」の独立した問題として扱う。レース内の6艇の関係性を直接学習しない。

LambdaRankの弱点：スコアとして出力されるが、確率として解釈しにくい。「このスコア0.8は、1着になる確率80%」ではない。

**2つを組み合わせることで、互いの弱点を補う。** Binaryで「絶対的な信頼度」を、LambdaRankで「レース内の相対強さ」を見る。

---

## Binary分類モデル：「1着か否か」を予測する

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

def train_binary_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> lgb.Booster:
    """
    Binary分類モデルを訓練する。
    ターゲット: 1着なら1、それ以外は0
    """
    # ラベル作成: 1着=1、2〜6着=0、DQ(NaN)=除外
    valid_df = train_df.dropna(subset=["place"])
    y = (valid_df["place"] == 1).astype(float)
    X = valid_df[feature_cols].values

    params_bin = {
        "objective": "binary",          # 2値分類
        "metric": "binary_logloss",     # 評価指標: 対数損失
        "num_leaves": 31,               # 木の葉の数（複雑さの制御）
        "learning_rate": 0.05,          # 学習率（小さいほど慎重に学習）
        "feature_fraction": 0.8,        # 各ツリーで使う特徴量の割合（過学習防止）
        "bagging_fraction": 0.8,        # 各ツリーで使うデータの割合
        "bagging_freq": 5,              # baggingの頻度（5ツリーごと）
        "min_child_samples": 20,        # 葉の最小サンプル数（小さい葉を防ぐ）
        "verbose": -1,                  # ログを出力しない
        "seed": 42,                     # 再現性のための乱数シード
    }

    # カテゴリ変数として扱う特徴量
    cat_names = [c for c in ["stadium_num"] if c in feature_cols]

    ds_bin = lgb.Dataset(
        X,
        label=y,
        feature_name=feature_cols,
        categorical_feature=cat_names,
    )

    # early_stoppingなし（データが少ない場合を考慮して固定round数）
    model_bin = lgb.train(
        params_bin,
        ds_bin,
        num_boost_round=300,  # 300本のツリーを学習
    )

    return model_bin
```

**出力:** `model_bin.predict(X)` → 各艇の「1着確率」（0〜1の実数）

6艇のうち最も確率が高い艇が予測1位。ただし1着確率の「合計が1にならない」点に注意（各艇を独立して予測するため）。そのため後で正規化が必要になる。

---

## LambdaRankモデル：「着順を正しく並べる」ことに特化

LambdaRankは、もともとGoogleが検索エンジンのランキング最適化に開発したアルゴリズムだ。「このクエリに対して、どの文書が上位に来るべきか」を学習する。

競艇に適用すると「このレースで、どの艇が上位（1着・2着）に来るべきか」になる。レース内の6艇を一つのグループとして、相対的な順位を最適化する点が、Binaryとの大きな違いだ。

```python
def train_lambdarank_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> lgb.Booster:
    """
    LambdaRankモデルを訓練する。
    ターゲット: 1着=6点, 2着=5点, ..., 6着=1点, DQ=0点（relevance）
    """
    # LambdaRankはレース(グループ)順に並べておく必要がある
    ts = train_df.sort_values(["race_id", "boat_number"]).dropna(subset=["place"])

    # relevance スコア: 上位着順ほど高い値
    # 1着→6点, 2着→5点, ..., 6着→1点, DQ→0点
    def place_to_relevance(p) -> int:
        if pd.isna(p) or p < 1 or p > 6:
            return 0
        return int(7 - p)  # 7 - 1 = 6, 7 - 6 = 1

    ts["relevance"] = ts["place"].apply(place_to_relevance)

    # グループ情報: 各レースに何艇いるか（通常は6艇）
    # LambdaRankはこの単位でランキングを最適化する
    groups = ts.groupby("race_id").size().values

    params_rank = {
        "objective": "lambdarank",
        "metric": "ndcg",            # NDCG（正規化割引累積利得）で評価
        "ndcg_eval_at": [1, 3],      # 1位・3位以内の正確さを重視
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }

    cat_names = [c for c in ["stadium_num"] if c in feature_cols]

    # group引数が必須（これがBinaryとの最大の違い）
    ds_rank = lgb.Dataset(
        ts[feature_cols].values,
        label=ts["relevance"].values,
        group=groups,              # ← レースごとのグループ情報
        feature_name=feature_cols,
        categorical_feature=cat_names,
    )

    model_rank = lgb.train(
        params_rank,
        ds_rank,
        num_boost_round=300,
    )

    return model_rank
```

**出力:** `model_rank.predict(X)` → 各艇の「スコア」（数値が高いほど上位予測）

このスコアは確率ではない。「このレースで最も高いスコアの艇が1位」という相対的な意味しか持たない。

---

## 2モデルの組み合わせ：予測ロジックの全貌

実際の予測では、2つのモデルの出力を組み合わせる。

```python
def select_best_bet(
    race_df: pd.DataFrame,
    feature_cols: list[str],
    model_bin: lgb.Booster,
    model_rank: lgb.Booster,
) -> dict:
    """
    1レース分の予測を行い、賭け対象を選ぶ。
    戻り値: 予測結果の辞書（predicted_1st, model_prob, bin_agree など）
    """
    X = race_df[feature_cols].values

    # === Step 1: 各モデルのスコアを取得 ===
    scores_bin = model_bin.predict(X)   # 各艇の1着確率（0〜1）
    scores_rank = model_rank.predict(X) # 各艇のランクスコア

    race_df = race_df.copy()
    race_df["score_bin"] = scores_bin
    race_df["score_rank"] = scores_rank

    # === Step 2: Binary確率を正規化（レース内合計=1にする）===
    # 元の確率は各艇が独立なので合計が1とは限らない
    # 正規化することで「このレースでの相対的な勝利確率」に変換
    total_prob = race_df["score_bin"].sum()
    race_df["score_bin_norm"] = race_df["score_bin"] / total_prob

    # === Step 3: LambdaRankスコアで1位・2位を決定 ===
    # LambdaRankは着順の相対関係を最適化しているため、
    # 「どの艇が1位か」の判断はRankモデルに任せる
    race_df = race_df.sort_values("score_rank", ascending=False)
    pred_1st = int(race_df.iloc[0]["boat_number"])
    pred_2nd = int(race_df.iloc[1]["boat_number"])

    # === Step 4: 2モデルの合意確認 ===
    # BinaryモデルとRankモデルが同じ艇を1位に選んでいるか？
    # 両者が一致 = 信頼度が高い（bin_agree = True）
    bin_ranked = race_df.sort_values("score_bin", ascending=False)
    bin_top = int(bin_ranked.iloc[0]["boat_number"])
    bin_agree = (bin_top == pred_1st)

    # 予測1位のBinary確率（正規化後）
    pred_prob = float(
        race_df[race_df["boat_number"] == pred_1st]["score_bin_norm"].iloc[0]
    )

    return {
        "predicted_1st": pred_1st,
        "predicted_2nd": pred_2nd,
        "model_prob": pred_prob,  # 正規化後の1着確率
        "bin_agree": bin_agree,   # 2モデル合意フラグ
        # スコア差（次のフィルタリングに使う）
        "rank_score_diff": float(race_df.iloc[0]["score_rank"] - race_df.iloc[1]["score_rank"]),
    }
```

`bin_agree = True`（2モデルが同じ艇を1位に選んだとき）は予測の信頼度が高い。このフラグと次の「確信度フィルタ」を組み合わせて、賭けるレースを絞り込む。

---

## 確信度フィルタ（CONF95）：上位5%のレースだけに賭ける

全レースに賭けるより、**確信度が高いレースだけに絞る**方がROIが改善する。

感覚的には当たり前だ。「よくわからないレース」に無理やり賭けるより、「絶対これだと思うレース」だけに絞った方がいい。

```python
import numpy as np

def apply_conf95_filter(
    all_predictions: list[dict],
    threshold_percentile: float = 95.0,
) -> tuple[list[dict], float]:
    """
    確信度フィルタを適用する（上位percentile%のレースのみ選択）。
    確信度 = LambdaRankの1位と2位のスコア差（差が大きい=予測に自信あり）
    """
    if not all_predictions:
        return [], 0.0

    # 全レースのスコア差を収集
    diffs = [p["rank_score_diff"] for p in all_predictions]

    # 上位percentile%の閾値を計算
    threshold = np.percentile(diffs, threshold_percentile)

    # 閾値を超えるレースだけを選択
    selected = [p for p in all_predictions if p["rank_score_diff"] >= threshold]

    print(f"全{len(all_predictions)}レース → 確信度TOP{100-threshold_percentile:.0f}%: {len(selected)}レース")
    return selected, threshold
```

**バックテスト（12期間Walk-Forward）での比較：**

- **全レースに賭ける（RANK_any）**: ROI 89.8%　黒字期間 4/12　平均ベット数/日 〜50件
- **上位25%（CONF75）**: ROI 92.7%　黒字期間 5/12　平均ベット数/日 〜13件
- **上位5%（CONF95+EX）**: ROI **99.2%**　黒字期間 **8/12**　平均ベット数/日 〜2件

絞るほどROIが改善する。これは「確信度の低いレース = ランダムより少し悪い」ことを示している。AIが「わからない」と言えるレースには賭けない方がいい。

ただし絞りすぎると1日のベット数が0〜2件になり、月単位での統計的信頼性が落ちる。5%（上位5%）が今の私の判断では最良のバランスだ。

---

## 毎日21日間のデータで再訓練する理由

モデルは直近21日のデータで毎朝再訓練している。

```python
from datetime import date, timedelta
import pandas as pd

def load_training_data(
    before_date: date,
    train_days: int = 21,
) -> pd.DataFrame:
    """
    指定日の直前train_days日間のデータを訓練用に読み込む。
    当日のデータは含まない（未来情報の混入を防ぐ）。
    """
    end_date = before_date - timedelta(days=1)   # 昨日まで
    start_date = end_date - timedelta(days=train_days - 1)

    query = f"""
        SELECT r.race_date, r.stadium_id, r.race_number,
               rr.boat_number, rr.place, rr.start_timing, rr.racer_number,
               rc.national_win_rate, rc.motor_2rate, rc.exhibition_time
        FROM races r
        JOIN race_results rr ON r.id = rr.race_id
        LEFT JOIN race_cards rc ON r.id = rc.race_id AND rr.boat_number = rc.boat_number
        WHERE r.race_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY r.race_date, r.id, rr.boat_number
    """
    df = pd.read_sql(query, engine)
    print(f"訓練データ: {start_date} 〜 {end_date} ({len(df)} rows)")
    return df
```

**なぜ21日なのか。**

競艇は節（開催期間）が単位で、同じメンバー・同じモーターで数日続けて開催される。節が変わると選手の組み合わせもモーターも変わる。古いデータは「別の状況」のデータになっていく。

21日はバックテストで試した中で最良だった期間だ。30日だと古いパターンの影響が出やすく、14日だとデータが少なすぎてモデルが不安定だった。

---

## 期待値フィルタ（EV_THRESHOLD）：オッズを組み合わせる

CONF95フィルタに加えて、**期待値フィルタ**も実装した。

```python
EV_THRESHOLD = 1.20  # 期待値がこの値以上のレースのみ賭ける

def apply_ev_filter(prediction: dict, win_odds: float) -> bool:
    """
    期待値（EV）フィルタを適用する。
    EV = モデルが計算した1着確率 × 実際の単勝オッズ
    EV > 1.2 = 市場が20%以上過小評価している = 賭けるべき
    """
    model_prob = prediction["model_prob"]  # モデルが出した1着確率（0〜1）
    ev = model_prob * win_odds              # 期待値（1以上で黒字期待）

    return ev >= EV_THRESHOLD
```

例えば、モデルが「この艇の1着確率は40%」と判断し、実際のオッズが3.5倍（配当350円）の場合：
- EV = 0.40 × 3.5 = **1.40**（閾値1.20を超えるので賭ける）

モデルが「40%」と判断しているのに、市場（オッズ）が「29%相当」の評価しかしていない場合に賭ける仕組みだ。

これを実装するにはリアルタイムオッズの取得が必要で、現在の私の実装では完全には機能していない。オッズの自動取得は今後の課題だ。

---

## バックテストAUCの実測値

参考として、現在のモデルのAUC（ROC曲線下面積）は0.847だ。

AUCは0.5がランダム（コインフリップと同じ）、1.0が完全予測。0.847は「かなり良い分離ができている」という水準だ。

ただし、**AUCが高くてもROIが低い**というパラドックスがある。

モデルが高確率で1着を予測できる艇は、市場（他の賭け手）も高確率で1着と見込んでいる人気艇だ。つまり低オッズになる。的中率80%でも配当が毎回1.5倍では、控除率25%の壁を超えられない。

「AUCよりROIを見ろ」というのが今の私の結論だ。

---

毎朝モデルが再訓練される音（正確にはターミナルのログが流れる音）を聞きながら、私はコーヒーを淹れる。Binary確率が画面に並ぶ。LambdaRankのスコア差が表示される。

「今日はどこに賭けるか。」

その瞬間が、一日の中で一番好きな時間だ。

---

*次回: バックテストの設計 — Walk-Forward法と過去検証の正しい使い方*
