---
title: "次世代AIアーキテクチャ：競艇AIをどこまで進化させられるか"
emoji: "🚀"
type: "tech"
topics: ["機械学習", "AI", "競艇", "Python", "アーキテクチャ"]
published: false
price: 500
---

深夜3時。

ウィスキーのグラスを脇に置いて、私はJupyter Notebookを開いた。

画面には v3 の訓練ログが残っている。AUC=0.847。バックテストROI 99.2%。実運用ROI 41.5%。

「AUCは良い。でも負けている。なぜ？」

この問いに答えを出せないまま、v3を半年以上動かしている。LightGBMの特徴量重要度を何度見ても、モデルが「なぜ」その予測を出したのかが分からない。ブラックボックスだ。

v4では、そこを変えたい。

---

---

## v3の限界を正直に整理する

```python
# v3 の成績サマリー
v3_summary = {
    "architecture":    "LightGBM (Binary Classification + LambdaRank)",
    "training_window": "21日間（毎日再訓練）",
    "backtest_auc":    0.847,
    "backtest_roi":    0.992,    # 99.2%
    "live_roi":        0.415,    # 41.5%（16件）
    "gap":             0.577,    # バックテストとの乖離

    "known_issues": [
        "展示タイム取得率12%（朝バッチの限界）",
        "EV計算不可（リアルタイムオッズ取得失敗）",
        "サンプル16件（統計的判断不可）",
        "会場別最適化なし（単一グローバルモデル）",
        "市場オッズを特徴量に使っていない（非合理）",
    ],

    "root_cause": (
        "単勝命中率82%は高いが平均配当581円。"
        "損益分岐点610円に29円足りない。"
        "モデルが人気艇（＝低オッズ）に集中しすぎている。"
    ),
}
```

ROI 41.5%の根本問題は「正解を当てているが、その配当が低すぎる」だ。

モデルが高確率と判断する＝市場も人気艇として評価している＝オッズが下がっている。

市場と同じ答えを出しているだけでは、25%の控除率に勝てない。

---

## アイデア1: 強化学習（RL）——「ベット」という意思決定を学ぶ

**現在の問題**: 分類モデルは「どの艇が勝つか」を予測するが、「いくらベットするか」を学習しない。

**RLの発想**: 「ベットする・しない・いくら賭けるか」を一つの意思決定として学習させる。

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BoatraceEnv(gym.Env):
    """
    競艇ベット環境（OpenAI Gymnasium準拠）。

    状態（State）:  各レースの特徴量ベクトル
    行動（Action）: 各艇へのベット選択（0=なし/1=500円/2=1000円/3=2000円）
    報酬（Reward）: 実際のPnL（利益または損失）

    エージェントは試行錯誤で「どの艇にいくら賭けるか」を学習する。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        race_data:         np.ndarray,   # (n_races, 6, n_features)
        results:           np.ndarray,   # (n_races,) 勝利艇番号
        payouts:           np.ndarray,   # (n_races,) 払戻オッズ
        initial_bankroll:  float = 100_000,
    ):
        super().__init__()
        self.race_data        = race_data
        self.results          = results
        self.payouts          = payouts
        self.n_races          = len(race_data)
        self.initial_bankroll = initial_bankroll
        self.n_boats          = 6
        self.n_features       = race_data.shape[-1]

        # 行動空間: 6艇 × 4択（0円/500円/1000円/2000円）
        self.action_space = spaces.MultiDiscrete([4] * self.n_boats)

        # 観測空間: 6艇分の特徴量ベクトル（フラット化）
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_boats * self.n_features,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_race = 0
        self.bankroll     = self.initial_bankroll
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """
        行動（ベット配分）を受け取り、報酬を返す。

        action: 各艇へのベット選択（0〜3）
        """
        bet_map = {0: 0, 1: 500, 2: 1_000, 3: 2_000}
        bets    = np.array([bet_map[a] for a in action])

        total_bet = bets.sum()

        # 残高を超えるベットは強制的にゼロに
        if total_bet > self.bankroll:
            bets      = np.zeros(self.n_boats, dtype=int)
            total_bet = 0

        winner   = self.results[self.current_race]
        payout   = self.payouts[self.current_race]
        winnings = bets[winner - 1] * payout if total_bet > 0 else 0
        pnl      = winnings - total_bet

        self.bankroll     += pnl
        self.current_race += 1

        terminated = (self.current_race >= self.n_races)
        truncated  = (self.bankroll <= 0)  # 破産したら終了

        return (
            self._get_obs() if not terminated else np.zeros(self.observation_space.shape),
            float(pnl),
            terminated,
            truncated,
            {"bankroll": self.bankroll, "pnl": pnl},
        )

    def _get_obs(self) -> np.ndarray:
        if self.current_race >= self.n_races:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.race_data[self.current_race].flatten().astype(np.float32)
```

RLの問題点は訓練データ量だ。PPOなどのアルゴリズムは、数百万ステップの試行錯誤が必要な場合がある。競艇のデータは1日144レースだから、4ヶ月でも約17,000レース。少ない。

**現実的な評価**: 面白いが、データが1年以上溜まってからの話だ。今すぐ実装しても過学習する。

---

## アイデア2: アンサンブル（スタッキング）

**発想**: 複数の異なるモデルの予測を組み合わせる。単体モデルより安定する。

```python
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

class BoatraceStacking:
    """
    2層スタッキングアンサンブル。

    Level 1: 異なるアルゴリズムの複数モデル（多様性確保）
    Level 2: Level1の予測を特徴量にしたメタモデル（ロジスティック回帰）

    スタッキングの利点:
    - LightGBM単体: 高速・高精度だが特定パターンに過学習しやすい
    - ランダムフォレスト: 安定しているが精度はやや低い
    - 両者を組み合わせることで、弱点を補い合う
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

        # Level 1 モデル（多様性のために異なる設定を使う）
        self.l1_models = {
            "lgbm_shallow": lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
            ),
            "lgbm_deep": lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=8,
                random_state=123,  # 異なるシードで多様性を確保
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=42,
            ),
        }

        # Level 2 メタモデル（シンプルなロジスティック回帰）
        self.l2_model = LogisticRegression(C=1.0, max_iter=1000)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoatraceStacking":
        """
        2段階でモデルを訓練する。

        Step 1: K-Foldでアウトオブフォールド（OOF）予測を生成
        Step 2: OOF予測を特徴量にメタモデルを訓練
        Step 3: 全データでLevel1モデルを再訓練（本番予測用）

        OOFを使う理由: メタモデルがLevel1の過学習予測を学ばないように。
        """
        n_samples = len(X)
        n_l1      = len(self.l1_models)
        oof_preds = np.zeros((n_samples, n_l1))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Step 1: K-FoldでOOF予測を生成
        for _, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train        = y[train_idx]

            for i, (_, model) in enumerate(self.l1_models.items()):
                model.fit(X_train, y_train)
                oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]

        # Step 2: OOF予測でメタモデルを訓練
        self.l2_model.fit(oof_preds, y)

        # Step 3: 全データでLevel1モデルを再訓練
        for _, model in self.l1_models.items():
            model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """各艇の勝利確率を予測する。"""
        if not self.is_fitted:
            raise RuntimeError("fit() を先に実行してください")

        l1_preds = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.l1_models.values()
        ])
        return self.l2_model.predict_proba(l1_preds)[:, 1]
```

---

## アイデア3: Transformer——「レースの文脈」を理解する

LightGBMは6艇の特徴量を独立に扱う。でも競艇は「6艇の相互作用」で結果が決まる。

Transformerの注意機構は、「他の艇の情報を参照しながら各艇を評価する」ことができる。

```python
import torch
import torch.nn as nn

class BoatraceTransformer(nn.Module):
    """
    競艇レース予測のためのTransformerモデル。

    入力: 6艇の特徴量シーケンス（各艇の特徴量ベクトル）
    出力: 各艇の勝利確率（Softmaxで合計1）

    LightGBMとの決定的な違い:
    「1号艇の勝率は、対面する2号艇の実力に依存して変わる」
    という相互作用をAttentionで自然に学習できる。
    """

    def __init__(
        self,
        n_features:  int,         # 入力特徴量数
        d_model:     int = 64,    # 埋め込み次元数
        n_heads:     int = 4,     # アテンションヘッド数
        n_layers:    int = 2,     # Transformerレイヤー数
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_boats = 6

        # 特徴量を埋め込み次元に投影
        self.input_proj = nn.Linear(n_features, d_model)

        # 艇番号の位置埋め込み（1〜6号艇のコース特性を学習）
        self.position_embed = nn.Embedding(self.n_boats, d_model)

        # Transformerエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # 出力層（各艇の勝利ロジット）
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 6, n_features) — 6艇分の特徴量
        戻り値: (batch_size, 6) — 各艇の勝利確率（合計1）
        """
        # 埋め込み次元に投影
        x_embed = self.input_proj(x)

        # 艇番号の位置埋め込みを加算
        positions = torch.arange(self.n_boats, device=x.device)
        x_embed   = x_embed + self.position_embed(positions).unsqueeze(0)

        # Transformerで艇間の相互作用をモデル化
        encoded = self.transformer(x_embed)  # (B, 6, d_model)

        # 各艇の勝利確率を出力
        logits = self.output_head(encoded).squeeze(-1)  # (B, 6)
        return torch.softmax(logits, dim=-1)
```

ただし、Transformerは訓練データが多く必要だ。現状4ヶ月のカードデータでは過学習のリスクが高い。1年以上データが溜まってから試すべきアーキテクチャだと判断している。

---

## アイデア4: オッズそのものを特徴量に

これは今すぐ実装できる改善だ。

現在のモデルは「市場オッズを無視して」予測を出している。でも市場オッズは数千人の集合知だ。それを無視するのは非合理だ。

```python
def add_market_features(
    df:       pd.DataFrame,
    odds_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    市場オッズから派生した特徴量を追加する。

    重要な発想:
    「モデル予測確率」と「市場の暗示確率」の差が
    Value Betのシグナルになる。差が大きいほど割安。
    """
    df = df.merge(
        odds_df[["race_id", "boat_number", "win_odds"]],
        on=["race_id", "boat_number"],
        how="left",
    )

    # 市場の暗示確率（控除率25%を考慮して調整）
    df["market_implied_prob"] = 1.0 / df["win_odds"] * 0.75

    # レース内正規化（6艇の合計が1になるよう）
    grp = df.groupby("race_id")
    df["market_prob_norm"] = (
        df["market_implied_prob"]
        / grp["market_implied_prob"].transform("sum")
    )

    # オッズの対数変換（外れ値の影響を軽減）
    df["log_odds"] = np.log(df["win_odds"].clip(lower=1.01))

    # 人気順位（1=1番人気）
    df["popularity_rank"] = grp["win_odds"].rank(
        method="min", ascending=True
    )

    return df
```

---

## アイデア5: 出走間隔・疲労指数

```python
def calculate_fatigue_features(
    df:      pd.DataFrame,
    db_path: str = "boatrace.db",
) -> pd.DataFrame:
    """
    選手の疲労度を推定する特徴量を追加する。

    直近3日間のレース数が多い選手は疲れている可能性がある。
    逆に、間隔が空きすぎると感覚が鈍る場合がある。
    """
    conn = sqlite3.connect(db_path)

    recent_counts  = []
    days_since_lst = []

    for _, row in df.iterrows():
        racer_id  = row["racer_id"]
        race_date = row["race_date"]

        # 直近3日間のレース数
        cnt = conn.execute("""
            SELECT COUNT(*) FROM race_results
            WHERE racer_id = ?
              AND race_date BETWEEN date(?, '-3 days') AND date(?, '-1 day')
        """, (racer_id, race_date, race_date)).fetchone()[0]

        # 前回出走からの日数
        last = conn.execute("""
            SELECT MAX(race_date) FROM race_results
            WHERE racer_id = ? AND race_date < ?
        """, (racer_id, race_date)).fetchone()[0]

        days = (
            (pd.to_datetime(race_date) - pd.to_datetime(last)).days
            if last else 30
        )

        recent_counts.append(cnt)
        days_since_lst.append(days)

    conn.close()

    df["races_last_3days"]    = recent_counts
    df["days_since_last_race"] = days_since_lst

    # 疲労スコア: 連戦多 → 疲労大。間隔長 → 感覚鈍化
    df["fatigue_score"] = (
        df["races_last_3days"] * 0.5
        - np.log1p(df["days_since_last_race"])
    )

    return df
```

---

## アイデア6: 気象データの高度活用

風速5m/s以上のレースは明らかに荒れる。Open-Meteo API（無料）から気象データを取得できる。

```python
import requests

def fetch_weather_data(
    stadium_lat: float,
    stadium_lon: float,
    race_date:   str,
) -> dict:
    """
    Open-Meteo API（無料）から気象データを取得する。

    対象: 風速・風向・降水量・気温
    取得時間帯: 12時〜17時（レース時間帯）の平均値
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   stadium_lat,
        "longitude":  stadium_lon,
        "hourly":     "windspeed_10m,winddirection_10m,precipitation",
        "start_date": race_date,
        "end_date":   race_date,
        "timezone":   "Asia/Tokyo",
    }

    response = requests.get(url, params=params, timeout=10)
    data     = response.json()
    hourly   = data.get("hourly", {})
    times    = hourly.get("time", [])

    # 12時〜17時のインデックスを取得
    race_hours = [i for i, t in enumerate(times) if "12:" <= t[-5:] <= "17:"]

    def mean_at(key):
        vals = hourly.get(key, [])
        selected = [vals[i] for i in race_hours if i < len(vals)]
        return np.mean(selected) if selected else None

    avg_wind = mean_at("windspeed_10m")
    avg_rain = mean_at("precipitation")

    return {
        "avg_windspeed":      avg_wind,
        "avg_winddirection":  mean_at("winddirection_10m"),
        "total_precipitation": avg_rain,
        "is_strong_wind":     (avg_wind or 0) >= 5.0,
        "is_rainy":           (avg_rain or 0) > 0.5,
    }

# 会場の座標（一部）
STADIUM_COORDS = {
    "01": (36.411, 139.238),  # 桐生
    "04": (35.576, 139.712),  # 平和島
    "19": (33.951, 130.956),  # 下関
    "24": (32.974, 129.850),  # 大村
}
```

---

## v4 実装ロードマップ

```
Phase 1（今すぐ着手・1〜2ヶ月）
├── リアルタイムオッズ取得（Playwright非同期）
├── 市場オッズを特徴量化（market_implied_prob）
└── 気象データの自動取得と特徴量化

Phase 2（3〜4ヶ月後）
├── 会場別個別モデル（300レース以上の会場）
├── 疲労指数特徴量の追加と検証
└── Kelly基準によるベット額の動的決定

Phase 3（サンプル200件到達後・6〜8ヶ月後）
├── スタッキングアンサンブル（LightGBM × RF × Logistic）
├── 3号艇フィルタの統計的再評価
└── 2連単専用モデルの追加

Phase 4（1年以上後）
├── Transformerアーキテクチャへの移行
└── 強化学習によるベット戦略の最適化
```

---

## 正直な見通し

**楽観シナリオ**: Phase 1完了でROI 85%まで改善。Phase 2完了で99%超え。1年後に月5万円の黒字。

**現実的シナリオ**: Phase 1でROI 60〜70%に改善。でも依然として赤字。競艇の25%控除率は本質的な壁であり、モデルの洗練だけでは超えられない可能性がある。

**最悪シナリオ**: 何を改善してもROI 100%を超えられない。でもそれはそれで、「競艇という市場の効率性が高い」という知見になる。

---

深夜4時になっていた。

ウィスキーのボトルが半分になっていた。画面にはv4のアーキテクチャ図がある。

「やること多すぎる」

でも、やることが多い＝まだ勝てる可能性があるということでもある。詰んでいない。

ヘッジファンドにいた頃、先輩に言われた言葉がある。「負けているときほど、ログを丁寧に読め。答えは必ずデータの中にある」。

私はまだデータを信じている。ROI 41.5%で負けていても。深夜4時でウィスキーを飲んでいても。

v4を作る。それだけだ。

---

*シリーズ完結。次の記事は実際の改善結果が出てから書く。数字が変わってから言葉を書く——それが私のルールだ。*
