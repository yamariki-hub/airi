---
title: "データ取得：公式サイトとOpen APIから競艇データを集める"
emoji: "📡"
type: "tech"
topics: ["Python", "データ分析", "競艇", "スクレイピング", "API"]
published: true
---

AIを作ると決めた翌朝、まず私がやったこと。

ラップトップを開いて、コーヒーを淹れて、「競艇 データ API」で検索した。

30分後には「データ取得より先にデータ設計をしないといけない」と気づき、さらに30分後には「そもそも何のデータが必要か整理しろ」と自分に言い聞かせ、気づいたら3時間経っていた。

AIを作る前に、データを集めなければならない。当たり前のことだけど、ここで詰まる人が多い。特に競艇は情報が散らばっているので整理が必要だ。

---

## 競艇データの全体像：3種類を理解する

競艇で予測に使えるデータは大きく3種類ある。

- **レース結果**（着順・配当・STタイム）: 取得タイミング レース後（翌日）　難易度 簡単
- **出走表（レースカード）**（選手成績・モーター・天候）: 取得タイミング 前日〜当日朝　難易度 中
- **直前情報**（展示タイム・オッズ）: 取得タイミング レース2時間前〜　難易度 難しい

この3種類は取得方法が全部違う。それぞれ順に説明していく。

**重要なのは、「AIに使う特徴量」と「取得タイミング」が完全に一致しているかを確認すること**だ。出走表は「前日に確定する情報」で、展示タイムは「当日レース直前にしかわからない情報」だ。両方を混ぜて使うと、本番予測のときに「まだ取れていないデータ」を使おうとして詰まる。私が最初にやらかしたのがまさにこれだった。

---

## 1. レース結果：Boatrace Open APIで過去データを一括取得

過去のレース結果を一括で取得するなら、これが一番楽だ。

有志が構築・運用しているOpen API（GitHubで公開）を使う。URL形式はシンプル：

```
https://boatraceopenapi.github.io/results/v2/{year}/{YYYYMMDD}.json
```

例えば2026年3月5日なら：
```
https://boatraceopenapi.github.io/results/v2/2026/20260305.json
```

実際のPythonコードはこうなる：

```python
import requests
import json
from datetime import date, timedelta
import time

def fetch_results_for_date(target_date: date) -> list[dict]:
    """
    指定日のレース結果をOpen APIから取得する。
    戻り値: レース結果のリスト（空リストの場合は非開催日）
    """
    year = target_date.year
    date_str = target_date.strftime("%Y%m%d")
    url = f"https://boatraceopenapi.github.io/results/v2/{year}/{date_str}.json"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # 404 = その日の結果がまだない（当日・未来）か非開催日
        if resp.status_code == 404:
            return []
        raise e

    results = []
    for race in data.get("results", []):
        stadium_id = str(race["race_stadium_number"]).zfill(2)
        race_number = race["race_number"]

        for r in race.get("race_results", []):
            results.append({
                "race_date": target_date,
                "stadium_id": stadium_id,
                "race_number": race_number,
                "boat_number": r["boat_number"],
                "place": r.get("result"),        # 着順（DQはNoneの場合あり）
                "start_timing": r.get("start_timing"),  # STタイム（秒）
                "racer_number": r.get("racer_number"),
            })

    return results


def fetch_results_bulk(start_date: date, end_date: date) -> list[dict]:
    """
    期間指定で一括取得。サーバー負荷配慮で1秒スリープ入れる。
    """
    all_results = []
    current = start_date
    while current <= end_date:
        daily = fetch_results_for_date(current)
        all_results.extend(daily)
        print(f"{current}: {len(daily)} rows")
        current += timedelta(days=1)
        time.sleep(1.0)  # 1秒待機（GitHub Pagesへの配慮）
    return all_results
```

**メリット:**
- GitHub Pages でホスティング → 高速・安定
- JSON形式で扱いやすい
- 2020年頃からのデータが揃っている（私は2025-07-15から使用中）

**デメリット:**
- リアルタイムではない（通常、当日の結果は翌日以降に反映）
- 展示タイム・オッズは含まれない

私は最初にこのAPIで2025年7月15日から現在までのデータを全部取得した。約8ヶ月分、34,000レース以上。これが私のデータベースのベースになっている。

---

## 2. 出走表（レースカード）：公式サイトをスクレイピング

出走表には選手情報・モーター成績・級別など、最重要の特徴量が詰まっている。これは公式サイトからスクレイピングする必要がある。

**取得対象のURL：**
```
https://www.boatrace.jp/owpc/pc/race/racelist?jcd={場コード2桁}&hd={YYYYMMDD}&rno={レース番号}
```

例：桐生（場コード01）の2026年3月5日・第3レース
```
https://www.boatrace.jp/owpc/pc/race/racelist?jcd=01&hd=20260305&rno=3
```

```python
import requests
from bs4 import BeautifulSoup
import time
from datetime import date

BOATRACE_BASE = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

def scrape_racecard(race_date: date, stadium_id: str, race_number: int) -> list[dict]:
    """
    出走表ページから1レース分のデータを取得する。
    戻り値: 6艇分のデータリスト
    """
    date_str = race_date.strftime("%Y%m%d")
    url = (
        f"{BOATRACE_BASE}/racelist"
        f"?jcd={stadium_id}&hd={date_str}&rno={race_number}"
    )

    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    boats = []

    # 各艇の行を解析（公式HTMLの構造に依存）
    for row in soup.select("tbody.is-fs12 tr.is-fs12"):
        cells = row.find_all("td")
        if len(cells) < 10:
            continue  # データが不完全な行はスキップ

        try:
            boat = {
                "boat_number": int(cells[0].get_text(strip=True)),
                "racer_number": int(cells[1].find("div", class_="is-fs11").get_text(strip=True)),
                "racer_class": cells[2].get_text(strip=True),   # A1/A2/B1/B2
                "weight": float(cells[3].get_text(strip=True).replace("kg", "")),

                # 全国成績（最新期）
                "national_win_rate": _safe_float(cells[4].get_text(strip=True)),
                "national_2rate": _safe_float(cells[5].get_text(strip=True)),

                # 当地成績
                "local_win_rate": _safe_float(cells[6].get_text(strip=True)),
                "local_2rate": _safe_float(cells[7].get_text(strip=True)),

                # 機材
                "motor_2rate": _safe_float(cells[8].get_text(strip=True)),
                "boat_2rate": _safe_float(cells[9].get_text(strip=True)),

                # 平均ST・フライング
                "avg_st": _safe_float(cells[10].get_text(strip=True) if len(cells) > 10 else None),
                "flying_count": _safe_int(cells[11].get_text(strip=True) if len(cells) > 11 else None),
                "late_count": _safe_int(cells[12].get_text(strip=True) if len(cells) > 12 else None),
            }
            boats.append(boat)
        except (ValueError, IndexError, AttributeError):
            continue  # パースエラーは無視して次へ

    return boats


def _safe_float(s) -> float | None:
    """文字列をfloatに変換。失敗したらNoneを返す"""
    try:
        return float(str(s).replace("-", "").strip()) if s else None
    except ValueError:
        return None


def _safe_int(s) -> int | None:
    try:
        return int(str(s).strip()) if s else None
    except ValueError:
        return None
```

**取得できる主なデータ:**
- 選手名・登録番号・**級別（A1/A2/B1/B2）**
- **全国勝率・全国2連率**（直近期）
- **当地勝率・当地2連率**（その競艇場での成績）
- **モーター2連率・モーター3連率**（機材の当たり外れ）
- ボート2連率
- 平均スタートタイミング・フライング回数

**スクレイピングの注意点:**
- アクセス間隔は3〜5秒空けること（公式サイトへの負荷配慮）
- 1日分（12レース × 開催場数≒10場）で120リクエスト
- 逐次で取ると1日分だけで10〜20分かかる

---

## 並列スクレイピングで速度を上げる

22,000レース分を逐次で取ったら延々と時間がかかる。私は `ThreadPoolExecutor` で並列化した。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def scrape_racecards_bulk(
    date_stadium_race_list: list[tuple],
    max_workers: int = 10,
    interval: float = 0.5,
) -> list[dict]:
    """
    並列スクレイピング。
    date_stadium_race_list: [(date, stadium_id, race_number), ...] のリスト
    max_workers: 並列数（10以下を推奨）
    interval: リクエスト間の最小待機時間（秒）
    """
    all_results = []

    def fetch_one(args):
        target_date, stadium_id, race_number = args
        time.sleep(interval)  # 過剰アクセス防止
        data = scrape_racecard(target_date, stadium_id, race_number)
        return (target_date, stadium_id, race_number, data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one, item): item
            for item in date_stadium_race_list
        }
        for future in as_completed(futures):
            try:
                target_date, sid, rno, data = future.result()
                all_results.extend(data)
                print(f"  {target_date} 場{sid} R{rno}: {len(data)}艇")
            except Exception as e:
                print(f"  エラー: {e}")

    return all_results
```

並列10ワーカーで22,000レース分を取得したところ、約4時間かかった。逐次なら数十時間かかっていた計算だ。

---

## 3. 直前情報：展示タイムとオッズはPlaywrightで取る

展示タイムとリアルタイムオッズは、レース当日にしか取得できない。

**展示タイム（直前情報ページ）のURL：**
```
https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={R}&jcd={場}&hd={日付}
```

このページは**JavaScriptで動的にレンダリング**されるため、`requests + BeautifulSoup` では取得できない。ヘッドレスブラウザの **Playwright** が必要だ。

```python
from playwright.sync_api import sync_playwright
import time

def scrape_exhibition(stadium_id: str, race_number: int, race_date: str) -> list[dict]:
    """
    展示タイム・チルト・スタート展示を取得する。
    race_date: "YYYYMMDD" 形式
    戻り値: 各艇の展示情報リスト
    """
    url = (
        f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
        f"?rno={race_number}&jcd={stadium_id}&hd={race_date}"
    )

    with sync_playwright() as p:
        # headless=True で非表示実行（サーバー向け）
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # ページが完全に読み込まれるまで待機
        page.goto(url, wait_until="networkidle", timeout=30000)

        exhibition_data = []
        # テーブルの各行を解析（各行が1艇分）
        rows = page.query_selector_all("table.is-w748 tbody tr")
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) < 5:
                continue
            try:
                boat_num = int(cells[0].inner_text().strip())
                ex_time_text = cells[4].inner_text().strip()
                ex_time = float(ex_time_text) if ex_time_text else None

                exhibition_data.append({
                    "boat_number": boat_num,
                    "exhibition_time": ex_time,
                })
            except (ValueError, IndexError):
                continue

        browser.close()
        return exhibition_data
```

**展示タイムの重要性:**

展示タイムは、レース前に実際に艇が走って計測したタイム（秒）だ。当日の機材コンディションを反映する「最新情報」であり、出走表の「モーター2連率」より直接的に状態を示す。

私のモデルでは展示タイムから以下の特徴量を計算している：
- `exhibition_time` — 生のタイム
- `ex_time_rel` — そのレース内の平均との差
- `ex_time_from_best` — レース内最速との差（小さいほど良い）
- `ex_time_rank` — レース内順位（1が最速）

重要度分析の結果、`ex_time_from_best`（重要度2.7%）と `ex_time_rank`（2.5%）が有効だとわかった。

---

## オッズの取得

単勝オッズは `requests` で取れる場合もある：

```
https://www.boatrace.jp/owpc/pc/race/odds1t?jcd={場}&hd={日付}&rno={R}
```

ただし**締切直前（レース30分前〜）でないと最終オッズにならない**点に注意。

オッズを使ったValue Bet（期待値ベースの賭け方）は理論的に有効だが、リアルタイムでオッズを取得する仕組みが必要になる。これは後の記事で詳しく書く。

---

## データベース設計：SQLiteで管理する

取得したデータはSQLiteに保存する。SQLAlchemyでテーブルを定義しておくと管理が楽だ。

```python
from sqlalchemy import (
    Column, Integer, String, Float, Date, ForeignKey, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class Race(Base):
    """レース基本情報（1レース1行）"""
    __tablename__ = "races"
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_date = Column(Date, nullable=False)
    stadium_id = Column(String(2), nullable=False)   # "01"〜"24"
    race_number = Column(Integer, nullable=False)    # 1〜12
    weather = Column(String(10))    # 天候
    wind = Column(Float)            # 風速 (m/s)
    wind_direction = Column(String(5))
    wave = Column(Float)            # 波高 (cm)
    temperature = Column(Float)     # 気温 (℃)
    water_temperature = Column(Float)  # 水温 (℃)


class RaceResult(Base):
    """レース結果（1艇1行、1レース6行）"""
    __tablename__ = "race_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    boat_number = Column(Integer, nullable=False)   # 1〜6
    course_number = Column(Integer)  # 実際のコース番号（枠番と異なる場合あり）
    place = Column(Integer)          # 着順（DQ等はNULL）
    start_timing = Column(Float)     # STタイム（秒）
    racer_number = Column(Integer)   # 選手登録番号


class RaceCard(Base):
    """出走表データ（1艇1行）"""
    __tablename__ = "race_cards"
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    boat_number = Column(Integer, nullable=False)
    racer_class = Column(String(2))      # A1/A2/B1/B2
    weight = Column(Float)               # 体重(kg)
    national_win_rate = Column(Float)    # 全国勝率
    national_2rate = Column(Float)       # 全国2連率
    local_win_rate = Column(Float)       # 当地勝率
    local_2rate = Column(Float)          # 当地2連率
    motor_2rate = Column(Float)          # モーター2連率
    motor_3rate = Column(Float)          # モーター3連率
    boat_2rate = Column(Float)           # ボート2連率
    avg_st = Column(Float)               # 平均ST
    flying_count = Column(Integer)       # フライング回数
    late_count = Column(Integer)         # 出遅れ回数
    exhibition_time = Column(Float)      # 展示タイム（秒）


class Payout(Base):
    """払い戻しデータ（配当金）"""
    __tablename__ = "payouts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    bet_type = Column(String(10))    # "win"（単勝）/"exacta"（2連単）等
    combination = Column(String(20)) # "1"/"1-2"/"1-2-3"
    payout = Column(Integer)         # 払い戻し金額（円）


# DB初期化
engine = create_engine("sqlite:///boatrace.db")
Base.metadata.create_all(engine)
```

---

## データ収集の現実：「早めに始める」が全て

私が今持っているデータ：
- レース結果: **2025-07-15〜現在**（約8ヶ月分・34,000レース超）
- 出走表: **2025-11-08〜現在**（約4ヶ月分・12,000レース分）
- 展示タイム: 毎朝スクレイピングで取得中

出走表データを後から大量に取得しようとすると時間がかかる。22,000レース分を並列10ワーカーで取得して約4時間かかった（その間ラップトップを起動したまま仕事もできず、ただ待った）。

**早めに収集を始めることを強くすすめる。** データ収集は「後でまとめてやろう」が最も損する行動だ。毎日少しずつ積み上げることが、最終的な精度の差になる。

自分でプロジェクトを始める人は、最初の日に「自動取得スクリプトを走らせる仕組み」を作ってしまうことを勧める。私は毎朝のデータ取得を全自動化している。起きたらコーヒーを淹れて、スクリプトの完了ログを確認するだけだ。

---

今日も朝6時にスクレイパーを起動した。展示タイムの取得を待ちながら、ベランダで缶コーヒーを飲んだ。競艇AIの開発で一番好きな時間かもしれない。データが積まれていく感覚が、なんとなく好きだ。

---

*次回: 特徴量エンジニアリング編 — 何をモデルに入力するか*
