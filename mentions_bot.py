"""
Kalshi Mentions Market Bot
Monitors buzz spikes (Google Trends + Reddit + RSS) for tracked entities,
then trades Kalshi markets when a spike indicates a likely outcome.

Strategy:
  - Baseline each entity's hourly mention count over a rolling window
  - When current count > BUZZ_THRESHOLD × baseline → signal
  - Use Claude Haiku to classify buzz as bullish/bearish for the outcome
  - Place paper (or live) order on the matching Kalshi market
"""

import os
import time
from flask import Flask, jsonify
import threading
import json
import uuid
import logging
import hashlib
import re
import base64
import statistics
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import httpx
import feedparser
from anthropic import Anthropic
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from risk_guard import RiskManager

load_dotenv()

# ── Quant Fund Shadow Evaluators ─────────────────────────────────────────
try:
    from bayesian_updater import BayesianUpdater
    from ensemble_model import EnsembleModel
    from time_decay_edge import calculate_time_weighted_edge
    from correlation_matrix import CorrelationTracker
    from vpin_toxicity import VPINTracker
    from market_impact import estimate_market_impact
    from feature_engine import FeatureEngine
    from portfolio_optimizer import PortfolioOptimizer
    _quant_modules_available = True
    _bayesian = BayesianUpdater()
    _ensemble = EnsembleModel()
    _correlation = CorrelationTracker()
    _vpin = VPINTracker()
    _features = FeatureEngine()
    _portfolio = PortfolioOptimizer()
except ImportError:
    _quant_modules_available = False


# ── Shadow Logging ────────────────────────────────────────────────────────────
SHADOW_LOG_FILE = os.getenv("SHADOW_LOG_FILE", "shadow_log.jsonl")

def shadow_log(opportunity: dict, taken: bool, reason: str = ""):
    entry = {"ts": time.time(), "taken": taken, "reason": reason, **opportunity}
    try:
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass


# ── Virtual Portfolio Testing ─────────────────────────────────────────────
VIRTUAL_PORTFOLIO_FILE = os.getenv("VIRTUAL_PORTFOLIO_FILE", "virtual_portfolios.jsonl")

VIRTUAL_PORTFOLIOS = [
    {"name": "aggressive", "kelly": 1.0, "min_edge": 0.02, "early_exit": 0.99},
    {"name": "moderate", "kelly": 0.5, "min_edge": 0.05, "early_exit": 0.93},
    {"name": "conservative", "kelly": 0.25, "min_edge": 0.08, "early_exit": 0.90},
    {"name": "original_v1", "kelly": 1.0, "min_edge": 0.03, "early_exit": 0.99},
    {"name": "high_edge", "kelly": 0.5, "min_edge": 0.10, "early_exit": 0.93},
    {"name": "ultra_conservative", "kelly": 0.25, "min_edge": 0.12, "early_exit": 0.90},
]

def evaluate_virtual_portfolios(opportunity: dict):
    """Evaluate what each virtual portfolio would do with this opportunity."""
    import json, time as _time
    edge = opportunity.get("edge", 0)
    price = opportunity.get("price", 0)
    results = []
    for vp in VIRTUAL_PORTFOLIOS:
        would_trade = edge >= vp["min_edge"]
        would_exit_early = price >= vp["early_exit"] * 100
        results.append({
            "portfolio": vp["name"],
            "would_trade": would_trade,
            "would_exit_early": would_exit_early,
            "kelly": vp["kelly"],
            "min_edge": vp["min_edge"],
        })
    entry = {
        "ts": _time.time(),
        "opportunity": opportunity,
        "portfolios": results,
    }
    try:
        with open(VIRTUAL_PORTFOLIO_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

# ── Multi-strike: scan ALL strikes per event/series, not just one ────────────

# ─── Regime Detection — pause trading during extreme volatility ────────────
import statistics as _stats

REGIME_WINDOW = int(os.getenv("REGIME_WINDOW", "20"))
REGIME_THRESHOLD = float(os.getenv("REGIME_THRESHOLD", "3.0"))
_regime_prices: list[float] = []

def check_regime(price: float) -> str:
    """Returns 'CALM', 'ELEVATED', or 'CRASH'. Skip trades during CRASH."""
    _regime_prices.append(price)
    if len(_regime_prices) > REGIME_WINDOW:
        _regime_prices.pop(0)
    if len(_regime_prices) < 5:
        return "CALM"
    rets = [(b - a) / a for a, b in zip(_regime_prices[:-1], _regime_prices[1:])]
    if not rets:
        return "CALM"
    mu = _stats.mean(rets)
    sd = _stats.stdev(rets) if len(rets) > 1 else 0.01
    z = abs(rets[-1] - mu) / max(sd, 0.0001)
    if z > REGIME_THRESHOLD:
        return "CRASH"
    elif z > REGIME_THRESHOLD * 0.6:
        return "ELEVATED"
    return "CALM"



# ── Early Exit Logic ─────────────────────────────────────────────────────────
EARLY_EXIT_THRESHOLD = float(os.getenv("EARLY_EXIT_THRESHOLD", "0.93"))

def should_early_exit(current_price_cents: float) -> bool:
    """Exit position early at 93c+ to lock in profit instead of holding to settlement."""
    return current_price_cents >= EARLY_EXIT_THRESHOLD * 100

# ── Circuit Breakers ─────────────────────────────────────────────────────────
CONSECUTIVE_LOSS_PAUSE = int(os.getenv("CONSECUTIVE_LOSS_PAUSE", "3"))
DAILY_DRAWDOWN_PAUSE_PCT = float(os.getenv("DAILY_DRAWDOWN_PAUSE_PCT", "0.05"))

_consecutive_losses = 0
_daily_pnl = 0.0
_circuit_paused_until = 0

def check_circuit_breaker() -> bool:
    """Returns True if trading should be paused."""
    import time as _time
    global _consecutive_losses, _daily_pnl, _circuit_paused_until
    if _time.time() < _circuit_paused_until:
        return True
    if _consecutive_losses >= CONSECUTIVE_LOSS_PAUSE:
        return True
    # Use PAPER_BALANCE if available, else 5000
    _balance = globals().get("PAPER_BALANCE", 2000)
    if _daily_pnl < -DAILY_DRAWDOWN_PAUSE_PCT * _balance:
        return True
    return False

def record_trade_result(won: bool, pnl: float):
    """Update circuit breaker state after each trade result."""
    global _consecutive_losses, _daily_pnl
    _daily_pnl += pnl
    if won:
        _consecutive_losses = 0
    else:
        _consecutive_losses += 1
MULTI_STRIKE = os.getenv("MULTI_STRIKE", "true").lower() == "true"
# When fetching markets, iterate through ALL contracts in each series/event
# and evaluate each strike independently. No single-ticker filtering.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

class Config:
    PAPER_MODE:             bool  = os.getenv("PAPER_MODE", "true").lower() == "true"
    PAPER_BALANCE:          float = float(os.getenv("PAPER_BALANCE", "2000"))
    KALSHI_API_KEY:         str   = os.getenv("KALSHI_API_KEY", "")
    KALSHI_KEY_ID:          str   = os.getenv("KALSHI_KEY_ID", "")
    ANTHROPIC_API_KEY:      str   = os.getenv("ANTHROPIC_API_KEY", "")
    REDDIT_CLIENT_ID:       str   = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET:   str   = os.getenv("REDDIT_CLIENT_SECRET", "")

    BUZZ_THRESHOLD:         float = float(os.getenv("BUZZ_THRESHOLD", "2.5"))
    BUZZ_WINDOW_HOURS:      int   = int(os.getenv("BUZZ_WINDOW_HOURS", "168"))   # 7-day baseline
    MIN_BASELINE_SAMPLES:   int   = int(os.getenv("MIN_BASELINE_SAMPLES", "6"))

    MIN_EDGE:               float = float(os.getenv("MIN_EDGE", "0.05"))         # 5¢ minimum edge
    MAKER_FEE:              float = float(os.getenv("MAKER_FEE", "0.0175"))
    MIN_PRICE:              int   = int(os.getenv("MIN_PRICE", "15"))            # don't buy >85¢ YES
    MAX_PRICE:              int   = int(os.getenv("MAX_PRICE", "85"))
    BET_SIZE_USD:           float = float(os.getenv("BET_SIZE_USD", "10.0"))
    KELLY_FRACTION:         float = float(os.getenv("KELLY_FRACTION", "0.25"))
    MAX_OPEN_POSITIONS:     int   = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    POLL_INTERVAL_SEC:      int   = int(os.getenv("POLL_INTERVAL_SEC", "900"))   # 15 min

    KALSHI_BASE:            str   = "https://api.elections.kalshi.com/trade-api/v2"

# ── Entity → Kalshi Market Mapping ────────────────────────────────────────────

# Each entry: entity_name → list of (series_ticker, outcome_direction, keywords)
# outcome_direction: "YES" = high price means entity does well, "NO" = inverse
ENTITY_MAP = {
    "bitcoin": {
        "series": ["KXBTCD", "KXBTCW"],
        "direction": "YES",   # buzz about BTC → buy YES on high-price markets
        "keywords": ["bitcoin", "btc", "crypto rally", "bitcoin price"],
        "bearish_keywords": ["bitcoin crash", "btc dump", "bitcoin bear", "crypto crash"],
        "subreddits": ["r/Bitcoin", "r/CryptoCurrency"],
    },
    "ethereum": {
        "series": ["KXETHD"],
        "direction": "YES",
        "keywords": ["ethereum", "eth", "ether"],
        "bearish_keywords": ["eth crash", "ethereum dump"],
        "subreddits": ["r/ethereum", "r/CryptoCurrency"],
    },
    "trump": {
        "series": ["KXPRESAPP"],   # presidential approval
        "direction": "YES",
        "keywords": ["trump", "donald trump", "president trump"],
        "bearish_keywords": ["trump impeach", "trump resign", "trump approval drop"],
        "subreddits": ["r/politics", "r/news"],
    },
    "fed": {
        "series": ["KXFED"],
        "direction": "NO",    # Fed rate hike buzz → buy NO on "rate stays same"
        "keywords": ["federal reserve", "fed rate", "interest rate hike", "fomc"],
        "bearish_keywords": ["rate cut", "fed cut", "dovish fed"],
        "subreddits": ["r/investing", "r/Economics"],
    },
    "sp500": {
        "series": ["KXSPX"],
        "direction": "YES",
        "keywords": ["s&p 500", "sp500", "stock market rally", "bull market"],
        "bearish_keywords": ["market crash", "stock market crash", "bear market", "recession"],
        "subreddits": ["r/investing", "r/wallstreetbets", "r/stocks"],
    },
}

# RSS feeds to monitor (already proven in news bot)
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # WSJ Markets
    "https://rss.politico.com/politics-news.rss",
]

# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class MentionSample:
    ts: datetime
    count: int
    entity: str

@dataclass
class BuzzSignal:
    entity: str
    current_count: int
    baseline: float
    buzz_ratio: float
    sentiment: str        # "bullish" | "bearish" | "neutral"
    confidence: float     # 0-1
    sample_headlines: list[str]
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class KalshiMarket:
    ticker: str
    title: str
    yes_price: int
    no_price: int
    volume: int
    close_time: datetime

# ── Mention History ────────────────────────────────────────────────────────────

class MentionHistory:
    """Rolling window of hourly mention counts per entity."""

    def __init__(self):
        self.samples: dict[str, list[MentionSample]] = {e: [] for e in ENTITY_MAP}
        self._seen_ids: set[str] = set()

    def add_sample(self, entity: str, count: int):
        now = datetime.now(timezone.utc)
        self.samples[entity].append(MentionSample(ts=now, count=count, entity=entity))
        # Prune old samples
        cutoff = now - timedelta(hours=Config.BUZZ_WINDOW_HOURS)
        self.samples[entity] = [s for s in self.samples[entity] if s.ts > cutoff]

    def baseline(self, entity: str) -> Optional[float]:
        samples = self.samples[entity]
        if len(samples) < Config.MIN_BASELINE_SAMPLES:
            return None
        counts = [s.count for s in samples[:-1]]  # exclude latest
        return statistics.median(counts) if counts else None

    def is_seen(self, item_id: str) -> bool:
        if item_id in self._seen_ids:
            return True
        self._seen_ids.add(item_id)
        if len(self._seen_ids) > 50_000:
            self._seen_ids = set(list(self._seen_ids)[-25_000:])
        return False


# ── Paper Ledger ──────────────────────────────────────────────────────────────

class PaperLedger:
    def __init__(self):
        self.balance = Config.PAPER_BALANCE
        self.trades: list[dict] = []
        self.open_positions: dict[str, dict] = {}  # ticker → position

    def open_position(self, ticker: str, side: str, price: int, contracts: int,
                       entity: str, signal: BuzzSignal) -> bool:
        if len(self.open_positions) >= Config.MAX_OPEN_POSITIONS:
            log.info(f"[PAPER] Max open positions reached, skipping {ticker}")
            return False
        cost = price * contracts / 100
        if cost > self.balance:
            log.info(f"[PAPER] Insufficient balance for {ticker}")
            return False
        self.balance -= cost
        self.open_positions[ticker] = {
            "ticker": ticker, "side": side, "price": price,
            "contracts": contracts, "cost": cost, "entity": entity,
            "ts": datetime.now(timezone.utc).isoformat(),
            "buzz_ratio": signal.buzz_ratio,
        }
        self.trades.append({"action": "OPEN", **self.open_positions[ticker]})
        log.info(f"[PAPER] OPEN {side} {ticker} @ {price}¢ × {contracts} = ${cost:.2f} | "
                 f"entity={entity} buzz={signal.buzz_ratio:.1f}x | balance=${self.balance:.2f}")
        return True

    def close_position(self, ticker: str, exit_price: int, reason: str = ""):
        pos = self.open_positions.pop(ticker, None)
        if not pos:
            return
        pnl = (exit_price - pos["price"]) * pos["contracts"] / 100
        if pos["side"] == "NO":
            pnl = (pos["price"] - exit_price) * pos["contracts"] / 100
        self.balance += pos["cost"] + pnl
        self.trades.append({"action": "CLOSE", "ticker": ticker,
                             "exit_price": exit_price, "pnl": pnl, "reason": reason})
        log.info(f"[PAPER] CLOSE {ticker} @ {exit_price}¢ | PnL=${pnl:+.2f} | "
                 f"reason={reason} | balance=${self.balance:.2f}")


# ── Kalshi Client ─────────────────────────────────────────────────────────────

class KalshiClient:
    def __init__(self):
        self._client = httpx.Client(timeout=15)
        self._private_key = self._load_private_key()

    @staticmethod
    def _load_private_key():
        pem_str = os.getenv("KALSHI_PRIVATE_KEY", "")
        if not pem_str:
            return None
        if "\\n" in pem_str:
            pem_str = pem_str.replace("\\n", "\n")
        return serialization.load_pem_private_key(pem_str.encode(), password=None)

    def _get_auth_headers(self, method: str, path: str) -> dict:
        if not self._private_key:
            return {"Content-Type": "application/json"}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + "/trade-api/v2" + path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
        return {
            "Kalshi-Access-Key": os.getenv("KALSHI_API_KEY", ""),
            "Kalshi-Access-Signature": base64.b64encode(sig).decode(),
            "Kalshi-Access-Timestamp": ts,
            "Content-Type": "application/json",
        }

    def get_markets_for_series(self, series_ticker: str) -> list[KalshiMarket]:
        try:
            r = self._client.get(
                f"{Config.KALSHI_BASE}/markets",
                params={"series_ticker": series_ticker, "status": "open"},
                headers=self._get_auth_headers("GET", "/markets"),
            )
            r.raise_for_status()
            markets = []
            for m in r.json().get("markets", []):
                yes_ask = m.get("yes_ask", 0)
                no_ask = m.get("no_ask", 0)
                close_str = m.get("close_time", "")
                try:
                    close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                except Exception:
                    close_dt = datetime.now(timezone.utc) + timedelta(hours=24)
                markets.append(KalshiMarket(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    yes_price=yes_ask,
                    no_price=no_ask,
                    volume=m.get("volume", 0),
                    close_time=close_dt,
                ))
            return markets
        except Exception as e:
            log.warning(f"get_markets_for_series({series_ticker}): {e}")
            return []

    def place_order(self, ticker: str, side: str, count: int, price: int) -> bool:
        if Config.PAPER_MODE:
            return True
        try:
            r = self._client.post(
                f"{Config.KALSHI_BASE}/portfolio/orders",
                json={"ticker": ticker, "action": "buy", "side": side.lower(),
                      "count": count, "type": "limit", "yes_price": price if side == "YES" else 100 - price,
                      "client_order_id": str(uuid.uuid4())},
                headers=self._get_auth_headers("POST", "/portfolio/orders"),
            )
            r.raise_for_status()
            log.info(f"[LIVE] Order placed: {side} {ticker} @ {price}¢ × {count}")
            return True
        except Exception as e:
            log.error(f"place_order failed: {e}")
            return False


# ── Mention Scanner ───────────────────────────────────────────────────────────

class MentionScanner:
    def __init__(self):
        self._history = MentionHistory()
        self._reddit_token: Optional[str] = None
        self._reddit_token_expiry: Optional[datetime] = None

    # ── Reddit ──────────────────────────────────────────────────────────────

    def _get_reddit_token(self) -> Optional[str]:
        if not Config.REDDIT_CLIENT_ID:
            return None
        now = datetime.now(timezone.utc)
        if self._reddit_token and self._reddit_token_expiry and now < self._reddit_token_expiry:
            return self._reddit_token
        try:
            r = httpx.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(Config.REDDIT_CLIENT_ID, Config.REDDIT_CLIENT_SECRET),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "kalshi-mentions-bot/1.0"},
                timeout=10,
            )
            r.raise_for_status()
            d = r.json()
            self._reddit_token = d["access_token"]
            self._reddit_token_expiry = now + timedelta(seconds=d.get("expires_in", 3600) - 60)
            return self._reddit_token
        except Exception as e:
            log.warning(f"Reddit auth failed: {e}")
            return None

    def _search_reddit(self, entity: str, keywords: list[str], subreddits: list[str]) -> tuple[int, list[str]]:
        token = self._get_reddit_token()
        if not token:
            return 0, []
        query = " OR ".join(f'"{kw}"' for kw in keywords[:3])
        sub = "+".join(s.replace("r/", "") for s in subreddits)
        try:
            r = httpx.get(
                f"https://oauth.reddit.com/r/{sub}/search",
                params={"q": query, "sort": "new", "t": "hour", "limit": 25, "restrict_sr": "1"},
                headers={"Authorization": f"Bearer {token}", "User-Agent": "kalshi-mentions-bot/1.0"},
                timeout=10,
            )
            r.raise_for_status()
            posts = r.json().get("data", {}).get("children", [])
            titles = [p["data"]["title"] for p in posts if not self._history.is_seen(p["data"]["id"])]
            return len(posts), titles
        except Exception as e:
            log.warning(f"Reddit search failed for {entity}: {e}")
            return 0, []

    # ── RSS ─────────────────────────────────────────────────────────────────

    def _scan_rss(self, entity: str, keywords: list[str]) -> tuple[int, list[str]]:
        kw_lower = [k.lower() for k in keywords]
        count = 0
        titles = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    item_id = hashlib.md5(entry.get("link", entry.get("title", "")).encode()).hexdigest()
                    text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                    if any(kw in text for kw in kw_lower):
                        if not self._history.is_seen(f"rss_{item_id}"):
                            count += 1
                            titles.append(entry.get("title", "")[:120])
            except Exception as e:
                log.warning(f"RSS parse error {feed_url}: {e}")
        return count, titles

    # ── Combined Scan ────────────────────────────────────────────────────────

    def scan_entity(self, entity: str) -> tuple[int, list[str]]:
        cfg = ENTITY_MAP[entity]
        keywords = cfg["keywords"]
        subreddits = cfg.get("subreddits", [])

        reddit_count, reddit_titles = self._search_reddit(entity, keywords, subreddits)
        rss_count, rss_titles = self._scan_rss(entity, keywords)

        total = reddit_count + rss_count
        titles = (reddit_titles + rss_titles)[:10]
        return total, titles

    def check_buzz(self, entity: str) -> Optional[BuzzSignal]:
        count, headlines = self.scan_entity(entity)
        self._history.add_sample(entity, count)
        baseline = self._history.baseline(entity)

        if baseline is None:
            log.info(f"[{entity.upper()}] count={count} (building baseline, need "
                     f"{Config.MIN_BASELINE_SAMPLES} samples)")
            return None

        if baseline == 0:
            baseline = 0.5  # avoid div by zero

        buzz_ratio = count / baseline
        log.info(f"[{entity.upper()}] count={count} baseline={baseline:.1f} ratio={buzz_ratio:.2f}x")

        if buzz_ratio < Config.BUZZ_THRESHOLD:
            return None

        return BuzzSignal(
            entity=entity,
            current_count=count,
            baseline=baseline,
            buzz_ratio=buzz_ratio,
            sentiment="unknown",
            confidence=0.0,
            sample_headlines=headlines,
        )


# ── Sentiment Analysis ────────────────────────────────────────────────────────

class SentimentAnalyzer:
    def __init__(self):
        if Config.ANTHROPIC_API_KEY:
            self._client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        else:
            self._client = None

    def classify(self, entity: str, headlines: list[str], cfg: dict) -> tuple[str, float]:
        """Returns (sentiment, confidence) where sentiment is 'bullish'|'bearish'|'neutral'."""
        if not headlines:
            return "neutral", 0.3

        # Fast keyword check first
        text = " ".join(headlines).lower()
        bearish_hits = sum(1 for kw in cfg.get("bearish_keywords", []) if kw.lower() in text)
        bullish_hits = sum(1 for kw in cfg.get("keywords", []) if kw.lower() in text)

        if not self._client:
            # Keyword-only fallback
            if bearish_hits > bullish_hits:
                return "bearish", min(0.6, 0.4 + bearish_hits * 0.05)
            elif bullish_hits > 0:
                return "bullish", min(0.6, 0.4 + bullish_hits * 0.05)
            return "neutral", 0.3

        # Claude Haiku classification
        direction = cfg.get("direction", "YES")
        try:
            prompt = (
                f"You are analyzing news headlines to predict a financial outcome.\n"
                f"Entity: {entity}\n"
                f"Market direction: '{direction}' means betting that {entity} performs well/increases.\n\n"
                f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines[:6]) + "\n\n"
                f"Classify the overall sentiment as exactly one of: bullish, bearish, neutral\n"
                f"Also give confidence 0.0-1.0.\n"
                f"Respond in JSON: {{\"sentiment\": \"bullish\", \"confidence\": 0.8}}"
            )
            msg = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(msg.content[0].text.strip())
            return result.get("sentiment", "neutral"), float(result.get("confidence", 0.5))
        except Exception as e:
            log.warning(f"Haiku sentiment error: {e}")
            if bearish_hits > bullish_hits:
                return "bearish", 0.5
            return "bullish", 0.5


# ── Trade Logic ───────────────────────────────────────────────────────────────

def pick_market_and_side(signal: BuzzSignal, markets: list[KalshiMarket],
                          cfg: dict) -> Optional[tuple[KalshiMarket, str, int]]:
    """Given a buzz signal and list of open markets, choose (market, side, price)."""
    direction = cfg["direction"]  # "YES" or "NO"

    # Choose side based on sentiment vs direction
    if signal.sentiment == "bullish":
        side = direction  # bullish on entity → trade in its "good" direction
    elif signal.sentiment == "bearish":
        side = "NO" if direction == "YES" else "YES"
    else:
        return None  # neutral — skip

    now = datetime.now(timezone.utc)

    # Filter markets that close within 7 days and have reasonable liquidity
    candidates = [
        m for m in markets
        if m.close_time > now + timedelta(hours=2)
        and m.close_time < now + timedelta(days=7)
        and m.volume > 50
    ]
    if not candidates:
        # Fallback: any market closing within 30 days
        candidates = [m for m in markets if m.close_time > now + timedelta(hours=6)]

    if not candidates:
        return None

    # Prefer markets where the price gives us best edge
    # For YES side: want low price (underpriced outcome)
    # For NO side: want low no_price
    for market in sorted(candidates, key=lambda m: m.yes_price if side == "YES" else m.no_price):
        price = market.yes_price if side == "YES" else market.no_price
        if Config.MIN_PRICE <= price <= Config.MAX_PRICE:
            # Fee-aware check: implied edge must exceed maker fee
            implied_edge = signal.buzz_ratio * 0.05 - (price / 100)  # rough edge estimate
            ev_after_fees = implied_edge - Config.MAKER_FEE if implied_edge > 0 else -Config.MAKER_FEE
            if ev_after_fees <= 0 and signal.buzz_ratio < 3.0:
                log.info(f"Skipping {market.ticker}: negative EV after {Config.MAKER_FEE*100}% fee")
                continue
            # Kelly: use confidence as model_prob proxy
            market_prob = price / 100
            kelly_f = max(0, (signal.confidence - market_prob) / (1 - market_prob)) if market_prob < 1 else 0
            kelly_bet = max(1, min(Config.PAPER_BALANCE * kelly_f * Config.KELLY_FRACTION, Config.BET_SIZE_USD * 5))
            contracts = max(1, int(kelly_bet * 100 / price))
            return market, side, price

    return None


# ── Main Loop ─────────────────────────────────────────────────────────────────

# ── Stats HTTP server ─────────────────────────────────────────────────────────
_stats_app = Flask(__name__)
_bot_stats = {"trades": 0, "wins": 0, "pnl": 0.0, "balance": 0.0, "start": time.time()}

@_stats_app.route("/stats")
def _stats_endpoint():
    t = _bot_stats
    total = t["trades"]
    return jsonify({"bot": "kalshi-mentions-bot", "paper_mode": True,
        "balance": t["balance"], "trades": total, "wins": t["wins"],
        "losses": total - t["wins"], "win_rate": round(t["wins"]/max(total,1), 4),
        "pnl": t["pnl"], "uptime_hours": round((time.time()-t["start"])/3600, 2)})

@_stats_app.route("/health")
def _health_endpoint():
    return jsonify({"status": "ok"})

def _run_stats_server():
    _stats_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


def main():
    log.info("=" * 60)
    log.info("Kalshi Mentions Market Bot starting")
    log.info(f"  Paper mode:     {Config.PAPER_MODE}")
    log.info(f"  Buzz threshold: {Config.BUZZ_THRESHOLD}x baseline")
    log.info(f"  Entities:       {list(ENTITY_MAP.keys())}")
    log.info(f"  Bet size:       ${Config.BET_SIZE_USD}")
    log.info(f"  Poll interval:  {Config.POLL_INTERVAL_SEC}s")
    log.info("=" * 60)

    scanner = MentionScanner()
    analyzer = SentimentAnalyzer()
    kalshi = KalshiClient()
    ledger = PaperLedger()
    risk_manager = RiskManager(starting_balance=Config.PAPER_BALANCE)
    _bot_stats['balance'] = ledger.balance
    threading.Thread(target=_run_stats_server, daemon=True).start()

    cycle = 0
    while True:
        cycle += 1
        log.info(f"── Cycle {cycle} ──────────────────────────")

        for entity, cfg in ENTITY_MAP.items():
            try:
                signal = scanner.check_buzz(entity)
                if signal is None:
                    continue

                log.info(f"[BUZZ] {entity.upper()} spike {signal.buzz_ratio:.1f}x | "
                         f"{signal.current_count} mentions | headlines: {signal.sample_headlines[:2]}")

                # Sentiment analysis
                sentiment, confidence = analyzer.classify(entity, signal.sample_headlines, cfg)
                signal.sentiment = sentiment
                signal.confidence = confidence
                log.info(f"[SENTIMENT] {entity}: {sentiment} ({confidence:.0%} confidence)")

                if confidence < 0.5 or sentiment == "neutral":
                    log.info(f"[SKIP] {entity}: low confidence or neutral sentiment")
                    shadow_log({"bot": "mentions", "entity": entity, "sentiment": sentiment, "confidence": confidence}, taken=False, reason="low confidence or neutral")
                    evaluate_virtual_portfolios({"bot": "mentions", "entity": entity, "sentiment": sentiment, "confidence": confidence})
                    if _quant_modules_available:
                        try:
                            _features.extract({"price": locals().get("price", 0), "volume": locals().get("volume", 0), "bid": locals().get("bid", 0), "ask": locals().get("ask", 0)})
                            _bayesian.update(locals().get("market_id", locals().get("ticker", "unknown")), locals().get("price", 0), time.time())
                            _td_edge = calculate_time_weighted_edge(locals().get("edge", 0), locals().get("minutes_remaining", locals().get("time_remaining", 15)), 15)
                            _vpin.update(locals().get("price", 0), locals().get("volume", 0))
                            _mi = estimate_market_impact(locals().get("contracts", 1), locals().get("volume", 100))
                        except:
                            pass
                    continue

                # Find matching Kalshi markets
                for series_ticker in cfg["series"]:
                    markets = kalshi.get_markets_for_series(series_ticker)
                    if not markets:
                        continue

                    result = pick_market_and_side(signal, markets, cfg)
                    if result is None:
                        log.info(f"[SKIP] {entity}/{series_ticker}: no suitable market found")
                        continue

                    market, side, price = result

                    # Check if already in this position
                    if market.ticker in ledger.open_positions:
                        log.info(f"[SKIP] Already holding {market.ticker}")
                        continue

                    # Kelly criterion sizing
                    market_prob = price / 100
                    kelly_f = max(0, (confidence - market_prob) / (1 - market_prob)) if market_prob < 1 else 0
                    kelly_bet = max(1, min(ledger.balance * kelly_f * Config.KELLY_FRACTION, Config.BET_SIZE_USD * 5))
                    contracts = max(1, int(kelly_bet * 100 / price))
                    log.info(f"[SIGNAL] {entity} → {side} {market.ticker} "
                             f"'{market.title[:60]}' @ {price}¢ × {contracts} kelly_f={kelly_f:.3f}")

                    # Risk guard check
                    if not Config.PAPER_MODE:
                        allowed, rg_reason, capped = risk_manager.pre_trade_check(
                            entity, price, contracts, side.lower(),
                            bot_name="mentions-bot")
                        if not allowed:
                            log.warning(f"Risk guard blocked: {rg_reason}")
                            continue
                        contracts = capped or contracts
                    else:
                        allowed, rg_reason, capped = risk_manager.pre_trade_check(
                            entity, price, contracts, side.lower(),
                            bot_name="mentions-bot")
                        if not allowed:
                            log.info(f"[PAPER] Risk guard would block: {rg_reason}")

                    # ── Regime detection ──
                    regime = check_regime(float(price))
                    if regime == "CRASH":
                        log.warning("REGIME CRASH on kalshi_mentions_bot — skipping trade")
                        shadow_log({"bot": "kalshi_mentions_bot", "regime": regime}, taken=False, reason="crash regime")
                        evaluate_virtual_portfolios({"bot": "kalshi_mentions_bot", "regime": regime})
                        continue
                    shadow_log({"bot": "mentions", "ticker": market.ticker, "entity": entity, "side": side, "price": price, "confidence": confidence, "buzz_ratio": signal.buzz_ratio}, taken=True)
                    evaluate_virtual_portfolios({"bot": "mentions", "ticker": market.ticker, "entity": entity, "side": side, "price": price, "confidence": confidence, "buzz_ratio": signal.buzz_ratio})
                    if Config.PAPER_MODE:
                        ledger.open_position(market.ticker, side, price, contracts, entity, signal)
                    else:
                        if kalshi.place_order(market.ticker, side, contracts, price):
                            log.info(f"[LIVE] Placed {side} {market.ticker} @ {price}¢ × {contracts}")

            except Exception as e:
                log.error(f"Entity loop error ({entity}): {e}", exc_info=True)

        # Summary
        open_count = len(ledger.open_positions)
        total_trades = len([t for t in ledger.trades if t["action"] == "OPEN"])
        _bot_stats['balance'] = ledger.balance
        _bot_stats['trades'] = sum(1 for t in ledger.trades if t['action'] == 'OPEN')
        _bot_stats['wins'] = sum(1 for t in ledger.trades if t['action'] == 'CLOSE' and t.get('pnl', 0) > 0)
        _bot_stats['pnl'] = sum(t.get('pnl', 0) for t in ledger.trades if t['action'] == 'CLOSE')
        closed_pnl = sum(t.get("pnl", 0) for t in ledger.trades if t["action"] == "CLOSE")
        log.info(f"[SUMMARY] Balance=${ledger.balance:.2f} | Open={open_count} | "
                 f"Trades={total_trades} | Closed PnL=${closed_pnl:+.2f}")

        time.sleep(Config.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
