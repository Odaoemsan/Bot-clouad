"""
market_data.py  —  الطبقة الأولى: محرك البيانات
=================================================
مسؤول عن:
  • جلب OHLCV من yfinance (افتراضي) أو Polygon.io (للباكتست الموسّع)
  • فلتر الأخبار الاقتصادية عالية التأثير (ForexFactory JSON)
  • فلتر جلسات التداول (Kill Zones: London + New York)
  • تعريف الأصول والإعدادات العامة

لا يحتوي هذا الملف على أي منطق تحليلي — مجرد بيانات نظيفة للطبقات العليا.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# تعريف الأصول
# ---------------------------------------------------------------------------
ASSETS: dict[str, dict] = {
    "الذهب":   {"yf": "GC=F",      "poly": "C:XAUUSD", "currencies": ["USD"],         "pip": 0.1},
    "EUR/USD": {"yf": "EURUSD=X",  "poly": "C:EURUSD", "currencies": ["EUR", "USD"],  "pip": 0.0001},
    "GBP/USD": {"yf": "GBPUSD=X",  "poly": "C:GBPUSD", "currencies": ["GBP", "USD"],  "pip": 0.0001},
    "USD/JPY": {"yf": "USDJPY=X",  "poly": "C:USDJPY", "currencies": ["USD", "JPY"],  "pip": 0.01},
    "AUD/USD": {"yf": "AUDUSD=X",  "poly": "C:AUDUSD", "currencies": ["AUD", "USD"],  "pip": 0.0001},
    "USD/CAD": {"yf": "USDCAD=X",  "poly": "C:USDCAD", "currencies": ["USD", "CAD"],  "pip": 0.0001},
    "USD/CHF": {"yf": "USDCHF=X",  "poly": "C:USDCHF", "currencies": ["USD", "CHF"],  "pip": 0.0001},
    "NZD/USD": {"yf": "NZDUSD=X",  "poly": "C:NZDUSD", "currencies": ["NZD", "USD"],  "pip": 0.0001},
}

# ---------------------------------------------------------------------------
# إعدادات المخاطرة والحساب
# ---------------------------------------------------------------------------
ACCOUNT_BALANCE_USD: float = 1000.0
RISK_PERCENT: float = 1.0          # % من الحساب لكل صفقة
NEWS_BUFFER_MIN: int = 60           # دقائق قبل/بعد خبر عالي التأثير

# ---------------------------------------------------------------------------
# Kill Zones (UTC) — الوقت الوحيد المسموح بالدخول فيه
# ---------------------------------------------------------------------------
KILL_ZONES_UTC: list[tuple[int, int]] = [
    (7, 10),    # London Kill Zone
    (13, 16),   # New York Kill Zone
]

# ---------------------------------------------------------------------------
# جلب البيانات — yfinance
# ---------------------------------------------------------------------------

def fetch_yf(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """
    جلب OHLCV من Yahoo Finance.
    قيود yfinance: 15m → آخر 60 يوم فقط | 1h → آخر 730 يوم | 1d → غير محدود
    """
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        # توحيد الـ timezone → UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df
    except Exception as e:
        logger.error(f"[yfinance] خطأ في {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# جلب البيانات — Polygon.io
# ---------------------------------------------------------------------------

def fetch_polygon(symbol: str, timespan: str, multiplier: int,
                  from_date: str, to_date: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    جلب OHLCV من Polygon.io (للباكتست الموسّع — سنة كاملة).
    timespan: 'minute' | 'hour' | 'day'
    multiplier: 15 للـ 15m, 1 للـ 1h أو daily
    """
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{multiplier}/{timespan}/{from_date}/{to_date}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    all_results = []
    url = base_url

    for _ in range(15):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[Polygon] خطأ في {symbol}: {e}")
            time.sleep(2)
            break

        if data.get("status") == "ERROR":
            logger.error(f"[Polygon] {symbol}: {data.get('error', data)}")
            break

        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)

        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": api_key}
            time.sleep(0.25)
        else:
            break

    if not all_results:
        return None

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("datetime")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                             "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index().dropna()


# ---------------------------------------------------------------------------
# جلب فريمات الأصل الواحد (لحظي / بوت حي)
# ---------------------------------------------------------------------------

class AssetData:
    """حاوية بيانات أصل واحد عبر كل الفريمات."""
    __slots__ = ("name", "d1", "h4", "h1", "m15")

    def __init__(self, name: str,
                 d1: Optional[pd.DataFrame],
                 h4: Optional[pd.DataFrame],
                 h1: Optional[pd.DataFrame],
                 m15: Optional[pd.DataFrame]):
        self.name = name
        self.d1  = d1
        self.h4  = h4
        self.h1  = h1
        self.m15 = m15

    @property
    def is_valid(self) -> bool:
        return all(
            df is not None and len(df) >= min_bars
            for df, min_bars in [
                (self.d1,  200),
                (self.h4,   50),
                (self.m15,  30),
            ]
        )


def resample_to_h4(df_1h: pd.DataFrame) -> Optional[pd.DataFrame]:
    """تحويل بيانات 1h إلى H4 بالإجماع."""
    if df_1h is None or df_1h.empty:
        return None
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    result = df_1h.resample("4h").agg(agg).dropna()
    return result if not result.empty else None


def load_asset_live(name: str) -> AssetData:
    """جلب كل فريمات الأصل للاستخدام الحي (بوت تلغرام)."""
    cfg = ASSETS[name]
    sym = cfg["yf"]

    d1  = fetch_yf(sym, "1d",  "730d")
    h1  = fetch_yf(sym, "1h",  "60d")
    m15 = fetch_yf(sym, "15m", "8d")
    h4  = resample_to_h4(h1) if h1 is not None else None

    return AssetData(name=name, d1=d1, h4=h4, h1=h1, m15=m15)


def load_asset_polygon(name: str, api_key: str,
                       backtest_days: int = 365) -> AssetData:
    """جلب كل فريمات الأصل من Polygon للباكتست الموسّع."""
    cfg = ASSETS[name]
    sym = cfg["poly"]

    to_date  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_15m = (datetime.now(timezone.utc) - timedelta(days=backtest_days)).strftime("%Y-%m-%d")
    from_htf = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")

    logger.info(f"[{name}] جلب Polygon 15m...")
    m15 = fetch_polygon(sym, "minute", 15, from_15m, to_date, api_key)
    time.sleep(0.3)

    logger.info(f"[{name}] جلب Polygon 1h...")
    h1 = fetch_polygon(sym, "hour", 1, from_htf, to_date, api_key)
    time.sleep(0.3)

    logger.info(f"[{name}] جلب Polygon Daily...")
    d1 = fetch_polygon(sym, "day", 1, from_htf, to_date, api_key)
    time.sleep(0.3)

    h4 = resample_to_h4(h1) if h1 is not None else None

    for tag, df in [("D1", d1), ("H4", h4), ("M15", m15)]:
        bars = len(df) if df is not None else 0
        logger.info(f"  → {tag}: {bars} شمعة")

    return AssetData(name=name, d1=d1, h4=h4, h1=h1, m15=m15)


# ---------------------------------------------------------------------------
# Kill Zone — هل نحن داخل وقت تداول مسموح؟
# ---------------------------------------------------------------------------

def in_kill_zone(dt_utc: datetime) -> bool:
    """True إذا كان الوقت داخل London KZ أو New York KZ."""
    h = dt_utc.hour
    return any(start <= h < end for start, end in KILL_ZONES_UTC)


def kill_zone_name(dt_utc: datetime) -> str:
    """اسم الجلسة الحالية أو 'خارج الجلسات'."""
    h = dt_utc.hour
    if 7 <= h < 10:
        return "London Kill Zone"
    if 13 <= h < 16:
        return "New York Kill Zone"
    return "خارج الجلسات"


# ---------------------------------------------------------------------------
# فلتر الأخبار الاقتصادية — ForexFactory
# ---------------------------------------------------------------------------

_news_cache: dict = {"data": None, "fetched_at": None}


def _fetch_ff_calendar() -> list:
    """جلب تقويم ForexFactory (مخزّن مؤقتاً لمدة ساعة)."""
    now = datetime.now(timezone.utc)
    if _news_cache["data"] and _news_cache["fetched_at"]:
        if now - _news_cache["fetched_at"] < timedelta(hours=1):
            return _news_cache["data"]
    try:
        resp = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10)
        resp.raise_for_status()
        _news_cache["data"] = resp.json()
        _news_cache["fetched_at"] = now
        return _news_cache["data"]
    except Exception as e:
        logger.warning(f"تعذّر جلب التقويم الاقتصادي: {e}")
        return _news_cache["data"] or []


def has_high_impact_news(currencies: list[str]) -> tuple[bool, str]:
    """
    هل يوجد خبر عالي التأثير قريب لأي من العملات المدرجة؟
    Returns: (True/False, وصف الخبر إن وُجد)
    """
    events = _fetch_ff_calendar()
    now = datetime.now(timezone.utc)
    for ev in events:
        try:
            if ev.get("impact") != "High":
                continue
            if ev.get("country") not in currencies:
                continue
            ev_time = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            delta_min = abs((ev_time - now).total_seconds()) / 60
            if delta_min <= NEWS_BUFFER_MIN:
                title   = ev.get("title", "خبر غير مسمّى")
                country = ev.get("country", "")
                return True, f"{country}: {title} (بعد {int(delta_min)} دقيقة)"
        except Exception:
            continue
    return False, ""
