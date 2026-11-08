"""
analysis_engine.py  —  الطبقة الثانية: محرك التحليل الكمي
===========================================================
مسؤول عن:
  • EMA 20 / 50 / 200
  • RSI 14
  • ATR 14
  • ADX 14 (قوة الاتجاه)
  • Volatility (نسبة ATR/Close)
  • Rolling Correlation بين أصلين
  • اتجاه الفريم بناءً على EMA50/200 (Bullish / Bearish / Neutral)
  • حساب Premium/Discount Zones (50% Fibonacci)
  • حجم المركز المقترح

لا يحتوي على أي منطق SMC — فقط أرقام ومؤشرات.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# المؤشرات الأساسية
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta     = series.diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss  = loss.ewm(com=period - 1, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index — يقيس قوة الاتجاه (وليس اتجاهه).
    > 25: اتجاه قوي | 20-25: اتجاه معتدل | < 20: سوق جانبي
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_series    = atr(df, period)
    plus_di      = pd.Series(plus_dm,  index=df.index).ewm(com=period - 1, adjust=False).mean() / tr_series * 100
    minus_di     = pd.Series(minus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean() / tr_series * 100

    dx           = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_series   = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_series


def volatility_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """نسبة ATR إلى السعر — مقياس موحّد للتقلب عبر الأصول."""
    atr_series = atr(df, period)
    return (atr_series / df["Close"] * 100).round(4)


def rolling_correlation(closes_a: pd.Series, closes_b: pd.Series,
                        window: int = 50) -> float:
    """
    Pearson Rolling Correlation على عائدات الأصلين.
    يُستخدم لتحذير المتداول من مضاعفة المخاطرة على أزواج مرتبطة.
    """
    try:
        ret_a   = closes_a.pct_change().dropna()
        ret_b   = closes_b.pct_change().dropna()
        aligned = pd.concat([ret_a, ret_b], axis=1, join="inner").tail(window)
        if len(aligned) < 10:
            return 0.0
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return 0.0 if pd.isna(corr) else round(float(corr), 3)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# تحديد اتجاه الفريم
# ---------------------------------------------------------------------------

class FrameTrend:
    """نتيجة تحليل اتجاه فريم واحد."""
    __slots__ = ("direction", "ema20", "ema50", "ema200",
                 "adx_val", "rsi_val", "atr_val", "volatility",
                 "is_trending", "notes")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return (f"FrameTrend({self.direction} | "
                f"EMA50={'↑' if self.ema50 else '↓'} | "
                f"ADX={self.adx_val:.1f} | RSI={self.rsi_val:.1f})")


def analyze_trend(df: pd.DataFrame, label: str = "") -> Optional[FrameTrend]:
    """
    تحليل اتجاه فريم واحد:
      Bullish  = EMA50 > EMA200 AND Close > EMA50
      Bearish  = EMA50 < EMA200 AND Close < EMA50
      Neutral  = لا يتحقق أي من الشرطين

    يتطلب على الأقل 205 شمعة لاستخدام EMA200؛
    وإن أقل، يستخدم EMA50 فقط.
    """
    if df is None or len(df) < 55:
        logger.debug(f"[{label}] بيانات غير كافية لتحليل الاتجاه")
        return None

    close  = df["Close"]
    e20    = ema(close, 20)
    e50    = ema(close, 50)
    e200   = ema(close, 200) if len(df) >= 205 else None

    atr_v  = atr(df, 14).iloc[-1]
    rsi_v  = rsi(close, 14).iloc[-1]
    adx_v  = adx(df, 14).iloc[-1]
    vol_v  = volatility_pct(df, 14).iloc[-1]

    last_close = close.iloc[-1]
    e50_last   = e50.iloc[-1]
    e200_last  = e200.iloc[-1] if e200 is not None else None

    notes = []

    if e200_last is not None:
        if e50_last > e200_last and last_close > e50_last:
            direction = "bullish"
            notes.append(f"EMA50 ({e50_last:.5f}) > EMA200 ({e200_last:.5f})")
            notes.append(f"Close ({last_close:.5f}) > EMA50")
        elif e50_last < e200_last and last_close < e50_last:
            direction = "bearish"
            notes.append(f"EMA50 ({e50_last:.5f}) < EMA200 ({e200_last:.5f})")
            notes.append(f"Close ({last_close:.5f}) < EMA50")
        else:
            direction = "neutral"
            notes.append("EMA50/EMA200 غير متوافقين مع السعر")
    else:
        # أقل من 205 شمعة — EMA50 فقط
        if last_close > e50_last:
            direction = "bullish"
            notes.append(f"Close > EMA50 ({e50_last:.5f}) [بدون EMA200]")
        elif last_close < e50_last:
            direction = "bearish"
            notes.append(f"Close < EMA50 ({e50_last:.5f}) [بدون EMA200]")
        else:
            direction = "neutral"

    # EMA20 (زخم قصير المدى)
    e20_last = e20.iloc[-1]
    if direction == "bullish" and last_close > e20_last:
        notes.append(f"EMA20 ({e20_last:.5f}) متوافق — زخم إضافي")
    elif direction == "bearish" and last_close < e20_last:
        notes.append(f"EMA20 ({e20_last:.5f}) متوافق — زخم إضافي")

    is_trending = not pd.isna(adx_v) and adx_v >= 20

    return FrameTrend(
        direction   = direction,
        ema20       = float(e20_last),
        ema50       = float(e50_last),
        ema200      = float(e200_last) if e200_last is not None else None,
        adx_val     = float(adx_v)  if not pd.isna(adx_v)  else 0.0,
        rsi_val     = float(rsi_v)  if not pd.isna(rsi_v)  else 50.0,
        atr_val     = float(atr_v)  if not pd.isna(atr_v)  else 0.0,
        volatility  = float(vol_v)  if not pd.isna(vol_v)  else 0.0,
        is_trending = is_trending,
        notes       = notes,
    )


# ---------------------------------------------------------------------------
# Premium / Discount Zones
# ---------------------------------------------------------------------------

def premium_discount(ob_high: float, ob_low: float) -> dict:
    """
    نصف الـ OB الأعلى = Premium (للبيع فقط)
    نصف الـ OB الأدنى = Discount (للشراء فقط)
    المتداول المحترف يدخل دائماً من القسم الصحيح.
    """
    mid = (ob_high + ob_low) / 2
    return {
        "mid":      round(mid, 6),
        "premium":  (mid, ob_high),     # للبيع
        "discount": (ob_low, mid),      # للشراء
    }


def is_in_discount(price: float, ob_high: float, ob_low: float) -> bool:
    """هل السعر في القسم المخفّض من الـ OB؟ (مناسب للشراء)"""
    zones = premium_discount(ob_high, ob_low)
    lo, hi = zones["discount"]
    return lo <= price <= hi


def is_in_premium(price: float, ob_high: float, ob_low: float) -> bool:
    """هل السعر في القسم المرتفع من الـ OB؟ (مناسب للبيع)"""
    zones = premium_discount(ob_high, ob_low)
    lo, hi = zones["premium"]
    return lo <= price <= hi


# ---------------------------------------------------------------------------
# حجم المركز
# ---------------------------------------------------------------------------

def position_size(entry: float, sl: float,
                  balance: float, risk_pct: float) -> float:
    """
    حجم المركز بناءً على المخاطرة الثابتة.
    risk_amount = balance * (risk_pct / 100)
    size = risk_amount / |entry - sl|
    """
    per_unit = abs(entry - sl)
    if per_unit == 0:
        return 0.0
    risk_amount = balance * (risk_pct / 100)
    return round(risk_amount / per_unit, 4)
