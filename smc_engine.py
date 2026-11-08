"""
smc_engine.py  —  الطبقة الثالثة: محرك Smart Money Concepts
=============================================================
مسؤول عن:
  • CHoCH  — Change of Character (تحوّل هيكل السوق)
  • BOS    — Break of Structure (كسر الهيكل في اتجاه الترند)
  • MSS    — Market Structure Shift (انعكاس هيكلي على M15)
  • Liquidity Sweep (اصطياد السيولة فوق القمم/تحت القيعان)
  • Order Block (OB) — أول شمعة عكسية قبل حركة قوية
  • FVG    — Fair Value Gap (فجوة سعرية)
  • IFVG   — Inverse FVG (فجوة اختُبرت وانعكست → منطقة دعم/مقاومة)
  • CISD   — Change in State of Delivery (ICT: إغلاق شمعة يعكس
              التسليم بعد وصول السعر لـ POI، مؤكداً الانعكاس الحقيقي)
  • HTF POI — مناطق الاهتمام من D1 + H4
  • Next HTF Liquidity — هدف الصفقة (أول مجموعة سيولة معاكسة)

منطق الدخول الكامل:
  1. Sweep of Liquidity  (M15)
  2. HTF POI             (D1 + H4)
  3. IFVG confirmation   (M15)
  4. CISD signal         (M15)
  + CHoCH على H4 كشرط HTF إلزامي
  + MSS على M15 كتأكيد إضافي
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# هياكل البيانات
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    index: int
    price: float
    bar_time: object
    kind: str           # 'high' | 'low'


@dataclass
class OrderBlock:
    high: float
    low: float
    mid: float
    bar_time: object
    direction: str      # 'bullish' | 'bearish'
    broken: bool = False


@dataclass
class FVG:
    top: float
    bottom: float
    mid: float
    bar_time: object
    direction: str      # 'bullish' | 'bearish'
    tested: bool = False    # True = أصبح IFVG


@dataclass
class LiquiditySweep:
    swept_level: float
    sweep_bar: int
    last_close: float
    direction: str          # 'buy' (اصطياد قيعان) | 'sell'
    rejection_body: float   # حجم جسم شمعة الارتداد


@dataclass
class StructureShift:
    kind: str               # 'CHoCH' | 'BOS' | 'MSS'
    direction: str          # 'bullish' | 'bearish'
    broken_level: float
    break_bar: int
    bar_time: object


@dataclass
class SMCAnalysis:
    """نتيجة كاملة لتحليل SMC على فريم واحد."""
    frame: str                              # 'D1' | 'H4' | 'M15'
    direction: str                          # 'bullish' | 'bearish' | 'neutral'
    structure_shifts: list[StructureShift]  = field(default_factory=list)
    order_blocks: list[OrderBlock]          = field(default_factory=list)
    fvgs: list[FVG]                         = field(default_factory=list)
    ifvgs: list[FVG]                        = field(default_factory=list)
    sweep: Optional[LiquiditySweep]        = None
    liquidity_pools: list[float]            = field(default_factory=list)
    next_buy_target: Optional[float]        = None   # أقرب سيولة أعلى
    next_sell_target: Optional[float]       = None   # أقرب سيولة أدنى
    notes: list[str]                        = field(default_factory=list)


# ---------------------------------------------------------------------------
# دوال مساعدة — Swing Points
# ---------------------------------------------------------------------------

def find_swing_highs(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[SwingPoint]:
    """قمم متأرجحة: أعلى من `left` شمعات قبلها و`right` بعدها."""
    highs = []
    n = len(df)
    for i in range(left, n - right):
        h = df["High"].iloc[i]
        if all(h > df["High"].iloc[i - j] for j in range(1, left + 1)) and \
           all(h > df["High"].iloc[i + j] for j in range(1, right + 1)):
            highs.append(SwingPoint(
                index=i, price=h,
                bar_time=df.index[i], kind="high"))
    return highs


def find_swing_lows(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[SwingPoint]:
    """قيعان متأرجحة."""
    lows = []
    n = len(df)
    for i in range(left, n - right):
        l = df["Low"].iloc[i]
        if all(l < df["Low"].iloc[i - j] for j in range(1, left + 1)) and \
           all(l < df["Low"].iloc[i + j] for j in range(1, right + 1)):
            lows.append(SwingPoint(
                index=i, price=l,
                bar_time=df.index[i], kind="low"))
    return lows


# ---------------------------------------------------------------------------
# CHoCH / BOS / MSS
# ---------------------------------------------------------------------------

def detect_structure_shifts(df: pd.DataFrame, frame: str = "",
                              lookback: int = 60) -> list[StructureShift]:
    """
    يكشف عن تغييرات الهيكل في آخر `lookback` شمعة:

    BOS   (Break of Structure):
          كسر آخر HH (للشراء) أو LL (للبيع) — استمرار الاتجاه.

    CHoCH (Change of Character):
          كسر آخر HL (للبيع) أو LH (للشراء) — تحوّل محتمل.

    MSS   (Market Structure Shift — M15 فقط):
          CHoCH بعد Sweep مباشرة = انعكاس مؤكد.
    """
    recent  = df.iloc[-lookback:].copy()
    swings_h = find_swing_highs(recent, left=2, right=2)
    swings_l = find_swing_lows(recent,  left=2, right=2)
    shifts   = []

    last_close = recent["Close"].iloc[-1]
    bar_time   = recent.index[-1]
    last_i     = len(recent) - 1

    if len(swings_h) >= 2:
        prev_hh = swings_h[-2].price
        last_hh = swings_h[-1].price
        if last_close > last_hh:                # BOS صاعد
            shifts.append(StructureShift(
                kind="BOS", direction="bullish",
                broken_level=last_hh, break_bar=last_i, bar_time=bar_time))
        if last_close < swings_h[-1].price and last_hh < prev_hh:   # CHoCH هابط
            shifts.append(StructureShift(
                kind="CHoCH", direction="bearish",
                broken_level=last_hh, break_bar=last_i, bar_time=bar_time))

    if len(swings_l) >= 2:
        prev_ll = swings_l[-2].price
        last_ll = swings_l[-1].price
        if last_close < last_ll:                # BOS هابط
            shifts.append(StructureShift(
                kind="BOS", direction="bearish",
                broken_level=last_ll, break_bar=last_i, bar_time=bar_time))
        if last_close > swings_l[-1].price and last_ll > prev_ll:   # CHoCH صاعد
            shifts.append(StructureShift(
                kind="CHoCH", direction="bullish",
                broken_level=last_ll, break_bar=last_i, bar_time=bar_time))

    return shifts


def has_choch(shifts: list[StructureShift], direction: str) -> bool:
    """هل يوجد CHoCH يؤكد الاتجاه المطلوب؟"""
    return any(s.kind == "CHoCH" and s.direction == direction for s in shifts)


def has_mss(shifts: list[StructureShift], direction: str) -> bool:
    return any(s.kind in ("CHoCH", "MSS") and s.direction == direction for s in shifts)


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------

def detect_liquidity_sweep(df: pd.DataFrame,
                            direction: str,
                            lookback: int = 20) -> Optional[LiquiditySweep]:
    """
    Bullish sweep (buy):  شمعة اخترقت قاعاً سابقاً ثم أغلقت فوقه.
    Bearish sweep (sell): شمعة اخترقت قمة سابقة ثم أغلقت دونها.
    نتحقق من قوة الارتداد عبر حجم جسم شمعة الارتداد.
    """
    if df is None or len(df) < lookback + 2:
        return None

    recent    = df.iloc[-lookback:]
    last      = df.iloc[-1]
    last_close = last["Close"]

    if direction == "buy":
        prev_low  = recent["Low"].iloc[:-1].min()
        if last["Low"] < prev_low and last_close > prev_low:
            body = abs(last_close - last["Open"])
            return LiquiditySweep(
                swept_level=prev_low, sweep_bar=len(df) - 1,
                last_close=last_close, direction="buy",
                rejection_body=body)

    else:  # sell
        prev_high = recent["High"].iloc[:-1].max()
        if last["High"] > prev_high and last_close < prev_high:
            body = abs(last_close - last["Open"])
            return LiquiditySweep(
                swept_level=prev_high, sweep_bar=len(df) - 1,
                last_close=last_close, direction="sell",
                rejection_body=body)

    return None


# ---------------------------------------------------------------------------
# Order Block
# ---------------------------------------------------------------------------

def find_order_blocks(df: pd.DataFrame, direction: str,
                      lookback: int = 15) -> list[OrderBlock]:
    """
    OB صاعد (bullish): آخر شمعة هابطة قبل حركة صاعدة قوية.
    OB هابط (bearish): آخر شمعة صاعدة قبل حركة هابطة قوية.
    نعتبر الحركة "قوية" إذا تجاوزت 1.5× متوسط النطاق الأخير.
    """
    if df is None or len(df) < lookback + 3:
        return []

    recent   = df.iloc[-lookback:]
    avg_range = (recent["High"] - recent["Low"]).mean()
    blocks   = []

    for i in range(1, len(recent) - 2):
        c = recent.iloc[i]
        # الشمعة بعدها
        next_c = recent.iloc[i + 1]
        move   = abs(next_c["Close"] - next_c["Open"])

        if direction == "bullish" and c["Close"] < c["Open"]:   # شمعة هابطة
            if move > 1.5 * avg_range and next_c["Close"] > next_c["Open"]:
                blocks.append(OrderBlock(
                    high=c["High"], low=c["Low"],
                    mid=(c["High"] + c["Low"]) / 2,
                    bar_time=recent.index[i],
                    direction="bullish"))

        elif direction == "bearish" and c["Close"] > c["Open"]: # شمعة صاعدة
            if move > 1.5 * avg_range and next_c["Close"] < next_c["Open"]:
                blocks.append(OrderBlock(
                    high=c["High"], low=c["Low"],
                    mid=(c["High"] + c["Low"]) / 2,
                    bar_time=recent.index[i],
                    direction="bearish"))

    # إزالة الـ OB المكسورة (كسرها السعر لاحقاً)
    last_close = df["Close"].iloc[-1]
    for ob in blocks:
        if direction == "bullish" and last_close < ob.low:
            ob.broken = True
        elif direction == "bearish" and last_close > ob.high:
            ob.broken = True

    return [ob for ob in blocks if not ob.broken]


# ---------------------------------------------------------------------------
# FVG و IFVG
# ---------------------------------------------------------------------------

def find_fvgs(df: pd.DataFrame, direction: str,
              lookback: int = 20) -> tuple[list[FVG], list[FVG]]:
    """
    FVG: فجوة بين شمعة[i-2] وشمعة[i] لا تغطيها شمعة[i-1].
      Bullish FVG: Low[i] > High[i-2]
      Bearish FVG: High[i] < Low[i-2]

    IFVG (Inverse FVG):
      FVG اختُبرت (دخل السعر نطاقها) ثم ارتدّ → أصبحت منطقة دعم/مقاومة قوية.
    """
    if df is None or len(df) < lookback + 3:
        return [], []

    recent  = df.iloc[-lookback:]
    fvgs    = []
    ifvgs   = []
    last_close = df["Close"].iloc[-1]

    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c3 = recent.iloc[i]

        if direction == "bullish":
            if c3["Low"] > c1["High"]:
                gap = FVG(
                    top=c3["Low"], bottom=c1["High"],
                    mid=(c3["Low"] + c1["High"]) / 2,
                    bar_time=recent.index[i],
                    direction="bullish")
                # هل اختُبرت؟ (دخل السعر النطاق بعدها)
                future = recent.iloc[i:]
                tested = any(
                    row["Low"] <= gap.top and row["High"] >= gap.bottom
                    for _, row in future.iterrows())
                if tested:
                    gap.tested = True
                    ifvgs.append(gap)
                else:
                    fvgs.append(gap)

        elif direction == "bearish":
            if c3["High"] < c1["Low"]:
                gap = FVG(
                    top=c1["Low"], bottom=c3["High"],
                    mid=(c1["Low"] + c3["High"]) / 2,
                    bar_time=recent.index[i],
                    direction="bearish")
                future = recent.iloc[i:]
                tested = any(
                    row["Low"] <= gap.top and row["High"] >= gap.bottom
                    for _, row in future.iterrows())
                if tested:
                    gap.tested = True
                    ifvgs.append(gap)
                else:
                    fvgs.append(gap)

    return fvgs, ifvgs


def price_at_ifvg(price: float, ifvgs: list[FVG], direction: str) -> Optional[FVG]:
    """هل السعر الحالي داخل أو قريب من IFVG مناسبة للاتجاه؟"""
    for ig in ifvgs:
        if ig.direction == direction:
            if ig.bottom <= price <= ig.top:
                return ig
            # قريب جداً (ضمن 20% من حجم الفجوة)
            gap_size = ig.top - ig.bottom
            if abs(price - ig.mid) < gap_size * 0.2:
                return ig
    return None


# ---------------------------------------------------------------------------
# CISD — Change in State of Delivery (ICT)
# ---------------------------------------------------------------------------

def detect_cisd(df: pd.DataFrame, direction: str,
                poi_high: float, poi_low: float,
                lookback: int = 10) -> bool:
    """
    CISD (ICT دقيق):
    بعد وصول السعر لمنطقة POI (OB أو IFVG)، ننتظر:
      • شمعة تدخل النطاق
      • ثم شمعة تُغلق بعيداً عنه في اتجاه الصفقة (تأكيد التسليم)

    للشراء:  السعر يلمس POI من الأعلى/الوسط ثم شمعة تُغلق فوق منتصف POI
    للبيع:   السعر يلمس POI من الأسفل/الوسط ثم شمعة تُغلق تحت منتصف POI
    """
    if df is None or len(df) < lookback + 2:
        return False

    recent  = df.iloc[-lookback:]
    poi_mid = (poi_high + poi_low) / 2

    for i in range(len(recent) - 1):
        c = recent.iloc[i]
        # هل لمست الشمعة الـ POI؟
        touched = c["Low"] <= poi_high and c["High"] >= poi_low

        if not touched:
            continue

        # هل الشمعة التالية تُغلق في اتجاه الصفقة؟
        next_c = recent.iloc[i + 1]
        if direction == "buy" and next_c["Close"] > poi_mid:
            return True
        if direction == "sell" and next_c["Close"] < poi_mid:
            return True

    return False


# ---------------------------------------------------------------------------
# Liquidity Pools (أهداف السعر)
# ---------------------------------------------------------------------------

def find_liquidity_pools(df: pd.DataFrame,
                          direction: str,
                          lookback: int = 50) -> list[float]:
    """
    مجموعات السيولة الموجودة أمام السعر:
    • Equal Highs / Equal Lows (فرق < 0.05%)
    • Swing Highs / Lows البارزة

    للشراء: نبحث عن مجموعات فوق السعر الحالي
    للبيع:  نبحث عن مجموعات تحت السعر الحالي
    """
    if df is None or len(df) < lookback:
        return []

    recent     = df.iloc[-lookback:]
    last_close = df["Close"].iloc[-1]
    pools      = []
    tolerance  = 0.0005   # 0.05%

    swings = find_swing_highs(recent) if direction == "buy" else find_swing_lows(recent)

    for sw in swings:
        if direction == "buy"  and sw.price > last_close:
            pools.append(sw.price)
        elif direction == "sell" and sw.price < last_close:
            pools.append(sw.price)

    # Equal Highs/Lows
    highs = recent["High"].values
    lows  = recent["Low"].values
    prices_to_check = highs if direction == "buy" else lows

    for i in range(len(prices_to_check)):
        for j in range(i + 1, len(prices_to_check)):
            if abs(prices_to_check[i] - prices_to_check[j]) / max(prices_to_check[i], 1e-9) < tolerance:
                level = (prices_to_check[i] + prices_to_check[j]) / 2
                if direction == "buy"  and level > last_close:
                    pools.append(level)
                elif direction == "sell" and level < last_close:
                    pools.append(level)

    return sorted(set(round(p, 6) for p in pools),
                  reverse=(direction == "sell"))


def next_htf_target(df_h4: pd.DataFrame, df_d1: pd.DataFrame,
                     direction: str, entry: float) -> Optional[float]:
    """
    أقرب هدف سيولة HTF (H4 أولاً، ثم D1 إن لم يوجد).
    هاد هو TP الحقيقي المبني على بنية السوق، وليس 2R/3R.
    """
    for df in [df_h4, df_d1]:
        if df is None:
            continue
        pools = find_liquidity_pools(df, direction)
        if direction == "buy":
            candidates = [p for p in pools if p > entry]
            if candidates:
                return min(candidates)
        else:
            candidates = [p for p in pools if p < entry]
            if candidates:
                return max(candidates)
    return None


# ---------------------------------------------------------------------------
# HTF POI — مناطق الاهتمام من D1 + H4
# ---------------------------------------------------------------------------

@dataclass
class HTF_POI:
    source: str             # 'D1' | 'H4'
    kind: str               # 'OB' | 'IFVG'
    high: float
    low: float
    mid: float
    direction: str


def find_htf_pois(df_d1: pd.DataFrame, df_h4: pd.DataFrame,
                   direction: str) -> list[HTF_POI]:
    """
    جمع مناطق الاهتمام من D1 و H4:
    • Order Blocks
    • IFVGs (Inverse FVGs)
    فقط المناطق التي السعر الحالي قريب منها أو داخلها.
    """
    pois = []

    for df, label in [(df_d1, "D1"), (df_h4, "H4")]:
        if df is None or len(df) < 20:
            continue

        last_close = df["Close"].iloc[-1]
        atr_val    = (df["High"] - df["Low"]).mean()   # ATR تقريبي

        # OBs
        obs = find_order_blocks(df, direction)
        for ob in obs:
            dist = abs(last_close - ob.mid)
            if dist <= atr_val * 5:    # قريب ضمن 5 × ATR
                pois.append(HTF_POI(
                    source=label, kind="OB",
                    high=ob.high, low=ob.low, mid=ob.mid,
                    direction=direction))

        # IFVGs
        _, ifvgs = find_fvgs(df, direction)
        for ig in ifvgs:
            dist = abs(last_close - ig.mid)
            if dist <= atr_val * 5:
                pois.append(HTF_POI(
                    source=label, kind="IFVG",
                    high=ig.top, low=ig.bottom, mid=ig.mid,
                    direction=direction))

    return pois


def price_near_poi(price: float, poi: HTF_POI, tolerance_pct: float = 0.3) -> bool:
    """هل السعر داخل أو قريب جداً من الـ POI؟"""
    span = poi.high - poi.low
    margin = span * tolerance_pct
    return (poi.low - margin) <= price <= (poi.high + margin)


# ---------------------------------------------------------------------------
# التحليل الكامل لفريم واحد
# ---------------------------------------------------------------------------

def analyze_frame(df: pd.DataFrame, frame: str,
                   direction: str) -> SMCAnalysis:
    """تحليل SMC كامل لفريم واحد وإرجاع نتيجة موحّدة."""
    result = SMCAnalysis(frame=frame, direction=direction)

    if df is None or len(df) < 30:
        result.notes.append(f"[{frame}] بيانات غير كافية")
        return result

    # الهيكل
    result.structure_shifts = detect_structure_shifts(df, frame)
    shift_kinds = [f"{s.kind}({s.direction})" for s in result.structure_shifts]
    if shift_kinds:
        result.notes.append(f"هيكل: {', '.join(shift_kinds)}")

    # Order Blocks
    result.order_blocks = find_order_blocks(df, direction)
    if result.order_blocks:
        result.notes.append(f"{len(result.order_blocks)} OB نشط")

    # FVG و IFVG
    result.fvgs, result.ifvgs = find_fvgs(df, direction)
    if result.fvgs:
        result.notes.append(f"{len(result.fvgs)} FVG")
    if result.ifvgs:
        result.notes.append(f"{len(result.ifvgs)} IFVG (مناطق انعكاس)")

    # Sweep
    result.sweep = detect_liquidity_sweep(df, direction)
    if result.sweep:
        result.notes.append(
            f"Sweep @ {result.sweep.swept_level:.5f} ← إغلاق {result.sweep.last_close:.5f}")

    # Liquidity Pools
    result.liquidity_pools = find_liquidity_pools(df, direction)

    return result
