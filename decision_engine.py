"""
decision_engine.py  —  الطبقة الرابعة: محرك قرار التداول
==========================================================
يجمع مخرجات الطبقات 1-3 ويتخذ القرار النهائي:

شروط الدخول الإلزامية (يجب توفر الكل):
  1. Sweep of Liquidity    — M15
  2. HTF POI               — D1 أو H4 (OB أو IFVG)
  3. IFVG confirmation     — M15
  4. CISD                  — M15
  + CHoCH على H4           — شرط HTF إلزامي
  + Kill Zone              — London أو New York فقط
  + لا أخبار عالية التأثير

شروط التأكيد الإضافية (ترفع درجة الثقة، لا تمنع الصفقة):
  • MSS على M15
  • OB على M15
  • ADX > 20
  • RSI غير متشبع
  • EMA20 متوافق

TP: أقرب سيولة HTF معاكسة (H4 ثم D1)
SL: تحت/فوق swept_level + 0.5 × ATR14
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import analysis_engine as ae
import market_data as md
import smc_engine as smc

logger = logging.getLogger(__name__)

HIGH_CORR_THRESHOLD = 0.75   # عتبة الارتباط للتحذير


# ---------------------------------------------------------------------------
# هيكل قرار الصفقة
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    asset: str
    direction: str              # 'buy' | 'sell'
    entry: float
    sl: float
    tp: float                   # HTF Liquidity target
    tp_fallback: float          # 2R fallback إن لم يوجد HTF target
    risk: float
    rr: float                   # R:R الفعلي
    position_size: float
    confluence_score: int       # عدد شروط التأكيد الإضافية (0-5)
    mandatory_checks: dict      # الشروط الإلزامية وحالتها
    confluence_notes: list[str]
    htf_poi: Optional[smc.HTF_POI]
    sweep: Optional[smc.LiquiditySweep]
    kill_zone: str
    correlation_warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """تقرير كامل لأصل واحد — سواء كان إشارة أو مراقبة أو تجاوز."""
    asset: str
    status: str                 # 'signal' | 'watch' | 'skip'
    signal: Optional[TradeSignal]
    d1_trend: Optional[ae.FrameTrend]
    h4_trend: Optional[ae.FrameTrend]
    m15_trend: Optional[ae.FrameTrend]
    d1_smc: Optional[smc.SMCAnalysis]
    h4_smc: Optional[smc.SMCAnalysis]
    m15_smc: Optional[smc.SMCAnalysis]
    skip_reason: str = ""
    watch_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# تحليل أصل واحد كامل
# ---------------------------------------------------------------------------

def analyze_asset(asset_data: md.AssetData,
                   balance: float = md.ACCOUNT_BALANCE_USD,
                   risk_pct: float = md.RISK_PERCENT) -> AnalysisReport:
    """
    التحليل الكامل بالتسلسل:
    D1 → H4 → M15 → قرار الدخول
    """
    name = asset_data.name
    cfg  = md.ASSETS[name]

    # ---- شروط مسبقة ----
    now_utc = pd.Timestamp.now(tz="UTC")

    if not md.in_kill_zone(now_utc):
        return AnalysisReport(
            asset=name, status="skip",
            signal=None, d1_trend=None, h4_trend=None, m15_trend=None,
            d1_smc=None, h4_smc=None, m15_smc=None,
            skip_reason=f"خارج Kill Zones (الوقت الحالي: {md.kill_zone_name(now_utc)})")

    has_news, news_desc = md.has_high_impact_news(cfg["currencies"])
    if has_news:
        return AnalysisReport(
            asset=name, status="skip",
            signal=None, d1_trend=None, h4_trend=None, m15_trend=None,
            d1_smc=None, h4_smc=None, m15_smc=None,
            skip_reason=f"خبر عالي التأثير: {news_desc}")

    if not asset_data.is_valid:
        return AnalysisReport(
            asset=name, status="skip",
            signal=None, d1_trend=None, h4_trend=None, m15_trend=None,
            d1_smc=None, h4_smc=None, m15_smc=None,
            skip_reason="بيانات غير كافية")

    # ---- تحليل D1 ----
    d1_trend = ae.analyze_trend(asset_data.d1, "D1")
    if d1_trend is None or d1_trend.direction == "neutral":
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=None, m15_trend=None,
            d1_smc=None, h4_smc=None, m15_smc=None,
            watch_notes=["D1: اتجاه غير واضح — انتظار"])

    direction_str = d1_trend.direction    # 'bullish' | 'bearish'
    trade_dir     = "buy" if direction_str == "bullish" else "sell"

    d1_smc = smc.analyze_frame(asset_data.d1, "D1", direction_str)

    # ---- تحليل H4 ----
    h4_trend = ae.analyze_trend(asset_data.h4, "H4")
    h4_smc   = smc.analyze_frame(asset_data.h4, "H4", direction_str)

    # H4 يجب أن يكون في نفس اتجاه D1 أو neutral
    if h4_trend and h4_trend.direction not in (direction_str, "neutral"):
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=None,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=None,
            watch_notes=[
                f"D1: {direction_str} ↔ H4: {h4_trend.direction} — تعارض الفريمات، انتظار توافق"])

    # CHoCH على H4 (شرط إلزامي)
    h4_choch = smc.has_choch(h4_smc.structure_shifts, direction_str)
    if not h4_choch:
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=None,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=None,
            watch_notes=["H4: لا يوجد CHoCH يؤكد تحوّل الهيكل بعد"])

    # ---- HTF POI ----
    htf_pois = smc.find_htf_pois(asset_data.d1, asset_data.h4, direction_str)
    if not htf_pois:
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=None,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=None,
            watch_notes=["لا توجد مناطق POI (OB/IFVG) قريبة على D1/H4"])

    # ---- تحليل M15 ----
    m15_trend = ae.analyze_trend(asset_data.m15, "M15")
    m15_smc   = smc.analyze_frame(asset_data.m15, "M15", direction_str)

    # 1. Sweep على M15 (إلزامي)
    sweep = smc.detect_liquidity_sweep(asset_data.m15, trade_dir)
    if sweep is None:
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
            watch_notes=[
                f"M15: لا يوجد Liquidity Sweep بعد في اتجاه {trade_dir}"])

    last_close = asset_data.m15["Close"].iloc[-1]

    # 2. السعر قريب من HTF POI (إلزامي)
    active_poi = None
    for poi in htf_pois:
        if smc.price_near_poi(last_close, poi):
            active_poi = poi
            break

    if active_poi is None:
        dist_to_pois = [abs(last_close - p.mid) for p in htf_pois]
        nearest = min(dist_to_pois)
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
            watch_notes=[
                f"السعر بعيد عن HTF POI (أقرب: {nearest:.5f} نقطة)"])

    # 3. IFVG على M15 (إلزامي)
    _, m15_ifvgs = smc.find_fvgs(asset_data.m15, direction_str)
    ifvg_hit = smc.price_at_ifvg(last_close, m15_ifvgs, direction_str)
    if ifvg_hit is None:
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
            watch_notes=["M15: لا توجد IFVG عند السعر الحالي، انتظار"])

    # 4. CISD على M15 (إلزامي)
    cisd_ok = smc.detect_cisd(
        asset_data.m15, trade_dir,
        poi_high=active_poi.high, poi_low=active_poi.low)
    if not cisd_ok:
        return AnalysisReport(
            asset=name, status="watch",
            signal=None,
            d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
            watch_notes=["M15: CISD لم يتأكد بعد (انتظار إغلاق شمعة تأكيد)"])

    # ---- كل الشروط الإلزامية متوفرة — حساب SL/TP ----
    atr_m15 = ae.atr(asset_data.m15, 14).iloc[-1]
    if pd.isna(atr_m15) or atr_m15 == 0:
        return AnalysisReport(
            asset=name, status="skip",
            signal=None, d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
            skip_reason="ATR = 0، بيانات غير صالحة")

    entry = round(last_close, 6)

    if trade_dir == "buy":
        sl   = round(sweep.swept_level - 0.5 * atr_m15, 6)
        risk = entry - sl
    else:
        sl   = round(sweep.swept_level + 0.5 * atr_m15, 6)
        risk = sl - entry

    if risk <= 0:
        return AnalysisReport(
            asset=name, status="skip",
            signal=None, d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
            d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
            skip_reason="المخاطرة سالبة أو صفر")

    # TP على HTF Liquidity
    tp_htf = smc.next_htf_target(asset_data.h4, asset_data.d1, trade_dir, entry)
    tp_fallback = round(entry + risk * 2 if trade_dir == "buy" else entry - risk * 2, 6)
    tp = round(tp_htf, 6) if tp_htf else tp_fallback

    # R:R
    if trade_dir == "buy":
        rr = (tp - entry) / risk if risk > 0 else 0
    else:
        rr = (entry - tp) / risk if risk > 0 else 0

    # حجم المركز
    pos_size = ae.position_size(entry, sl, balance, risk_pct)

    # ---- شروط التأكيد الإضافية (Confluence) ----
    confluence_notes = [
        "✅ Sweep of Liquidity (M15)",
        f"✅ HTF POI ({active_poi.source} {active_poi.kind})",
        "✅ IFVG تأكيد (M15)",
        "✅ CISD تأكيد (M15)",
        "✅ CHoCH (H4)",
    ]
    score = 0

    # MSS على M15
    if smc.has_mss(m15_smc.structure_shifts, direction_str):
        confluence_notes.append("✅ MSS (M15) — انعكاس هيكلي مؤكد")
        score += 1

    # OB على M15
    m15_obs = smc.find_order_blocks(asset_data.m15, direction_str)
    if m15_obs:
        confluence_notes.append(f"✅ Order Block (M15) @ {m15_obs[-1].mid:.5f}")
        score += 1

    # ADX
    if m15_trend and m15_trend.adx_val >= 20:
        confluence_notes.append(f"✅ ADX = {m15_trend.adx_val:.1f} (اتجاه قوي)")
        score += 1

    # RSI
    if m15_trend:
        rsi_v = m15_trend.rsi_val
        if trade_dir == "buy"  and 30 < rsi_v < 65:
            confluence_notes.append(f"✅ RSI = {rsi_v:.1f} (غير متشبع شراءً)")
            score += 1
        elif trade_dir == "sell" and 35 < rsi_v < 70:
            confluence_notes.append(f"✅ RSI = {rsi_v:.1f} (غير متشبع بيعاً)")
            score += 1

    # EMA20
    if m15_trend:
        e20 = m15_trend.ema20
        if trade_dir == "buy"  and last_close > e20:
            confluence_notes.append(f"✅ EMA20 = {e20:.5f} متوافق (زخم صاعد)")
            score += 1
        elif trade_dir == "sell" and last_close < e20:
            confluence_notes.append(f"✅ EMA20 = {e20:.5f} متوافق (زخم هابط)")
            score += 1

    # تحذير إن كان RR منخفضاً
    if rr < 1.5:
        confluence_notes.append(f"⚠️ RR = {rr:.2f} (منخفض — TP قريب جداً من السعر)")

    kill_zone_lbl = md.kill_zone_name(now_utc)

    mandatory = {
        "Sweep (M15)":    True,
        "HTF POI":        True,
        "IFVG (M15)":     True,
        "CISD (M15)":     True,
        "CHoCH (H4)":     True,
        "Kill Zone":      True,
        "No News":        True,
    }

    signal = TradeSignal(
        asset=name, direction=trade_dir,
        entry=entry, sl=sl,
        tp=tp, tp_fallback=tp_fallback,
        risk=round(risk, 6), rr=round(rr, 2),
        position_size=pos_size,
        confluence_score=score,
        mandatory_checks=mandatory,
        confluence_notes=confluence_notes,
        htf_poi=active_poi,
        sweep=sweep,
        kill_zone=kill_zone_lbl,
    )

    return AnalysisReport(
        asset=name, status="signal",
        signal=signal,
        d1_trend=d1_trend, h4_trend=h4_trend, m15_trend=m15_trend,
        d1_smc=d1_smc, h4_smc=h4_smc, m15_smc=m15_smc,
    )


# ---------------------------------------------------------------------------
# تحذيرات الارتباط (بعد تحليل كل الأصول)
# ---------------------------------------------------------------------------

def apply_correlation_warnings(reports: list[AnalysisReport],
                                 all_data: dict[str, md.AssetData]) -> None:
    """
    مقارنة الإشارات المتزامنة:
    إذا زوجان بنفس الاتجاه وارتباطهما >= HIGH_CORR_THRESHOLD،
    يُضاف تحذير لكليهما (لا يُلغى الإشارة).
    """
    signals = [(r.asset, r.signal) for r in reports
               if r.status == "signal" and r.signal is not None]

    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            name_a, sig_a = signals[i]
            name_b, sig_b = signals[j]

            if sig_a.direction != sig_b.direction:
                continue

            data_a = all_data.get(name_a)
            data_b = all_data.get(name_b)
            if data_a is None or data_b is None:
                continue
            if data_a.m15 is None or data_b.m15 is None:
                continue

            corr = ae.rolling_correlation(
                data_a.m15["Close"], data_b.m15["Close"], window=50)

            if abs(corr) >= HIGH_CORR_THRESHOLD:
                warn = (f"⚠️ ارتباط عالٍ مع {name_b} = {corr:.2f} "
                        f"— مخاطرة مزدوجة على نفس الاتجاه")
                sig_a.correlation_warnings.append(warn)
                sig_b.correlation_warnings.append(
                    warn.replace(name_b, name_a))
