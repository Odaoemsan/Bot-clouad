"""
forex_bot.py  —  الطبقة الخامسة: بوت تلغرام
=============================================
يستخدم الطبقات 1-4 ويرسل الإشارات والتقارير التفصيلية.

تشغيل:
    python forex_bot.py

أوامر:
    /start   — مرحبا + شرح الاستراتيجية
    /scan    — فحص كل الأصول الآن
    /report  — تقرير تفصيلي لكل فريم (D1/H4/M15) لكل أصل
    /status  — حالة النظام والإعدادات
"""

import logging
import os
from datetime import timezone

import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import analysis_engine as ae
import decision_engine as de
import market_data as md

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN",
                            "9092777792:AAFzPCoBKK3MSF-hGxI8gJHfl8tDD2qxOVQ")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# تنسيق الرسائل
# ---------------------------------------------------------------------------

def _fmt_signal(rep: de.AnalysisReport) -> str:
    s = rep.signal
    arrow  = "🟢 شراء" if s.direction == "buy" else "🔴 بيع"
    tp_src = "HTF Liquidity" if s.tp != s.tp_fallback else "2R (fallback)"

    lines = [
        f"🚨 *إشارة — {s.asset}*",
        f"الاتجاه: {arrow}  |  الجلسة: {s.kill_zone}",
        "",
        f"📍 الدخول: `{s.entry}`",
        f"🛑 وقف الخسارة: `{s.sl}`  (Swept: {s.sweep.swept_level:.5f})",
        f"🎯 الهدف: `{s.tp}` ({tp_src})",
        f"📊 R:R = {s.rr:.2f}",
        f"📐 حجم المركز: {s.position_size} وحدة",
        "",
        f"*POI*: {s.htf_poi.source} {s.htf_poi.kind}"
        f" [{s.htf_poi.low:.5f} – {s.htf_poi.high:.5f}]",
        "",
        "*شروط الدخول الإلزامية:*",
    ]
    for k, v in s.mandatory_checks.items():
        lines.append(f"  {'✅' if v else '❌'} {k}")

    lines.append(f"\n*توافق إضافي ({s.confluence_score}/5):*")
    for note in s.confluence_notes:
        lines.append(f"  {note}")

    if s.correlation_warnings:
        lines.append("")
        for w in s.correlation_warnings:
            lines.append(w)

    lines.append("\n⚠️ _إشارة فنية فقط — ليست توصية استثمارية._")
    return "\n".join(lines)


def _fmt_watch(rep: de.AnalysisReport) -> str:
    notes = " | ".join(rep.watch_notes) if rep.watch_notes else "—"
    d1 = rep.d1_trend.direction if rep.d1_trend else "—"
    h4 = rep.h4_trend.direction if rep.h4_trend else "—"
    return f"👀 *{rep.asset}*: D1={d1} | H4={h4}\n   ↳ {notes}"


def _fmt_skip(rep: de.AnalysisReport) -> str:
    return f"⏭️ *{rep.asset}*: {rep.skip_reason}"


def _fmt_frame_report(rep: de.AnalysisReport) -> str:
    """تقرير تفصيلي بالفريمات الثلاثة."""
    lines = [f"📋 *تقرير {rep.asset}* (D1 → H4 → M15)"]

    for label, trend, smca in [
        ("D1",  rep.d1_trend,  rep.d1_smc),
        ("H4",  rep.h4_trend,  rep.h4_smc),
        ("M15", rep.m15_trend, rep.m15_smc),
    ]:
        lines.append(f"\n*── {label} ──*")
        if trend:
            dir_ar = {"bullish": "صاعد", "bearish": "هابط", "neutral": "جانبي"}.get(
                trend.direction, trend.direction)
            trending = "✅ اتجاه قوي" if trend.is_trending else "⚠️ سوق جانبي"
            lines.append(f"الاتجاه: {dir_ar}  {trending}")
            lines.append(f"EMA20={trend.ema20:.5f} | EMA50={trend.ema50:.5f}"
                          + (f" | EMA200={trend.ema200:.5f}" if trend.ema200 else ""))
            lines.append(f"RSI={trend.rsi_val:.1f} | ADX={trend.adx_val:.1f} | ATR={trend.atr_val:.5f}")
            for n in trend.notes:
                lines.append(f"  · {n}")
        else:
            lines.append("لا توجد بيانات كافية")

        if smca:
            if smca.structure_shifts:
                kinds = [f"{s.kind}({s.direction[:4]})" for s in smca.structure_shifts]
                lines.append(f"هيكل: {', '.join(kinds)}")
            if smca.order_blocks:
                lines.append(f"OB نشطة: {len(smca.order_blocks)}")
            if smca.ifvgs:
                lines.append(f"IFVG: {len(smca.ifvgs)}")
            if smca.sweep:
                lines.append(f"Sweep @ {smca.sweep.swept_level:.5f}")
            if smca.liquidity_pools:
                top3 = smca.liquidity_pools[:3]
                lines.append(f"أقرب سيولة: {top3}")

    if rep.status == "signal" and rep.signal:
        lines.append(f"\n*القرار: إشارة {rep.signal.direction} — TP={rep.signal.tp}*")
    elif rep.status == "watch":
        lines.append(f"\n*القرار: مراقبة — {' | '.join(rep.watch_notes)}*")
    else:
        lines.append(f"\n*القرار: تجاوز — {rep.skip_reason}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# أوامر البوت
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *بوت إشارات الفوركس والذهب — v2*\n\n"
        "*الاستراتيجية (5 طبقات)*:\n"
        "1. Market Data — OHLCV + أخبار + جلسات\n"
        "2. Analysis Engine — EMA20/50/200، RSI، ATR، ADX\n"
        "3. SMC Engine — CHoCH، BOS، Sweep، OB، FVG، IFVG، CISD\n"
        "4. Decision Engine — قرار الدخول + SL/TP\n"
        "5. هذا البوت — الإخراج والتقارير\n\n"
        "*شروط الدخول الإلزامية:*\n"
        "1. Sweep of Liquidity (M15)\n"
        "2. HTF POI — D1 أو H4 (OB/IFVG)\n"
        "3. IFVG تأكيد (M15)\n"
        "4. CISD تأكيد (M15)\n"
        "5. CHoCH على H4\n"
        "6. Kill Zone (London أو New York)\n"
        "7. لا أخبار عالية التأثير\n\n"
        "🎯 TP: أقرب سيولة HTF معاكسة (ليس 2R ثابت)\n"
        "🛑 SL: تحت/فوق Swept Level + 0.5 × ATR\n\n"
        f"*الأصول*: {', '.join(md.ASSETS.keys())}\n\n"
        "/scan — فحص السوق الآن\n"
        "/report — تقرير تفصيلي D1/H4/M15\n"
        "/status — حالة النظام\n\n"
        "⚠️ _إشارات فنية فقط، ليست توصية استثمارية._"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جاري فحص الأصول الثمانية...")

    all_data: dict[str, md.AssetData] = {}
    reports:  list[de.AnalysisReport] = []

    for name in md.ASSETS:
        logger.info(f"جلب بيانات {name}...")
        data = md.load_asset_live(name)
        all_data[name] = data
        rep = de.analyze_asset(data)
        reports.append(rep)

    # تطبيق تحذيرات الارتباط
    de.apply_correlation_warnings(reports, all_data)

    signals = [r for r in reports if r.status == "signal"]
    watches = [r for r in reports if r.status == "watch"]
    skips   = [r for r in reports if r.status == "skip"]

    # --- إرسال الإشارات ---
    if signals:
        await update.message.reply_text(
            f"🎯 *{len(signals)} إشارة مكتملة الشروط:*",
            parse_mode="Markdown")
        for rep in signals:
            await update.message.reply_text(
                _fmt_signal(rep), parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "⏳ لا توجد إشارة مكتملة الشروط حالياً.")

    # --- قائمة المراقبة ---
    if watches:
        watch_text = "👀 *تحت المراقبة:*\n" + "\n".join(_fmt_watch(r) for r in watches)
        await update.message.reply_text(watch_text, parse_mode="Markdown")

    # --- متجاوزة ---
    if skips:
        skip_text = "⏭️ *تم التجاوز:*\n" + "\n".join(_fmt_skip(r) for r in skips)
        await update.message.reply_text(skip_text, parse_mode="Markdown")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تقرير تفصيلي لكل فريم بكل أصل."""
    await update.message.reply_text("📋 جاري إعداد التقارير التفصيلية...")

    for name in md.ASSETS:
        data = md.load_asset_live(name)
        rep  = de.analyze_asset(data)
        try:
            await update.message.reply_text(
                _fmt_frame_report(rep), parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"خطأ في إرسال تقرير {name}: {e}")
            await update.message.reply_text(
                _fmt_frame_report(rep))   # بدون Markdown كـ fallback


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now_utc = pd.Timestamp.now(tz="UTC")
    kz = md.kill_zone_name(now_utc)
    kz_icon = "🟢" if md.in_kill_zone(now_utc) else "🔴"

    text = (
        f"🟢 *الحالة: متصل*\n\n"
        f"🕒 الوقت (UTC): {now_utc.strftime('%H:%M')}\n"
        f"{kz_icon} الجلسة: {kz}\n\n"
        f"*Kill Zones (UTC):*\n"
        f"  🇬🇧 London: 07:00–10:00\n"
        f"  🇺🇸 New York: 13:00–16:00\n\n"
        f"*الفريمات:* D1 (اتجاه) → H4 (CHoCH + POI) → M15 (دخول)\n"
        f"*TP:* HTF Liquidity (Equal H/L + Swing Points)\n"
        f"*SL:* Swept Level ± 0.5 × ATR\n\n"
        f"*الأصول ({len(md.ASSETS)}):*\n"
        + "\n".join(f"  • {k}" for k in md.ASSETS) +
        f"\n\n*الرصيد الافتراضي:* {md.ACCOUNT_BALANCE_USD}$ | "
        f"*مخاطرة:* {md.RISK_PERCENT}% / صفقة\n"
        f"*هامش الأخبار:* ±{md.NEWS_BUFFER_MIN} دقيقة"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# نقطة الدخول
# ---------------------------------------------------------------------------

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("status", cmd_status))
    logger.info("🤖 البوت يعمل...")
    app.run_polling()


if __name__ == "__main__":
    main()
