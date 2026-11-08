"""
backtest_v2.py  —  باكتست البنية الخماسية
==========================================
يختبر نفس منطق decision_engine على بيانات تاريخية حقيقية.

مصادر البيانات:
  • yfinance (افتراضي): 15m لآخر 60 يوم — سريع، بدون تسجيل
  • Polygon.io (--source polygon --key KEY): 15m لسنة كاملة

تشغيل:
    python backtest_v2.py
    python backtest_v2.py --source polygon --key YOUR_KEY
    python backtest_v2.py --source polygon --key YOUR_KEY --days 365
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

import analysis_engine as ae
import market_data as md
import smc_engine as smc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_HOLD_BARS = 96   # أقصى شموع 15m ننتظرها (~يوم كامل)


# ---------------------------------------------------------------------------
# جلب البيانات التاريخية
# ---------------------------------------------------------------------------

def load_historical(name: str, source: str = "yfinance",
                     api_key: str = "", days: int = 60) -> md.AssetData:
    if source == "polygon" and api_key:
        return md.load_asset_polygon(name, api_key, backtest_days=days)
    else:
        return md.load_asset_live(name)


# ---------------------------------------------------------------------------
# إزالة Timezone للتصدير
# ---------------------------------------------------------------------------

def _strip_tz(ts):
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


# ---------------------------------------------------------------------------
# محاكاة أصل واحد (Vectorized حيث ممكن)
# ---------------------------------------------------------------------------

def simulate_asset(name: str, asset_data: md.AssetData) -> pd.DataFrame:
    """
    يطبّق نفس منطق decision_engine لكن بشكل دُفعي على البيانات التاريخية.

    الفرق الوحيد عن البوت الحي:
    - فلتر Kill Zone مطبّق (نفس البوت)
    - فلتر الأخبار غير مطبّق (البيانات التاريخية لا تتوفر مجاناً)
    - كل الشروط الإلزامية مطبّقة: Sweep، HTF POI، IFVG، CISD، CHoCH
    """
    if not asset_data.is_valid:
        logger.warning(f"[{name}] بيانات غير كافية")
        return pd.DataFrame()

    df_15m  = asset_data.m15
    df_h4   = asset_data.h4
    df_d1   = asset_data.d1

    n = len(df_15m)
    if n < 50:
        return pd.DataFrame()

    # اتجاه H4 و D1 لكل لحظة (Vectorized + shift لتجنب lookahead)
    def _trend_series(df: pd.DataFrame) -> pd.Series:
        """نفس منطق analyze_trend لكن series كاملة."""
        close = df["Close"]
        e50   = ae.ema(close, 50)
        e200  = ae.ema(close, 200) if len(df) >= 205 else None
        trend = pd.Series("neutral", index=df.index)
        if e200 is not None:
            up   = (e50 > e200) & (close > e50)
            down = (e50 < e200) & (close < e50)
        else:
            up   = close > e50
            down = close < e50
        trend[up]   = "bullish"
        trend[down] = "bearish"
        return trend.shift(1)   # لا lookahead

    trend_d1_s = _trend_series(df_d1)
    trend_h4_s = _trend_series(df_h4)

    idx15     = df_15m.index
    d1_on_15  = trend_d1_s.reindex(idx15, method="ffill")
    h4_on_15  = trend_h4_s.reindex(idx15, method="ffill")

    atr_series = ae.atr(df_15m, 14)
    rsi_series = ae.rsi(df_15m["Close"], 14)

    trades        = []
    cooldown_until = -1
    i = 30

    while i < n - 1:
        if i <= cooldown_until:
            i += 1
            continue

        bar_time = df_15m.index[i]
        bar_utc  = bar_time if bar_time.tzinfo else bar_time.tz_localize("UTC")

        # Kill Zone
        if not md.in_kill_zone(bar_utc):
            i += 1
            continue

        # D1 اتجاه
        t_d1 = d1_on_15.iloc[i]
        if pd.isna(t_d1) or t_d1 == "neutral":
            i += 1
            continue

        # H4 يجب ألا يعارض D1
        t_h4 = h4_on_15.iloc[i]
        if pd.isna(t_h4) or (t_h4 not in (t_d1, "neutral")):
            i += 1
            continue

        direction_str = t_d1
        trade_dir     = "buy" if direction_str == "bullish" else "sell"

        # ATR
        atr_val = atr_series.iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            i += 1
            continue

        # RSI كفلتر
        rsi_val = rsi_series.iloc[i]
        if not pd.isna(rsi_val):
            if trade_dir == "buy"  and rsi_val >= 70: i += 1; continue
            if trade_dir == "sell" and rsi_val <= 30: i += 1; continue

        # --- CHoCH على H4 (نسخة مبسّطة للباكتست السريع) ---
        h4_window = df_h4[df_h4.index <= bar_time].iloc[-60:]
        if len(h4_window) < 10:
            i += 1
            continue
        h4_shifts = smc.detect_structure_shifts(h4_window, "H4")
        if not smc.has_choch(h4_shifts, direction_str):
            i += 1
            continue

        # --- Sweep على M15 ---
        m15_window = df_15m.iloc[max(0, i - 20): i + 1]
        sweep = smc.detect_liquidity_sweep(m15_window, trade_dir)
        if sweep is None:
            i += 1
            continue

        # --- HTF POI ---
        d1_hist = df_d1[df_d1.index <= bar_time].iloc[-200:]
        h4_hist = df_h4[df_h4.index <= bar_time].iloc[-100:]
        htf_pois = smc.find_htf_pois(d1_hist, h4_hist, direction_str)

        last_close = df_15m["Close"].iloc[i]
        active_poi = None
        for poi in htf_pois:
            if smc.price_near_poi(last_close, poi):
                active_poi = poi
                break
        if active_poi is None:
            i += 1
            continue

        # --- IFVG على M15 ---
        _, m15_ifvgs = smc.find_fvgs(m15_window, direction_str)
        if not smc.price_at_ifvg(last_close, m15_ifvgs, direction_str):
            i += 1
            continue

        # --- CISD على M15 ---
        if not smc.detect_cisd(m15_window, trade_dir,
                                poi_high=active_poi.high,
                                poi_low=active_poi.low):
            i += 1
            continue

        # --- كل الشروط متوفرة: حساب SL/TP ---
        entry = last_close
        if trade_dir == "buy":
            sl   = sweep.swept_level - 0.5 * atr_val
            risk = entry - sl
            tp_fallback = entry + risk * 2
        else:
            sl   = sweep.swept_level + 0.5 * atr_val
            risk = sl - entry
            tp_fallback = entry - risk * 2

        if risk <= 0:
            i += 1
            continue

        # TP على HTF Liquidity
        tp_htf = smc.next_htf_target(h4_hist, d1_hist, trade_dir, entry)
        tp = tp_htf if tp_htf else tp_fallback

        # توافق SMC إضافي (للتوثيق فقط)
        ctx = df_15m.iloc[max(0, i - 30): i + 1]
        m15_obs   = smc.find_order_blocks(ctx, direction_str)
        m15_shifts = smc.detect_structure_shifts(ctx, "M15")
        has_ob    = len(m15_obs) > 0
        has_mss   = smc.has_mss(m15_shifts, direction_str)
        has_e20   = (last_close > ae.ema(ctx["Close"], 20).iloc[-1]
                     if trade_dir == "buy"
                     else last_close < ae.ema(ctx["Close"], 20).iloc[-1])
        confluence = 5 + sum([has_ob, has_mss, has_e20])

        # --- محاكاة سير الصفقة ---
        outcome    = "time_stop"
        exit_price = entry
        exit_i     = min(i + MAX_HOLD_BARS, n - 1)

        for j in range(i + 1, min(i + MAX_HOLD_BARS, n)):
            bar = df_15m.iloc[j]
            if trade_dir == "buy":
                hit_sl = bar["Low"]  <= sl
                hit_tp = bar["High"] >= tp
            else:
                hit_sl = bar["High"] >= sl
                hit_tp = bar["Low"]  <= tp

            if hit_sl and hit_tp:
                outcome, exit_price, exit_i = "sl",  sl,  j
                break
            elif hit_sl:
                outcome, exit_price, exit_i = "sl",  sl,  j
                break
            elif hit_tp:
                outcome, exit_price, exit_i = "tp",  tp,  j
                break
        else:
            exit_price = df_15m.iloc[exit_i]["Close"]

        r_multiple = (
            (exit_price - entry) / risk if trade_dir == "buy"
            else (entry - exit_price) / risk
        )

        trades.append({
            "asset":        name,
            "direction":    trade_dir,
            "entry_time":   _strip_tz(df_15m.index[i]),
            "exit_time":    _strip_tz(df_15m.index[exit_i]),
            "entry":        round(entry,       6),
            "sl":           round(sl,          6),
            "tp":           round(tp,          6),
            "tp_htf":       tp_htf is not None,
            "swept_level":  round(sweep.swept_level, 6),
            "poi_source":   active_poi.source,
            "poi_kind":     active_poi.kind,
            "outcome":      outcome,
            "r_multiple":   round(r_multiple,  3),
            "confluence":   confluence,
            "has_ob":       has_ob,
            "has_mss":      has_mss,
            "has_ema20":    has_e20,
        })

        cooldown_until = exit_i
        i += 1

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# تصدير Excel
# ---------------------------------------------------------------------------

def export_excel(trades: pd.DataFrame) -> str:
    balance = md.ACCOUNT_BALANCE_USD
    equity  = []
    for r in trades["r_multiple"]:
        risk_amt = balance * (md.RISK_PERCENT / 100)
        balance += r * risk_amt
        equity.append(round(balance, 2))

    trades = trades.copy()
    trades["equity_after"] = equity

    rename = {
        "asset":        "الأصل",
        "direction":    "الاتجاه",
        "entry_time":   "تاريخ الدخول",
        "exit_time":    "تاريخ الخروج",
        "entry":        "سعر الدخول",
        "sl":           "وقف الخسارة",
        "tp":           "الهدف",
        "tp_htf":       "HTF Target",
        "swept_level":  "مستوى Sweep",
        "poi_source":   "POI المصدر",
        "poi_kind":     "POI النوع",
        "outcome":      "سبب الإغلاق",
        "r_multiple":   "R",
        "confluence":   "توافق",
        "has_ob":       "OB",
        "has_mss":      "MSS",
        "has_ema20":    "EMA20",
        "equity_after": "الرصيد",
    }
    t = trades.rename(columns=rename)
    t.insert(0, "رقم", range(1, len(t) + 1))
    t["الاتجاه"] = t["الاتجاه"].map({"buy": "شراء", "sell": "بيع"})
    t["سبب الإغلاق"] = t["سبب الإغلاق"].map(
        {"sl": "وقف الخسارة", "tp": "الهدف", "time_stop": "زمني"})
    t["النتيجة"] = t["R"].apply(lambda r: "✅" if r > 0 else "❌")

    for col in ["تاريخ الدخول", "تاريخ الخروج"]:
        if col in t.columns:
            t[col] = t[col].apply(_strip_tz)
            t[col] = pd.to_datetime(t[col])

    # ملخص إجمالي
    total  = len(trades)
    wins   = trades[trades["r_multiple"] > 0]
    losses = trades[trades["r_multiple"] <= 0]
    wr     = len(wins) / total * 100 if total else 0
    pf_num = wins["r_multiple"].sum()
    pf_den = abs(losses["r_multiple"].sum())
    pf     = pf_num / pf_den if pf_den > 0 else float("inf")

    eq_s   = pd.Series([md.ACCOUNT_BALANCE_USD] + equity)
    peak   = eq_s.cummax()
    max_dd = ((eq_s - peak) / peak * 100).min()

    summary = [
        ("الصفقات الكلية",       total),
        ("الصفقات الرابحة",      len(wins)),
        ("الصفقات الخاسرة",      len(losses)),
        ("نسبة النجاح %",         round(wr, 1)),
        ("Profit Factor",          round(pf, 2)),
        ("متوسط R",                round(trades["r_multiple"].mean(), 3)),
        ("إجمالي R",               round(trades["r_multiple"].sum(), 2)),
        ("الرصيد الابتدائي $",    md.ACCOUNT_BALANCE_USD),
        ("الرصيد النهائي $",      round(balance, 2)),
        ("ربح صافي $",             round(balance - md.ACCOUNT_BALANCE_USD, 2)),
        ("Max Drawdown %",          round(max_dd, 1)),
        ("صفقات بـ HTF Target",    int(trades["tp_htf"].sum())),
    ]

    asset_rows = []
    for a in md.ASSETS:
        sub = trades[trades["asset"] == a]
        if not len(sub):
            continue
        wr_a  = (sub["r_multiple"] > 0).mean() * 100
        pf_a  = (sub[sub["r_multiple"] > 0]["r_multiple"].sum() /
                 max(abs(sub[sub["r_multiple"] <= 0]["r_multiple"].sum()), 1e-9))
        asset_rows.append({
            "الأصل": a,
            "الصفقات": len(sub),
            "نسبة النجاح %": round(wr_a, 1),
            "Profit Factor": round(pf_a, 2),
            "إجمالي R": round(sub["r_multiple"].sum(), 2),
            "HTF Targets": int(sub["tp_htf"].sum()),
        })

    fname = "backtest_v2_results.xlsx"
    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        t.to_excel(writer, sheet_name="الصفقات", index=False)
        pd.DataFrame(summary, columns=["البند", "القيمة"]).to_excel(
            writer, sheet_name="الملخص", index=False)
        if asset_rows:
            pd.DataFrame(asset_rows).to_excel(
                writer, sheet_name="ملخص الأصول", index=False)

        for sheet in writer.sheets.values():
            for col in sheet.columns:
                w = max((len(str(c.value)) for c in col if c.value), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(w + 4, 45)

    return fname, balance, wr, pf, max_dd, equity


# ---------------------------------------------------------------------------
# الدالة الرئيسية
# ---------------------------------------------------------------------------

def run_backtest(source: str = "yfinance", api_key: str = "",
                  days: int = 60) -> None:
    logger.info(f"بدء الباكتست | مصدر: {source} | {days} يوم | {len(md.ASSETS)} أصل")

    all_trades = []
    for name in md.ASSETS:
        logger.info(f"[{name}] جلب البيانات...")
        data = load_historical(name, source, api_key, days)
        logger.info(f"[{name}] محاكاة...")
        df = simulate_asset(name, data)
        if not df.empty:
            logger.info(f"  ← {len(df)} صفقة")
            all_trades.append(df)

    if not all_trades:
        print("لا توجد صفقات — تحقق من البيانات والإعدادات.")
        return

    trades = pd.concat(all_trades, ignore_index=True).sort_values("entry_time")

    print("\n========== نتائج الباكتست v2 ==========")
    total  = len(trades)
    wins   = trades[trades["r_multiple"] > 0]
    losses = trades[trades["r_multiple"] <= 0]
    wr     = len(wins) / total * 100 if total else 0
    pf_n   = wins["r_multiple"].sum()
    pf_d   = abs(losses["r_multiple"].sum())
    pf     = pf_n / pf_d if pf_d > 0 else float("inf")

    print(f"الصفقات : {total}")
    print(f"نسبة نجاح: {wr:.1f}%")
    print(f"PF       : {pf:.2f}")
    print(f"R إجمالي : {trades['r_multiple'].sum():.2f}")
    print()
    for a in md.ASSETS:
        sub = trades[trades["asset"] == a]
        if len(sub):
            wr_a = (sub["r_multiple"] > 0).mean() * 100
            print(f"  {a:12}: {len(sub):3} صفقة | {wr_a:.1f}% | "
                  f"R={sub['r_multiple'].sum():.2f}")

    try:
        fname, final_bal, _, _, max_dd, equity = export_excel(trades)
        print(f"\nالرصيد النهائي: {final_bal:.2f}$")
        print(f"Max Drawdown  : {max_dd:.1f}%")
        print(f"Excel         : {fname}")
    except Exception as e:
        logger.error(f"خطأ في التصدير: {e}")
        trades.to_csv("backtest_v2_results.csv", index=False)
        print("CSV: backtest_v2_results.csv")

    # منحنى رأس المال
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, _, _, _, _, equity_curve = export_excel(trades)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

        ax1.plot(equity_curve, color="#2ecc71", lw=1.5)
        ax1.axhline(y=md.ACCOUNT_BALANCE_USD, color="gray",
                    ls="--", alpha=0.5, label="رأس المال الابتدائي")
        ax1.set_title("منحنى رأس المال — باكتست v2 (5 طبقات)")
        ax1.set_xlabel("رقم الصفقة")
        ax1.set_ylabel("الرصيد ($)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.hist(trades["r_multiple"], bins=30,
                 color="#3498db", alpha=0.7, edgecolor="white")
        ax2.axvline(x=0, color="red", ls="--", alpha=0.7)
        ax2.set_title("توزيع نتائج الصفقات (R-multiple)")
        ax2.set_xlabel("R"); ax2.set_ylabel("عدد الصفقات")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("equity_curve_v2.png", dpi=150)
        print("الرسم: equity_curve_v2.png")
    except Exception as e:
        logger.warning(f"تعذّر الرسم: {e}")

    print("=" * 42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest v2 — البنية الخماسية")
    parser.add_argument("--source", default="yfinance",
                        choices=["yfinance", "polygon"],
                        help="مصدر البيانات (افتراضي: yfinance)")
    parser.add_argument("--key",  default="",
                        help="Polygon.io API Key (مطلوب مع --source polygon)")
    parser.add_argument("--days", type=int, default=60,
                        help="عدد أيام الباكتست (يfinance: 60، Polygon: 365)")
    args = parser.parse_args()

    if args.source == "polygon" and not args.key:
        print("❌ --key مطلوب مع --source polygon")
    else:
        run_backtest(source=args.source, api_key=args.key, days=args.days)
