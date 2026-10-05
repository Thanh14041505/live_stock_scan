#!/usr/bin/env python3
"""VN Quant Desk — Streamlit scanner gom 9 notebook (Ichimoku / VSA / RSI-DG / Daily)."""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from vn_quant.engine import (
    VN100,
    AppDataLayer,
    QuantConfig,
    QuantEngine,
    register_api_key,
    scan_universe,
)
from vn_quant.smc import analyze_smc, smc_plotly

st.set_page_config(
    page_title="VN Quant Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.stApp { background: radial-gradient(1200px 500px at 10% -10%, #163024 0%, transparent 50%), #0B1220; }

.hero {
  border: 1px solid #1E2A3F;
  background: linear-gradient(180deg, #121A2A 0%, #0E1624 100%);
  border-radius: 16px;
  padding: 18px 22px;
  margin-bottom: 14px;
}
.hero h1 { margin: 0; font-size: 1.55rem; color: #F4F7FB; }
.hero p { margin: 6px 0 0; color: #9AA8BC; font-size: 0.92rem; }
.badge {
  display: inline-block; font-family: "IBM Plex Mono", monospace;
  font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 999px; margin-right: 6px;
  border: 1px solid #2A3B55; color: #9AA8BC;
}
.badge.live { color: #3DDC97; border-color: #1F6A4A; background: #0F2A1E; }
.badge.demo { color: #F5C451; border-color: #6A5218; background: #2A210C; }

.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0 16px; }
.metric {
  background: #121A2A; border: 1px solid #1E2A3F; border-radius: 12px; padding: 12px 14px;
}
.metric .k { color: #9AA8BC; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
.metric .v { color: #F4F7FB; font-size: 1.4rem; font-weight: 700; font-family: "IBM Plex Mono", monospace; margin-top: 4px; }
.metric .v.buy { color: #3DDC97; }
.metric .v.watch { color: #F5C451; }
.metric .v.exit { color: #FF6B6B; }

div[data-testid="stDataFrame"] { border: 1px solid #1E2A3F; border-radius: 12px; overflow: hidden; }
.stButton>button {
  background: #3DDC97 !important; color: #062016 !important; font-weight: 700 !important;
  border: 0 !important; border-radius: 10px !important;
}
.hint { color: #9AA8BC; font-size: 0.85rem; }
footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DEFAULT_WL = "FPT, VCB, HPG, TCB, MWG, SSI, VHM, MSN, GAS, STB, MBB, HDB, GVR, PNJ, VIC"


def pill(kind: str) -> str:
    color = {"BUY": "#3DDC97", "WATCH": "#F5C451", "EXIT": "#FF6B6B", "SKIP": "#6B7789"}.get(kind, "#6B7789")
    return f'<span style="color:{color};font-weight:700;font-family:IBM Plex Mono,monospace">{kind}</span>'


def show_df(df: pd.DataFrame, extra_cols=None):
    if df is None or df.empty:
        st.info("Chưa có mã trong nhóm này.")
        return
    cols = [
        "Mã", "consensus", "votes_buy", "votes_watch", "votes_exit",
        "S_Ichimoku", "S_VSA", "S_RSI_DG", "S_Daily", "S_SMC",
        "Tín hiệu", "Score", "Win%", "Giá", "VSA", "Ichimoku",
        "PK RSI (Composite)", "Cảnh báo DG", "Dòng tiền", "Trend",
        "SMC bias", "SMC PD", "SMC POI", "SMC entry", "SMC SL", "SMC TP1", "SMC TP2", "SMC R:R",
        "Entry", "SL T+", "T+ T1", "T+ %", "T+ R:R",
        "Khuyến nghị Hold", "Action",
    ]
    if extra_cols:
        cols.extend(extra_cols)
    show = [c for c in cols if c in df.columns]
    view = df[show].copy()
    st.dataframe(view, width="stretch", hide_index=True, height=min(560, 80 + 36 * len(view)))


with st.sidebar:
    st.markdown("### VN Quant Desk")
    st.caption("Gom Ichimoku 3TF · VSA · RSI/Dao Găm · Daily EXIT · SMC")

    api_key = st.text_input(
        "vnstock API key",
        type="password",
        placeholder="vnstock_xxxxxxxx",
        help="Dán key từ vnstocks.com. Không lưu server — chỉ trong session này.",
    )
    demo = st.toggle("Chế độ DEMO (data giả, không cần API)", value=True)
    if api_key and not demo:
        st.session_state["_key_status"] = register_api_key(api_key)
        st.success("Đã nạp API key vào session.")
    elif not demo and not api_key:
        st.warning("Thiếu API key — bật DEMO hoặc dán key.")

    st.divider()
    mode = st.radio("Khung giao dịch", ["swing", "hold"], horizontal=True, format_func=lambda x: "Swing T+" if x == "swing" else "Hold")
    capital_m = st.number_input("Vốn (triệu VNĐ)", min_value=10, max_value=100000, value=100, step=10)
    risk = st.slider("Risk / lệnh (%)", 0.25, 2.0, 1.0, 0.25)
    min_tp = st.slider("T1 tối thiểu (%)", 5.0, 15.0, 9.0, 0.5)

    st.markdown("**Triết lý bật**")
    use_ichi = st.checkbox("Ichimoku 3TF (Claude)", True)
    use_vsa = st.checkbox("VSA Smart Money (Grok)", True)
    use_dg = st.checkbox("RSI Divergence + Dao Găm", True)
    use_daily = st.checkbox("Daily BUY / WATCH / EXIT", True)
    use_smc = st.checkbox("SMC (structure / OB / FVG)", True)

    universe_choice = st.selectbox("Watchlist", ["Top thanh khoản (15 mã)", "VN100+", "Tự nhập"])
    custom = st.text_area("Danh sách mã (phẩy hoặc xuống dòng)", DEFAULT_WL, height=90)
    delay = st.slider("Delay API (giây)", 0.0, 3.0, 0.0 if demo else 1.6, 0.1)
    run_btn = st.button("Chạy scan hôm nay", width="stretch")

active = []
if use_ichi:
    active.append("Ichimoku")
if use_vsa:
    active.append("VSA")
if use_dg:
    active.append("RSI/DG")
if use_daily:
    active.append("Daily")
if use_smc:
    active.append("SMC")

mode_badge = "demo" if demo else "live"
st.markdown(
    f"""
<div class="hero">
  <span class="badge {mode_badge}">{'DEMO DATA' if demo else 'LIVE VNSTOCK'}</span>
  <span class="badge">MODE {mode.upper()}</span>
  <span class="badge">{datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
  <h1>Bàn scan tín hiệu cổ phiếu VN</h1>
  <p>Năm triết lý chạy song song trên cùng bộ nến, rồi vote consensus.
  Đang bật: {", ".join(active) or "không có"}.</p>
</div>
""",
    unsafe_allow_html=True,
)

if universe_choice == "VN100+":
    symbols = list(dict.fromkeys(VN100))
elif universe_choice == "Tự nhập":
    symbols = [s.strip().upper() for s in custom.replace("\n", ",").split(",") if s.strip()]
else:
    symbols = [s.strip().upper() for s in DEFAULT_WL.split(",") if s.strip()]

if run_btn:
    if not demo and not api_key:
        st.error("Cần API key hoặc bật chế độ DEMO.")
    else:
        cfg = QuantConfig(
            capital=float(capital_m) * 1_000_000,
            risk_per_trade=float(risk) / 100.0,
            min_tp_pct=float(min_tp),
            delay_sec=float(delay),
            lookback_short=80,
            lookback_long=400 if demo else 600,
        )
        engine = QuantEngine(capital=cfg.capital, cfg=cfg)
        dl = AppDataLayer(cfg, demo=demo)
        engine.dl = dl
        engine.regime.dl = dl

        bar = st.progress(0, text="Khởi động scanner…")
        status = st.empty()

        def on_progress(i, n, sym):
            pct = 0 if n == 0 else min(i / n, 1.0)
            bar.progress(pct, text=f"Đang quét {i}/{n} — {sym}")
            status.caption(f"{sym}")

        with st.spinner("Đang tính chỉ báo + 4 chiến lược…"):
            df, meta = scan_universe(engine, symbols, mode=mode, delay=float(delay), progress_cb=on_progress)
        bar.empty()
        status.empty()
        st.session_state["scan_df"] = df
        st.session_state["scan_meta"] = meta
        st.session_state["scan_mode"] = mode
        st.session_state["data_layer"] = dl
        st.session_state["lookback_long"] = cfg.lookback_long

df = st.session_state.get("scan_df")
meta = st.session_state.get("scan_meta")

if df is None:
    st.markdown(
        """
        <p class="hint">Nhấn <b>Chạy scan hôm nay</b> ở sidebar.
        Lần đầu nên để DEMO để xem giao diện; khi deploy Streamlit Cloud hãy tắt DEMO và dán API key.</p>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Ichimoku 3TF**  \nHội tụ mây D/W/M, pullback Kijun, T1 ≥ 9%.")
    c2.markdown("**VSA + SMC**  \nNổ vol / Order Block / FVG / BOS-CHoCH.")
    c3.markdown("**RSI-DG + Daily**  \nPhân kỳ Composite, Dao Găm, cảnh báo EXIT.")
    st.stop()

# Filter by enabled philosophies: if a philosophy is off, ignore its vote in display
work = df.copy()
if not work.empty and "S_SMC" not in work.columns:
    work["S_SMC"] = "SKIP"
if not work.empty:
    vote_cols = []
    mapping = {
        "Ichimoku": "S_Ichimoku",
        "VSA": "S_VSA",
        "RSI/DG": "S_RSI_DG",
        "Daily": "S_Daily",
    }
    enabled_cols = []
    if use_ichi:
        enabled_cols.append("S_Ichimoku")
    if use_vsa:
        enabled_cols.append("S_VSA")
    if use_dg:
        enabled_cols.append("S_RSI_DG")
    if use_daily:
        enabled_cols.append("S_Daily")
    if use_smc:
        enabled_cols.append("S_SMC")
    if enabled_cols:
        work["votes_buy"] = work[enabled_cols].apply(lambda r: int((r == "BUY").sum()), axis=1)
        work["votes_watch"] = work[enabled_cols].apply(lambda r: int((r == "WATCH").sum()), axis=1)
        work["votes_exit"] = work[enabled_cols].apply(lambda r: int((r == "EXIT").sum()), axis=1)

        def recon(row):
            if row["votes_exit"] >= max(1, len(enabled_cols) // 2):
                return "EXIT"
            if row["votes_buy"] >= 2 and row["votes_exit"] == 0:
                return "BUY"
            if row["votes_buy"] + row["votes_watch"] >= 1 and row["votes_exit"] == 0:
                return "WATCH" if row["votes_buy"] < 2 else "BUY"
            if row["votes_exit"] >= 1:
                return "EXIT"
            return "SKIP"

        work["consensus"] = work.apply(recon, axis=1)

n_buy = int((work["consensus"] == "BUY").sum()) if not work.empty else 0
n_watch = int((work["consensus"] == "WATCH").sum()) if not work.empty else 0
n_exit = int((work["consensus"] == "EXIT").sum()) if not work.empty else 0
regime = (meta or {}).get("regime", {})
label = regime.get("label", "—")
rname = str(regime.get("regime", "—")).upper()

st.markdown(
    f"""
<div class="metric-row">
  <div class="metric"><div class="k">Regime</div><div class="v">{rname}</div><div class="k">{label}</div></div>
  <div class="metric"><div class="k">Consensus BUY</div><div class="v buy">{n_buy}</div></div>
  <div class="metric"><div class="k">WATCH</div><div class="v watch">{n_watch}</div></div>
  <div class="metric"><div class="k">EXIT / REDUCE</div><div class="v exit">{n_exit}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

errs = (meta or {}).get("errors") or []
if errs:
    with st.expander(f"Mã lỗi / thiếu data ({len(errs)})"):
        st.dataframe(pd.DataFrame(errs), hide_index=True, width="stretch")

tab_c, tab_b, tab_w, tab_e, tab_all, tab_x, tab_smc, tab_d = st.tabs(
    ["Consensus", "BUY", "WATCH", "EXIT", "Toàn bộ", "Conflict", "SMC + Chart", "Chi tiết 1 mã"]
)

with tab_c:
    st.caption("Mã được xếp theo consensus. BUY cần ≥2 triết lý cùng chiều và không bị EXIT.")
    show_df(work[work["consensus"].isin(["BUY", "WATCH", "EXIT"])] if not work.empty else work)

with tab_b:
    show_df(work[work["consensus"] == "BUY"] if not work.empty else work)

with tab_w:
    show_df(work[work["consensus"] == "WATCH"] if not work.empty else work)

with tab_e:
    show_df(work[work["consensus"] == "EXIT"] if not work.empty else work)

with tab_all:
    show_df(work)

with tab_x:
    if work.empty:
        st.info("Chưa có dữ liệu.")
    else:
        conf = work[
            ((work["votes_buy"] >= 1) & (work["votes_exit"] >= 1))
            | (work[enabled_cols].nunique(axis=1) >= 3)
        ]
        st.caption("Mã mà các triết lý mâu thuẫn — đúng use-case bạn mô tả.")
        show_df(conf)

with tab_smc:
    st.caption(
        "SMC đúng nghĩa, long-only (sàn VN không short): swing → BOS/CHoCH → demand OB/FVG. "
        "Bias giảm = EXIT/WATCH, không đặt entry trên giá hay TP dưới giá."
    )
    if work.empty:
        st.info("Chạy scan trước.")
    else:
        smc_view = work.sort_values(["S_SMC", "_score"] if "_score" in work.columns else ["S_SMC"], ascending=[True, False])
        show_df(smc_view, extra_cols=["SMC thesis"])
        pick_smc = st.selectbox("Đồ thị SMC theo mã", list(work["Mã"].astype(str)), key="smc_pick")
        dl = st.session_state.get("data_layer")
        lb = int(st.session_state.get("lookback_long") or 400)
        ohlc = dl.fetch(pick_smc, lb) if dl is not None else None
        if ohlc is None:
            st.warning("Không lấy được nến để vẽ SMC.")
        else:
            plan = analyze_smc(ohlc)
            if plan is None:
                st.warning("Không đủ nến để đọc structure.")
            else:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Bias", plan.bias.upper())
                m2.metric("SMC", plan.action)
                m3.metric("PD", f"{plan.pd} ({plan.pd_pct:.0%})")
                m4.metric("Entry", f"{plan.entry_low:.2f}–{plan.entry_high:.2f}" if plan.has_long_setup else "không long")
                m5.metric("R:R", f"1:{plan.rr:.1f}" if plan.has_long_setup else "—")
                st.plotly_chart(smc_plotly(ohlc, plan, title=f"{pick_smc} · {plan.structure}"), width="stretch")
                left, right = st.columns(2)
                with left:
                    st.markdown("**Nhận định**")
                    st.write(plan.thesis)
                    if plan.has_long_setup:
                        lv = (
                            f"- POI: {plan.poi_label}\n"
                            f"- Entry long **{plan.entry_low:.2f}–{plan.entry_high:.2f}** (dưới/sát giá)\n"
                            f"- SL {plan.sl:.2f} (dưới entry) · TP1 {plan.tp1:.2f} · TP2 {plan.tp2:.2f} (trên giá)"
                        )
                    else:
                        lv = (
                            f"- Không mở long. POI: {plan.poi_label}\n"
                            f"- Nếu đang cầm: cắt dưới swing low **{plan.invalidation:.2f}**"
                        )
                    st.markdown(
                        f"- Dealing range: **{plan.dealing_low:.2f} – {plan.dealing_high:.2f}** (EQ {plan.eq:.2f})\n"
                        f"{lv}\n"
                        f"- Liquidity trên: {', '.join(map(str, plan.liquidity_up)) or '—'}\n"
                        f"- Liquidity dưới: {', '.join(map(str, plan.liquidity_dn)) or '—'}"
                    )
                with right:
                    st.markdown("**Chiến thuật**")
                    for t in plan.tactics:
                        st.markdown(f"- {t}")

with tab_d:
    if work.empty:
        st.info("Chạy scan trước.")
    else:
        pick = st.selectbox("Chọn mã", list(work["Mã"].astype(str)))
        row = work[work["Mã"] == pick].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Giá", row.get("Giá", "—"))
        b.metric("Score", row.get("Score", "—"))
        c.metric("Win%", row.get("Win%", "—"))
        d.metric("Consensus", row.get("consensus", "—"))
        st.markdown(
            f"""
**Tín hiệu:** {row.get('Tín hiệu','—')}  
**Ichimoku:** {row.get('Ichimoku','—')} · S1 {row.get('S1','—')} / R1 {row.get('R1','—')}  
**VSA:** {row.get('VSA','—')} · **Dòng tiền:** {row.get('Dòng tiền','—')}  
**Phân kỳ:** {row.get('PK RSI (Composite)', row.get('Phân kỳ','—'))}  
**Dao Găm:** {row.get('Cảnh báo DG','—')}  
**Entry** {row.get('Entry','—')} ({row.get('Entry zone','—')}) · **SL T+** {row.get('SL T+','—')}  
**T+ T1** {row.get('T+ T1','—')} (+{row.get('T+ %','—')}%) · R:R {row.get('T+ R:R','—')}  
**Hold** T1 {row.get('Hold T1','—')} · {row.get('Khuyến nghị Hold','—')}  
**Action:** {row.get('Action','—')} · {row.get('Giá trị','—')} · {row.get('% NAV','—')} NAV
"""
        )
        reasons = row.get("Lý do")
        if isinstance(reasons, list) and reasons:
            st.warning("Lý do lọc: " + " · ".join(map(str, reasons)))
        st.markdown("Vote từng triết lý")
        st.write(
            {
                "Ichimoku": row.get("S_Ichimoku"),
                "VSA": row.get("S_VSA"),
                "RSI/DG": row.get("S_RSI_DG"),
                "Daily": row.get("S_Daily"),
                "SMC": row.get("S_SMC"),
            }
        )
        st.markdown(
            f"**SMC:** {row.get('SMC bias','—')} · {row.get('SMC PD','—')} · {row.get('SMC POI','—')}  \n"
            f"Entry {row.get('SMC entry','—')} · SL {row.get('SMC SL','—')} · "
            f"TP1 {row.get('SMC TP1','—')} / TP2 {row.get('SMC TP2','—')} · {row.get('SMC R:R','—')}"
        )
        if row.get("SMC thesis"):
            st.caption(row.get("SMC thesis"))

csv = work.to_csv(index=False).encode("utf-8-sig") if not work.empty else b""
st.download_button(
    "Tải CSV kết quả",
    data=csv,
    file_name=f"vn_quant_scan_{datetime.now():%Y%m%d_%H%M}.csv",
    mime="text/csv",
    disabled=work.empty,
)

with st.expander("Deploy Streamlit Cloud"):
    st.markdown(
        """
1. Push repo chứa `app.py`, `vn_quant/`, `requirements.txt`.
2. Trên share.streamlit.io chọn file `app.py`.
3. Dán vnstock API key vào ô sidebar mỗi lần chạy (không commit key).
4. Tắt DEMO để lấy data thật. Delay ~1.6s nếu dùng gói community.
        """
    )
