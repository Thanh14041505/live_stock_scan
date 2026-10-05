"""Smart Money Concepts on daily VN OHLCV.

Implements the ICT / SMC stack used for cash equities (no session killzones):
  - swing structure → BOS / CHoCH
  - dealing range, premium / discount / EQ
  - unmitigated order blocks
  - 3-candle FVG
  - swing + equal-highs/lows liquidity
  - POI confluence → entry zone, SL, TP1/TP2
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

Bias = Literal["bull", "bear", "range"]
Action = Literal["BUY", "WATCH", "EXIT", "SKIP"]


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Confirmed fractal swings. A pivot at i is only known at i+right."""
    n = len(df)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        w_h = highs[i - left : i + right + 1]
        w_l = lows[i - left : i + right + 1]
        if highs[i] >= w_h.max() and (w_h == highs[i]).sum() == 1:
            sh[i] = True
        if lows[i] <= w_l.min() and (w_l == lows[i]).sum() == 1:
            sl[i] = True
    out = df[["time", "open", "high", "low", "close"]].copy()
    out["swing_high"] = sh
    out["swing_low"] = sl
    return out


@dataclass
class Swing:
    i: int
    price: float
    kind: str  # "H" | "L"
    time: object


@dataclass
class Event:
    i: int
    kind: str  # BOS_UP / BOS_DN / CHOCH_UP / CHOCH_DN
    price: float
    time: object


@dataclass
class Zone:
    kind: str  # BULL_OB / BEAR_OB / BULL_FVG / BEAR_FVG
    i0: int
    i1: int
    low: float
    high: float
    time0: object
    time1: object
    mitigated: bool = False
    mid: float = 0.0

    def __post_init__(self):
        self.mid = (self.low + self.high) / 2.0


@dataclass
class SmcPlan:
    bias: Bias
    last_event: str
    structure: str
    dealing_low: float
    dealing_high: float
    eq: float
    pd: str  # PREMIUM / DISCOUNT / EQ
    pd_pct: float  # 0 = discount extreme, 1 = premium extreme
    poi: Optional[Zone]
    poi_label: str
    entry_low: float
    entry_high: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    action: Action
    thesis: str
    tactics: list
    liquidity_up: list
    liquidity_dn: list
    events: list
    zones: list
    swings: list
    has_long_setup: bool = True
    invalidation: float = 0.0


def _alternating_swings(df: pd.DataFrame) -> list[Swing]:
    rows = []
    for i, r in df.iterrows():
        if r["swing_high"]:
            rows.append(Swing(int(i), float(r["high"]), "H", r["time"]))
        if r["swing_low"]:
            rows.append(Swing(int(i), float(r["low"]), "L", r["time"]))
    rows.sort(key=lambda s: s.i)
    alt: list[Swing] = []
    for s in rows:
        if not alt:
            alt.append(s)
            continue
        if s.kind == alt[-1].kind:
            # keep more extreme
            if s.kind == "H" and s.price >= alt[-1].price:
                alt[-1] = s
            elif s.kind == "L" and s.price <= alt[-1].price:
                alt[-1] = s
        else:
            alt.append(s)
    return alt


def _structure_events(df: pd.DataFrame, swings: list[Swing]) -> tuple[list[Event], Bias, str]:
    events: list[Event] = []
    if len(swings) < 3:
        return events, "range", "Chưa đủ swing để đọc structure"

    last_h = next((s for s in reversed(swings) if s.kind == "H"), None)
    last_l = next((s for s in reversed(swings) if s.kind == "L"), None)
    # seed bias from last two opposite swings vs previous same kind
    hs = [s for s in swings if s.kind == "H"]
    ls = [s for s in swings if s.kind == "L"]
    bias: Bias = "range"
    if len(hs) >= 2 and len(ls) >= 2:
        if hs[-1].price > hs[-2].price and ls[-1].price > ls[-2].price:
            bias = "bull"
        elif hs[-1].price < hs[-2].price and ls[-1].price < ls[-2].price:
            bias = "bear"

    close = df["close"].to_numpy(float)
    times = df["time"].tolist()
    # Walk bars after each swing; detect close beyond last opposite swing
    confirmed_h = None
    confirmed_l = None
    trend: Bias = bias if bias != "range" else "range"
    for s in swings:
        if s.kind == "H":
            confirmed_h = s
        else:
            confirmed_l = s
        if confirmed_h is None or confirmed_l is None:
            continue
        start = s.i + 1
        end = min(len(df), start + 40)
        for j in range(start, end):
            if trend in ("bull", "range") and close[j] > confirmed_h.price:
                kind = "BOS_UP" if trend == "bull" else "CHOCH_UP"
                events.append(Event(j, kind, float(confirmed_h.price), times[j]))
                trend = "bull"
                break
            if trend in ("bear", "range") and close[j] < confirmed_l.price:
                kind = "BOS_DN" if trend == "bear" else "CHOCH_DN"
                events.append(Event(j, kind, float(confirmed_l.price), times[j]))
                trend = "bear"
                break

    # collapse consecutive same-kind events (keep last)
    compact: list[Event] = []
    for e in events:
        if compact and compact[-1].kind == e.kind:
            compact[-1] = e
        else:
            compact.append(e)
    last = compact[-1].kind if compact else "NONE"
    if last.endswith("UP"):
        trend = "bull"
    elif last.endswith("DN"):
        trend = "bear"
    label = {
        "bull": "HH / HL — xu hướng tăng (SMT bullish)",
        "bear": "LH / LL — xu hướng giảm (SMT bearish)",
        "range": "Range — chưa CHoCH rõ",
    }[trend]
    return compact, trend, label


def _order_blocks(df: pd.DataFrame, events: list[Event]) -> list[Zone]:
    """Last opposing candle before a displacement event."""
    zones: list[Zone] = []
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    times = df["time"].tolist()
    for e in events:
        bull = e.kind.endswith("UP")
        # search back up to 8 bars for last opposite candle
        found = None
        for k in range(e.i - 1, max(-1, e.i - 9), -1):
            bearish = c[k] < o[k]
            bullish = c[k] > o[k]
            if bull and bearish:
                found = k
                break
            if (not bull) and bullish:
                found = k
                break
        if found is None:
            continue
        z = Zone(
            kind="BULL_OB" if bull else "BEAR_OB",
            i0=found,
            i1=found,
            low=float(l[found]),
            high=float(h[found]),
            time0=times[found],
            time1=times[found],
        )
        # mitigated if later wick trades through 50%
        mid = z.mid
        after = slice(e.i + 1, None)
        if bull:
            z.mitigated = bool((l[after] <= mid).any()) if e.i + 1 < len(df) else False
        else:
            z.mitigated = bool((h[after] >= mid).any()) if e.i + 1 < len(df) else False
        zones.append(z)
    return zones


def _fvgs(df: pd.DataFrame) -> list[Zone]:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    times = df["time"].tolist()
    n = len(df)
    out: list[Zone] = []
    for i in range(1, n - 1):
        # bullish FVG: gap between high[i-1] and low[i+1]
        if l[i + 1] > h[i - 1]:
            z = Zone("BULL_FVG", i - 1, i + 1, float(h[i - 1]), float(l[i + 1]), times[i - 1], times[i + 1])
            # fill if later low goes into gap
            filled = bool((l[i + 2 :] <= z.high).any()) if i + 2 < n else False
            # full mitigate if low <= gap low
            full = bool((l[i + 2 :] <= z.low).any()) if i + 2 < n else False
            z.mitigated = full
            if not full:
                out.append(z)
        elif h[i + 1] < l[i - 1]:
            z = Zone("BEAR_FVG", i - 1, i + 1, float(h[i + 1]), float(l[i - 1]), times[i - 1], times[i + 1])
            full = bool((h[i + 2 :] >= z.high).any()) if i + 2 < n else False
            z.mitigated = full
            if not full:
                out.append(z)
    return out[-12:]


def _equal_levels(swings: list[Swing], atr: float, kind: str) -> list[float]:
    pts = [s.price for s in swings if s.kind == kind]
    if len(pts) < 2:
        return []
    tol = max(atr * 0.25, 0.003 * np.median(pts))
    used = [False] * len(pts)
    levels = []
    for i in range(len(pts)):
        if used[i]:
            continue
        cluster = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if abs(pts[j] - pts[i]) <= tol:
                cluster.append(pts[j])
                used[j] = True
        if len(cluster) >= 2:
            levels.append(float(np.mean(cluster)))
    return levels[-4:]


def _pick_long_poi(obs: list[Zone], fvgs: list[Zone], close: float) -> Optional[Zone]:
    """Chỉ lấy demand (bullish OB / FVG) nằm dưới hoặc sát giá — thị trường VN long-only."""
    cands = [z for z in obs + fvgs if z.kind in ("BULL_OB", "BULL_FVG") and not z.mitigated]
    if not cands:
        return None
    ranked = []
    for z in cands:
        below = z.high <= close * 1.012
        dist = close - z.mid
        score = 2.5 if "OB" in z.kind else 0.0
        if below and dist >= 0:
            score += 4
            score -= min(dist / max(close, 1) * 20, 3)  # gần giá hơn thì tốt hơn, miễn còn dưới
        else:
            score -= 6  # FVG đã chạy trên đầu — không đuổi
        ranked.append((score, z))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_s, best = ranked[0]
    return best if best_s > 0 else None


def _long_sl(entry_low: float, dealing_low: float, close: float, atr: float) -> float:
    buf = max(atr * 0.35, close * 0.004)
    sl = min(entry_low, dealing_low) - buf
    cap = min(entry_low, close) - buf
    if sl >= cap:
        sl = cap
    return float(sl)


def _long_targets(close: float, entry_mid: float, liq_up: list, dealing_high: float, atr: float) -> tuple[float, float]:
    """TP luôn phía trên giá hiện tại và trên entry — không bao giờ target short."""
    floor = max(close, entry_mid) * 1.005
    above = sorted(x for x in liq_up if x > floor)
    tp1 = above[0] if above else floor + max(atr * 1.2, floor * 0.03)
    tp2 = above[1] if len(above) > 1 else max(dealing_high, tp1 + max(atr, floor * 0.03))
    if tp2 <= tp1:
        tp2 = tp1 + max(atr, floor * 0.03)
    if tp1 <= close:
        tp1 = close + max(atr * 1.2, close * 0.03)
    if tp2 <= tp1:
        tp2 = tp1 + max(atr, close * 0.03)
    return float(tp1), float(tp2)


def analyze_smc(df: pd.DataFrame, swing_n: int = 3) -> Optional[SmcPlan]:
    if df is None or len(df) < 40:
        return None
    work = df.reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close"])
    if "time" not in work.columns:
        work["time"] = np.arange(len(work))
    atr = float(_atr(work).iloc[-1] or work["close"].iloc[-1] * 0.02)
    sw = find_swings(work, swing_n, swing_n)
    swings = _alternating_swings(sw)
    events, bias, structure = _structure_events(work, swings)
    obs = _order_blocks(work, events)
    fvgs = _fvgs(work)

    hs = [s for s in swings if s.kind == "H"]
    ls = [s for s in swings if s.kind == "L"]
    dealing_high = float(max(hs[-1].price, work["high"].iloc[-1])) if hs else float(work["high"].tail(40).max())
    dealing_low = float(min(ls[-1].price, work["low"].iloc[-1])) if ls else float(work["low"].tail(40).min())
    # range của 2 swing gần nhất cùng chiều structure
    if len(hs) and len(ls):
        dealing_high = float(max(hs[-1].price, work["high"].tail(5).max()))
        dealing_low = float(min(ls[-1].price, work["low"].tail(5).min()))
    if dealing_high <= dealing_low:
        dealing_high = float(work["high"].tail(40).max())
        dealing_low = float(work["low"].tail(40).min())
    eq = (dealing_high + dealing_low) / 2.0
    close = float(work["close"].iloc[-1])
    span = max(dealing_high - dealing_low, 1e-9)
    pd_pct = float(np.clip((close - dealing_low) / span, 0, 1))
    if pd_pct >= 0.55:
        pd_zone = "PREMIUM"
    elif pd_pct <= 0.45:
        pd_zone = "DISCOUNT"
    else:
        pd_zone = "EQ"

    liq_up = _equal_levels(swings, atr, "H")
    liq_dn = _equal_levels(swings, atr, "L")
    if hs:
        liq_up.append(hs[-1].price)
    if ls:
        liq_dn.append(ls[-1].price)
    liq_up = sorted(set(round(x, 2) for x in liq_up), reverse=True)[:4]
    liq_dn = sorted(set(round(x, 2) for x in liq_dn))[:4]

    last_event = events[-1].kind if events else "NONE"
    poi = _pick_long_poi(obs, fvgs, close)
    last_swing_low = float(ls[-1].price) if ls else float(work["low"].tail(20).min())
    invalidation = last_swing_low
    if invalidation >= close:
        invalidation = float(work["low"].tail(8).min())
    if invalidation >= close:
        invalidation = close - max(atr * 0.35, close * 0.004)

    poi_names = {
        "BULL_OB": "Bullish Order Block (demand — nến giảm cuối trước đẩy lên)",
        "BULL_FVG": "Bullish FVG / imbalance (demand)",
        "BEAR_OB": "Bearish Order Block",
        "BEAR_FVG": "Bearish FVG",
    }

    buf = max(atr * 0.35, close * 0.004)
    has_long_setup = False
    entry_low = entry_high = sl = tp1 = tp2 = rr = 0.0
    poi_label = "Không có demand sống dưới giá"

    def arm_long(z_low: float, z_high: float, label: str):
        nonlocal entry_low, entry_high, sl, tp1, tp2, rr, has_long_setup, poi_label
        entry_low, entry_high = float(z_low), float(z_high)
        if entry_high < entry_low:
            entry_low, entry_high = entry_high, entry_low
        # Long-only: zone không được nằm trên đầu giá
        if (entry_low + entry_high) / 2 > close * 1.01:
            has_long_setup = False
            return
        sl = _long_sl(entry_low, dealing_low, close, atr)
        mid = (entry_low + entry_high) / 2
        tp1, tp2 = _long_targets(close, mid, liq_up, dealing_high, atr)
        risk = mid - sl
        reward = tp1 - mid
        rr = float(reward / risk) if risk > 0 else 0.0
        has_long_setup = True
        poi_label = label

    if poi is not None:
        arm_long(poi.low, poi.high, poi_names.get(poi.kind, poi.kind))
    if not has_long_setup and bias in ("bull", "range") and pd_zone in ("DISCOUNT", "EQ"):
        band = span * 0.04
        arm_long(max(dealing_low, eq - band), min(eq + band * 0.3, close), "EQ / discount (chờ limit long)")

    in_zone = has_long_setup and entry_low * 0.997 <= close <= entry_high * 1.003
    near_zone = has_long_setup and abs(close - (entry_low + entry_high) / 2) / close <= 0.03

    tactics: list[str] = []
    thesis_bits: list[str] = [
        "VN long-only — không short",
        structure,
        f"Giá đang ở {pd_zone} ({pd_pct*100:.0f}% dealing range)",
    ]
    if last_event != "NONE":
        thesis_bits.append(f"Event gần nhất: {last_event.replace('_', ' ')}")
    if has_long_setup:
        thesis_bits.append(f"POI long: {poi_label}")
    else:
        thesis_bits.append("Chưa có vùng demand hợp lệ dưới giá")

    action: Action = "SKIP"

    if bias == "bull":
        if last_event == "CHOCH_DN":
            action = "EXIT"
            tactics += [
                "CHoCH xuống — structure tăng đã gãy. Thị trường VN không short: chỉ thoát long.",
                f"Nếu đang cầm: cắt quanh invalidation {invalidation:.2f} (swing low).",
                "Không mua dip cho đến khi có CHoCH lên + demand mới.",
            ]
            has_long_setup = False
        elif pd_zone == "PREMIUM":
            action = "WATCH"
            tactics += [
                "Bias tăng nhưng giá đang premium — không đuổi long.",
                "Chờ hồi về discount / bullish OB-FVG dưới giá rồi limit buy.",
            ]
            if has_long_setup:
                tactics.append(
                    f"Vùng chờ long {entry_low:.2f}–{entry_high:.2f}. SL {sl:.2f} (dưới POI). "
                    f"TP1 {tp1:.2f} / TP2 {tp2:.2f} (trên giá)."
                )
        elif has_long_setup and (in_zone or (pd_zone in ("DISCOUNT", "EQ") and near_zone)):
            action = "BUY"
            tactics += [
                "Limit long trong demand (OB/FVG), không market đuổi nến xanh.",
                f"Entry {entry_low:.2f}–{entry_high:.2f} (≤ giá {close:.2f}).",
                f"SL {sl:.2f} dưới low POI. TP1 {tp1:.2f} / TP2 {tp2:.2f} trên giá hiện tại. R:R {rr:.1f}.",
                "Nến đóng cửa xuyên low POI → setup fail, không gồng.",
            ]
        elif has_long_setup:
            action = "WATCH"
            tactics += [
                "Bias tăng, chưa vào demand.",
                f"Alert khi giá vào {entry_low:.2f}–{entry_high:.2f}. SL {sl:.2f}. TP1 {tp1:.2f} / TP2 {tp2:.2f}.",
            ]
        else:
            action = "WATCH"
            tactics += ["Bias tăng nhưng chưa có demand dưới giá — đứng ngoài, không đuổi."]

    elif bias == "bear":
        # Long-only: bear = không mở long. Nếu đang cầm → EXIT. Không bao giờ đưa entry/SL/TP short.
        if last_event == "CHOCH_UP":
            action = "WATCH"
            tactics += [
                "CHoCH lên trong downtrend — có thể sắp lật bias. Vẫn chưa short (VN không short).",
                "Chờ retest demand sau CHoCH rồi mới long.",
            ]
            if has_long_setup:
                tactics.append(
                    f"Vùng long sau retest: {entry_low:.2f}–{entry_high:.2f}. SL {sl:.2f}. TP {tp1:.2f}/{tp2:.2f}."
                )
        else:
            action = "EXIT"
            has_long_setup = False
            tactics += [
                "Bias giảm (LH/LL hoặc BOS xuống). Sàn VN chỉ long → không mở vị thế mới.",
                f"Nếu đang cầm long: thoát / hạ tỷ trọng. Invalidation {invalidation:.2f} (dưới giá).",
                "Không đặt entry trên giá, không lấy TP dưới giá (đó là logic short).",
            ]
            if pd_zone == "DISCOUNT":
                tactics.append(
                    "Giá đã discount trong downtrend: chỉ theo dõi demand, mua khi có CHoCH lên xác nhận — không bắt dao rơi."
                )

    else:  # range
        if pd_zone == "PREMIUM":
            action = "EXIT" if close > eq else "WATCH"
            tactics += [
                "Range + premium. Long-only: không mua, nếu đang cầm thì canh chốt về EQ.",
                f"Invalidation dưới range low {dealing_low:.2f}.",
            ]
            if action == "EXIT":
                has_long_setup = False
        elif has_long_setup and (in_zone or pd_zone == "DISCOUNT"):
            action = "BUY" if in_zone or near_zone else "WATCH"
            tactics += [
                "Range + discount — long cạnh dealing low, size nhỏ.",
                f"Entry {entry_low:.2f}–{entry_high:.2f}. SL {sl:.2f}. TP1 EQ/liquidity {tp1:.2f} (trên giá), TP2 {tp2:.2f}.",
            ]
        else:
            action = "WATCH"
            tactics += [f"Range giữa EQ {eq:.2f}. Đợi giá về discount mới long."]

    if action == "BUY" and (not has_long_setup or rr < 1.5 or sl >= close or tp1 <= close):
        action = "WATCH"
        tactics.append("Không đủ điều kiện long (R:R / SL-TP so với giá) — hạ WATCH.")
        if sl >= close or (has_long_setup and tp1 <= close):
            has_long_setup = False

    thesis = " · ".join(thesis_bits)
    return SmcPlan(
        bias=bias,
        last_event=last_event,
        structure=structure,
        dealing_low=round(dealing_low, 2),
        dealing_high=round(dealing_high, 2),
        eq=round(eq, 2),
        pd=pd_zone,
        pd_pct=round(pd_pct, 3),
        poi=poi if has_long_setup else None,
        poi_label=poi_label if has_long_setup else "Không mở long (bias giảm / không có demand)",
        entry_low=round(entry_low, 2) if has_long_setup else 0.0,
        entry_high=round(entry_high, 2) if has_long_setup else 0.0,
        sl=round(sl, 2) if has_long_setup else round(invalidation, 2),
        tp1=round(tp1, 2) if has_long_setup else 0.0,
        tp2=round(tp2, 2) if has_long_setup else 0.0,
        rr=round(rr, 2) if has_long_setup else 0.0,
        action=action,
        thesis=thesis,
        tactics=tactics,
        liquidity_up=liq_up,
        liquidity_dn=liq_dn,
        events=events[-8:],
        zones=(obs + fvgs)[-16:],
        swings=swings[-16:],
        has_long_setup=has_long_setup,
        invalidation=round(invalidation, 2),
    )


def plan_to_row(plan: SmcPlan) -> dict:
    if plan.has_long_setup:
        entry = f"{plan.entry_low:.2f}–{plan.entry_high:.2f}"
        sl = f"{plan.sl:.2f}"
        tp1 = f"{plan.tp1:.2f}"
        tp2 = f"{plan.tp2:.2f}"
        rr = f"1:{plan.rr:.1f}"
    else:
        entry = "—"
        sl = f"cắt long {plan.invalidation:.2f}" if plan.action == "EXIT" else "—"
        tp1 = "—"
        tp2 = "—"
        rr = "—"
    return {
        "S_SMC": plan.action,
        "SMC bias": plan.bias.upper(),
        "SMC event": plan.last_event.replace("_", " "),
        "SMC PD": plan.pd,
        "SMC POI": plan.poi_label,
        "SMC entry": entry,
        "SMC SL": sl,
        "SMC TP1": tp1,
        "SMC TP2": tp2,
        "SMC R:R": rr,
        "SMC thesis": plan.thesis,
        "_smc_action": plan.action,
        "_smc_long": plan.has_long_setup,
    }


def smc_plotly(df: pd.DataFrame, plan: SmcPlan, title: str = ""):
    import plotly.graph_objects as go

    work = df.tail(120).reset_index(drop=True)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=work["time"],
                open=work["open"],
                high=work["high"],
                low=work["low"],
                close=work["close"],
                name="OHLC",
                increasing_line_color="#3DDC97",
                decreasing_line_color="#FF6B6B",
            )
        ]
    )
    # swings
    for s in plan.swings:
        if s.i < len(df) - 120:
            continue
        fig.add_trace(
            go.Scatter(
                x=[s.time],
                y=[s.price],
                mode="markers+text",
                text=["H" if s.kind == "H" else "L"],
                textposition="top center" if s.kind == "H" else "bottom center",
                marker=dict(size=9, color="#F5C451" if s.kind == "H" else "#60A5FA"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    # events
    for e in plan.events:
        fig.add_vline(
            x=e.time,
            line_dash="dot",
            line_color="#A78BFA" if "CHOCH" in e.kind else "#94A3B8",
            annotation_text=e.kind.replace("_", " "),
            annotation_font_size=10,
        )
    # zones
    t0, t1 = work["time"].iloc[0], work["time"].iloc[-1]
    recent_i0 = max(0, len(work) - 90)
    for z in [zz for zz in plan.zones if zz.i1 >= recent_i0][-6:]:
        color = "rgba(61,220,151,0.18)" if "BULL" in z.kind else "rgba(255,107,107,0.18)"
        fig.add_hrect(
            y0=z.low,
            y1=z.high,
            line_width=0,
            fillcolor=color,
            annotation_text=z.kind.replace("_", " "),
            annotation_font_size=9,
        )
    fig.add_hline(y=plan.eq, line_dash="dash", line_color="#F5C451", annotation_text="EQ")
    if plan.has_long_setup:
        fig.add_hline(y=plan.sl, line_color="#FF6B6B", annotation_text="SL long")
        fig.add_hline(y=plan.tp1, line_color="#3DDC97", annotation_text="TP1")
        fig.add_hline(y=plan.tp2, line_dash="dash", line_color="#3DDC97", annotation_text="TP2")
        fig.add_hrect(
            y0=plan.entry_low,
            y1=plan.entry_high,
            line_width=1,
            line_color="#3DDC97",
            fillcolor="rgba(61,220,151,0.08)",
            annotation_text="ENTRY LONG",
        )
    else:
        fig.add_hline(
            y=plan.invalidation,
            line_color="#FF6B6B",
            annotation_text="cắt long / invalidation",
        )
    fig.update_layout(
        title=title or "SMC",
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0E1624",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=16, r=16, t=48, b=16),
        font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF5"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1E2A3F")
    fig.update_yaxes(gridcolor="#1E2A3F")
    return fig
