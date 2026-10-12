"""SMC khung lớn cho cổ phiếu VN (chỉ long) + đồ thị gộp mức giá.

Không đọc sóng vài phiên. Zigzag tối thiểu ~12% (kèm tuần/tháng):
  - Phá đỉnh (BOS lên): trend tăng còn, không đuổi, chờ hồi 50–78.6%.
  - Phá đáy (BOS xuống): không mua, đang cầm thì bán.
  - Đảo đáy (CHoCH lên): chỉ mua nếu còn sát đáy và có dòng tiền hoặc test đáy.
  - Đảo đỉnh (CHoCH xuống): không mua mới, đang cầm thì bán.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

Bias = Literal["bull", "bear", "range"]
Action = Literal["BUY", "WATCH", "EXIT", "SKIP"]
FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    v = tr.rolling(n, min_periods=max(5, n // 2)).mean().iloc[-1]
    if pd.isna(v) or v <= 0:
        return float(c.iloc[-1] * 0.02)
    return float(v)


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    x = df.copy()
    x["time"] = pd.to_datetime(x["time"])
    x = x.set_index("time").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in x.columns:
        agg["volume"] = "sum"
    out = x.resample(freq).agg(agg).dropna(subset=["close"]).reset_index()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


@dataclass
class Swing:
    i: int
    price: float
    kind: str
    time: object


@dataclass
class Event:
    i: int
    kind: str
    price: float
    time: object


@dataclass
class Zone:
    kind: str
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
    pd: str
    pd_pct: float
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
    has_long_setup: bool = False
    invalidation: float = 0.0
    plain: str = ""
    tf_label: str = ""
    fibs: list = field(default_factory=list)


def major_zigzag(df: pd.DataFrame, min_pct: float) -> list[Swing]:
    if df is None or len(df) < 10:
        return []
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    times = df["time"].tolist()
    n = len(df)
    pivots: list[Swing] = []
    direction = 1 if hi[min(5, n - 1)] >= hi[0] else -1
    ext_i, ext_p = 0, hi[0] if direction == 1 else lo[0]
    for i in range(1, n):
        if direction == 1:
            if hi[i] >= ext_p:
                ext_p, ext_i = hi[i], i
            elif ext_p > 0 and (ext_p - lo[i]) / ext_p >= min_pct:
                pivots.append(Swing(ext_i, float(ext_p), "H", times[ext_i]))
                direction = -1
                ext_p, ext_i = lo[i], i
        else:
            if lo[i] <= ext_p:
                ext_p, ext_i = lo[i], i
            elif ext_p > 0 and (hi[i] - ext_p) / ext_p >= min_pct:
                pivots.append(Swing(ext_i, float(ext_p), "L", times[ext_i]))
                direction = 1
                ext_p, ext_i = hi[i], i
    kind = "H" if direction == 1 else "L"
    if not pivots or pivots[-1].i != ext_i:
        pivots.append(Swing(ext_i, float(ext_p), kind, times[ext_i]))
    alt: list[Swing] = []
    for s in pivots:
        if not alt or s.kind != alt[-1].kind:
            alt.append(s)
            continue
        if s.kind == "H" and s.price >= alt[-1].price:
            alt[-1] = s
        elif s.kind == "L" and s.price <= alt[-1].price:
            alt[-1] = s
    return alt


def _space_swings(swings: list[Swing], min_bars: int, min_pct: float) -> list[Swing]:
    out: list[Swing] = []
    for s in swings:
        if not out:
            out.append(s)
            continue
        if s.i - out[-1].i < min_bars:
            move = abs(s.price - out[-1].price) / max(out[-1].price, 1e-9)
            if s.kind == out[-1].kind:
                if (s.kind == "H" and s.price >= out[-1].price) or (s.kind == "L" and s.price <= out[-1].price):
                    out[-1] = s
                continue
            if move < min_pct:
                continue
        if s.kind == out[-1].kind:
            if (s.kind == "H" and s.price >= out[-1].price) or (s.kind == "L" and s.price <= out[-1].price):
                out[-1] = s
            continue
        out.append(s)
    return out


def _events_from_swings(df: pd.DataFrame, swings: list[Swing]) -> tuple[list[Event], Bias]:
    events: list[Event] = []
    if len(swings) < 3:
        return events, "range"
    close = df["close"].to_numpy(float)
    times = df["time"].tolist()
    trend: Bias = "range"
    last_h = None
    last_l = None
    for s in swings:
        if s.kind == "H":
            prev = last_h
            last_h = s
            if prev is None:
                continue
            for j in range(prev.i + 1, min(len(df), s.i + 1)):
                if close[j] > prev.price:
                    kind = "BOS_UP" if trend == "bull" else "CHOCH_UP"
                    events.append(Event(j, kind, float(prev.price), times[j]))
                    trend = "bull"
                    break
        else:
            prev = last_l
            last_l = s
            if prev is None:
                continue
            for j in range(prev.i + 1, min(len(df), s.i + 1)):
                if close[j] < prev.price:
                    kind = "BOS_DN" if trend == "bear" else "CHOCH_DN"
                    events.append(Event(j, kind, float(prev.price), times[j]))
                    trend = "bear"
                    break
    compact: list[Event] = []
    for e in events:
        if compact and compact[-1].kind == e.kind and abs(e.i - compact[-1].i) < 8:
            compact[-1] = e
        else:
            compact.append(e)
    if compact:
        trend = "bull" if compact[-1].kind.endswith("UP") else "bear"
    return compact, trend


def _tf_note(df: pd.DataFrame, freq: str, min_pct: float, name: str) -> str:
    try:
        w = _resample(df, freq)
    except Exception:
        return f"{name}: —"
    if len(w) < 8:
        return f"{name}: ít nến"
    sw = _space_swings(major_zigzag(w, min_pct), min_bars=2, min_pct=min_pct)
    ev, bias = _events_from_swings(w, sw)
    bias_vi = {"bull": "tăng", "bear": "giảm", "range": "đi ngang"}.get(bias, "—")
    if not ev:
        return f"{name}: {bias_vi}"
    last = ev[-1]
    fresh = last.i >= len(w) - 6
    label = {"BOS_UP": "phá đỉnh", "BOS_DN": "phá đáy", "CHOCH_UP": "đảo đáy", "CHOCH_DN": "đảo đỉnh"}.get(last.kind, last.kind)
    return f"{name}: {bias_vi}, {label} ({'mới' if fresh else 'cũ'})"


def _flow_and_retest(df: pd.DataFrame, major_low: float, hint: Optional[dict]) -> tuple[bool, bool, str]:
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    tail = slice(-6, None)
    up = float(v[tail][c[tail] >= o[tail]].sum()) if len(c) >= 6 else 0.0
    dn = float(v[tail][c[tail] < o[tail]].sum()) if len(c) >= 6 else 0.0
    price_up = bool(len(c) >= 6 and c[-1] >= c[-6])
    flow = up > dn * 1.1 and price_up
    why = []
    if flow:
        why.append("khối lượng nghiêng mua 6 phiên")
    if hint:
        try:
            cmf = float(str(hint.get("CMF", "0")).replace(",", ""))
        except ValueError:
            cmf = 0.0
        vsa = str(hint.get("VSA", "") or "")
        vol_s = str(hint.get("Vol", "") or "").lower().replace("x", "")
        try:
            vr = float(vol_s) if vol_s else 0.0
        except ValueError:
            vr = 0.0
        if cmf > 0.02:
            flow = True
            why.append(f"CMF {cmf:.2f} dương")
        if any(k in vsa for k in ("NỔ VOL", "RÚT CHÂN", "CẠN VOL")):
            flow = True
            why.append(f"VSA {vsa}")
        if vr >= 1.3 and price_up:
            flow = True
            why.append(f"vol {vr:.1f}x")
        if "PHÂN PHỐI" in vsa or "CẠN CẦU" in vsa:
            flow = False
            why.append("VSA phân phối/cạn cầu — bỏ dòng tiền")
    window_lo = lo[-15:] if len(lo) >= 15 else lo
    window_c = c[-15:] if len(c) >= 15 else c
    touched = float(np.min(window_lo)) <= major_low * 1.03
    held = float(window_c[-1]) >= major_low * 0.995 and float(np.min(window_c)) >= major_low * 0.97
    retest = bool(touched and held)
    if retest:
        why.append("test đáy lớn rồi giữ được")
    return flow, retest, ", ".join(why) if why else "chưa có dòng tiền"


def _fib_prices(top: float, bot: float) -> list[tuple[float, float]]:
    span = top - bot
    return [(r, top - span * r) for r in FIB_RATIOS]


def _plain(kind: str) -> str:
    return {
        "BOS_UP": "Phá đỉnh (BOS): giá vượt đỉnh lớn cũ — xu hướng tăng còn sống. Đừng mua đuổi. Chờ hồi về khoảng 50–78.6% của nhịp tăng.",
        "BOS_DN": "Phá đáy (BOS): giá thủng đáy lớn cũ — đà giảm chưa xong. Không mua. Nếu đang cầm thì bán.",
        "CHOCH_UP": "Đảo đáy (CHoCH): giá phá đỉnh của nhịp giảm — đáy lớn có thể đã hình thành. Chỉ mua nếu giá còn sát đáy và có dòng tiền hoặc test đáy thành công.",
        "CHOCH_DN": "Đảo đỉnh (CHoCH): giá thủng đáy của nhịp tăng — đỉnh lớn có thể đã hình thành. Không mua mới. Nếu đang cầm thì bán.",
        "NONE": "Chưa có phá đỉnh/đáy lớn mới. Đứng ngoài, chỉ canh vùng gần đáy khung ngày/tuần.",
    }.get(kind, kind)


def analyze_smc(df: pd.DataFrame, swing_n: int = 3, flow_hint: Optional[dict] = None) -> Optional[SmcPlan]:
    del swing_n
    if df is None or len(df) < 40:
        return None
    work = df.reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = 0.0
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
    work = work.dropna(subset=["open", "high", "low", "close"])
    if "time" not in work.columns:
        work["time"] = np.arange(len(work))
    if len(work) < 40:
        return None

    atr = _atr(work)
    close = float(work["close"].iloc[-1])
    atr_pct = atr / max(close, 1e-9)
    min_pct = float(np.clip(max(0.12, atr_pct * 4.0), 0.12, 0.18))
    swings = _space_swings(major_zigzag(work, min_pct), min_bars=10, min_pct=min_pct)
    if len(swings) < 4:
        swings = _space_swings(major_zigzag(work, 0.08), min_bars=8, min_pct=0.08)
    events, bias = _events_from_swings(work, swings)
    last_event = events[-1].kind if events else "NONE"
    fresh = bool(events) and events[-1].i >= len(work) - 25
    if fresh and last_event in ("BOS_DN", "CHOCH_DN") and close > events[-1].price * 1.015:
        fresh = False
    if fresh and last_event in ("BOS_UP", "CHOCH_UP") and close < events[-1].price * 0.985:
        fresh = False

    hs = [s for s in swings if s.kind == "H"]
    ls = [s for s in swings if s.kind == "L"]
    if not hs or not ls:
        return None

    a, b = swings[-2], swings[-1]
    leg_top = max(a.price, b.price)
    leg_bot = min(a.price, b.price)
    if leg_top <= leg_bot:
        leg_top = float(work["high"].tail(60).max())
        leg_bot = float(work["low"].tail(60).min())
    span = max(leg_top - leg_bot, 1e-9)
    fibs = _fib_prices(leg_top, leg_bot)
    fib_map = {r: p for r, p in fibs}
    depth = float(np.clip((leg_top - close) / span, 0, 1))
    eq = fib_map[0.5]
    pd_zone = "DISCOUNT" if depth >= 0.55 else ("PREMIUM" if depth <= 0.45 else "EQ")

    major_low = float(ls[-1].price)
    major_high = float(hs[-1].price)
    flow, retest, flow_why = _flow_and_retest(work, major_low, flow_hint)
    near_low = depth >= 0.62 or close <= major_low * 1.06
    in_golden = 0.50 <= depth <= 0.86

    week = _tf_note(work, "W-FRI", max(0.08, min_pct), "Tuần")
    month = _tf_note(work, "ME", 0.12, "Tháng")
    tf_label = f"{week} · {month}"
    higher_tf_blocking = ("phá đáy (mới)" in week) or ("đảo đỉnh (mới)" in week) or ("phá đáy (mới)" in month)

    zone_low = min(fib_map[1.0], fib_map[0.618])
    zone_high = max(fib_map[1.0], fib_map[0.618])
    demand = Zone("BULL_OB", int(ls[-1].i), int(ls[-1].i), float(zone_low), float(zone_high), ls[-1].time, ls[-1].time)

    buf = max(atr * 0.4, close * 0.006)
    sl = min(major_low, zone_low) - buf
    if sl >= close:
        sl = close - buf
    tp1 = fib_map[0.382] if fib_map[0.382] > close * 1.01 else fib_map[0.236]
    if tp1 <= close:
        tp1 = max(major_high, close * 1.04)
    tp2 = major_high if major_high > tp1 else tp1 + max(atr * 2, span * 0.25)
    if tp2 <= tp1:
        tp2 = tp1 * 1.04
    entry_mid = (zone_low + zone_high) / 2
    has_long = entry_mid <= close * 1.015 and sl < min(close, zone_low)
    if zone_high > close * 1.01:
        zone_high = min(zone_high, close)
        zone_low = min(zone_low, zone_high)
        has_long = zone_low < close and sl < zone_low
    risk = max(entry_mid - sl, 1e-9)
    reward = tp1 - entry_mid
    rr = float(reward / risk) if reward > 0 else 0.0
    invalidation = sl
    structure = {
        "bull": "Sóng lớn đang tăng",
        "bear": "Sóng lớn đang giảm",
        "range": "Sóng lớn đi ngang",
    }[bias]
    plain = _plain(last_event) if fresh else _plain("NONE")
    if not fresh and last_event != "NONE":
        plain += " Lần phá mức trước đã cũ, không dùng để vào lệnh nữa."

    action: Action = "WATCH"
    tactics: list[str] = []
    if last_event == "BOS_DN" and fresh:
        action, has_long = "EXIT", False
        tactics = [
            "Phá đáy lớn: không mua. Đang cầm thì bán.",
            f"Mốc cắt long: {major_low:.2f}.",
            "Chờ đảo đáy (CHoCH lên) rồi mới nghĩ đến mua.",
        ]
    elif last_event == "CHOCH_DN" and fresh:
        action, has_long = "EXIT", False
        tactics = [
            "Đảo đỉnh: chuỗi tăng gãy. Không mua mới. Đang cầm thì bán.",
            f"Đáy nhịp vừa mất quanh {major_low:.2f}.",
        ]
    elif last_event == "CHOCH_UP" and fresh:
        if near_low and (flow or retest) and not higher_tf_blocking and rr >= 1.5 and has_long:
            action = "BUY"
            tactics = [
                "Đảo đáy + còn sát đáy + có dòng tiền hoặc test đáy.",
                f"Mua {zone_low:.2f}–{zone_high:.2f}. Cắt lỗ {sl:.2f}. Chốt {tp1:.2f} rồi {tp2:.2f}.",
                f"Xác nhận: {flow_why}.",
            ]
        else:
            action = "WATCH"
            has_long = has_long and not higher_tf_blocking
            why_no = []
            if not near_low:
                why_no.append("giá đã chạy xa đáy")
            if not (flow or retest):
                why_no.append("chưa có dòng tiền / chưa test đáy")
            if higher_tf_blocking:
                why_no.append("tuần hoặc tháng vừa phá đáy / đảo đỉnh")
            if rr < 1.5:
                why_no.append("lãi/lỗ chưa đủ")
            tactics = [
                "Có đảo đáy nhưng chưa đủ điều kiện mua: " + (", ".join(why_no) or "chờ thêm") + ".",
                "Không mua đuổi. Chỉ mua khi giá còn gần đáy và có dòng tiền.",
            ]
            if has_long:
                tactics.append(f"Vùng chờ: {zone_low:.2f}–{zone_high:.2f}. Cắt lỗ {sl:.2f}. Mục tiêu {tp1:.2f} / {tp2:.2f}.")
    elif last_event == "BOS_UP" and fresh:
        if in_golden and (flow or retest) and not higher_tf_blocking and rr >= 1.5 and has_long and close <= zone_high * 1.02:
            action = "BUY"
            tactics = [
                "Đã phá đỉnh và nay hồi về 50–78.6%, có dòng tiền hoặc giữ đáy nhịp.",
                f"Mua {zone_low:.2f}–{zone_high:.2f}. Cắt lỗ {sl:.2f}. Mục tiêu {tp1:.2f} / {tp2:.2f}.",
                f"Xác nhận: {flow_why}.",
            ]
        else:
            action = "WATCH"
            tactics = [
                "Phá đỉnh: xu hướng tăng còn, nhưng không mua đuổi.",
                f"Chờ hồi về {fib_map[0.5]:.2f}–{fib_map[0.786]:.2f}. Cắt lỗ dưới {sl:.2f}. Mục tiêu {tp1:.2f} rồi {tp2:.2f}.",
            ]
            if higher_tf_blocking:
                tactics.append("Khung tuần/tháng đang xấu — đứng ngoài dù ngày vừa phá đỉnh.")
                has_long = False
    elif higher_tf_blocking:
        action, has_long = "EXIT", False
        plain = "Khung tuần hoặc tháng vừa phá đáy hoặc đảo đỉnh. Không mua. Nếu đang cầm thì bán."
        tactics = [
            "Tín hiệu bán nằm ở khung tuần/tháng, không phải vài phiên ngày.",
            f"Mốc cắt tham chiếu: {major_low:.2f}.",
            tf_label,
        ]
    elif bias == "bear":
        action, has_long = "WATCH", False
        tactics = [
            "Sóng lớn vẫn giảm nhưng chưa thủng đáy mới. Không mua.",
            "Chỉ bán khi có phá đáy hoặc đảo đỉnh. Chưa tới thì đứng ngoài.",
        ]
    elif bias == "bull" and in_golden and (flow or retest) and rr >= 1.5 and has_long:
        action = "WATCH"
        tactics = [
            "Giá gần đáy một nhịp tăng và có dòng tiền, nhưng chưa phá đỉnh/đảo đáy mới.",
            "Chưa mua một mình — cần chiến thuật kia cùng chiều hoặc một lần phá đỉnh rõ.",
            f"Vùng nếu được xác nhận: {zone_low:.2f}–{zone_high:.2f}. Cắt lỗ {sl:.2f}. Mục tiêu {tp1:.2f} / {tp2:.2f}.",
            f"Xác nhận giá: {flow_why}.",
        ]
    elif pd_zone == "PREMIUM" or depth < 0.45:
        action = "WATCH"
        tactics = [
            "Giá đang gần đỉnh nhịp. Không mua đuổi.",
            f"Chờ về {fib_map[0.5]:.2f}–{fib_map[0.786]:.2f} (50–78.6% từ đỉnh về đáy) mới xem lại.",
        ]
    else:
        action = "WATCH"
        tactics = [
            "Đứng ngoài. Chưa phá đỉnh mới, cũng chưa có đáy được dòng tiền xác nhận.",
            f"Vùng đáng mua nếu sau này có xác nhận: {zone_low:.2f}–{zone_high:.2f}.",
        ]

    if action == "BUY" and (not has_long or sl >= close or tp1 <= close):
        action = "WATCH"
        tactics.append("Mức cắt lỗ/mục tiêu không hợp lệ so với giá hiện tại — hạ xuống chờ.")
        has_long = False

    thesis = " · ".join([
        plain,
        structure,
        f"Giá ở {depth * 100:.0f}% từ đỉnh nhịp về đáy (0%=đỉnh, 100%=đáy)",
        tf_label,
        f"Dòng tiền: {flow_why}",
    ])
    poi_label = (
        f"Vùng gần đáy nhịp (fib 61.8–100%): {zone_low:.2f}–{zone_high:.2f}"
        if has_long else "Không mở long"
    )
    return SmcPlan(
        bias=bias,
        last_event=last_event if fresh else "NONE",
        structure=structure,
        dealing_low=round(leg_bot, 2),
        dealing_high=round(leg_top, 2),
        eq=round(eq, 2),
        pd=pd_zone,
        pd_pct=round(depth, 3),
        poi=demand if has_long else None,
        poi_label=poi_label,
        entry_low=round(zone_low, 2) if has_long else 0.0,
        entry_high=round(min(zone_high, close), 2) if has_long else 0.0,
        sl=round(sl, 2) if has_long else round(min(invalidation, close - buf), 2),
        tp1=round(tp1, 2) if has_long else 0.0,
        tp2=round(tp2, 2) if has_long else 0.0,
        rr=round(rr, 2) if has_long else 0.0,
        action=action,
        thesis=thesis,
        tactics=tactics,
        liquidity_up=[round(major_high, 2), round(fib_map[0.236], 2)],
        liquidity_dn=[round(major_low, 2), round(leg_bot, 2)],
        events=events[-6:],
        zones=[demand],
        swings=swings[-12:],
        has_long_setup=has_long,
        invalidation=round(min(major_low, close - buf), 2),
        plain=plain,
        tf_label=tf_label,
        fibs=[(r, round(p, 2)) for r, p in fibs],
    )


def plan_to_row(plan: SmcPlan) -> dict:
    if plan.has_long_setup:
        entry = f"{plan.entry_low:.2f}–{plan.entry_high:.2f}"
        sl, tp1, tp2, rr = f"{plan.sl:.2f}", f"{plan.tp1:.2f}", f"{plan.tp2:.2f}", f"1:{plan.rr:.1f}"
    else:
        entry = "—"
        sl = f"cắt long {plan.invalidation:.2f}" if plan.action == "EXIT" else "—"
        tp1 = tp2 = rr = "—"
    ev = {
        "BOS_UP": "Phá đỉnh",
        "BOS_DN": "Phá đáy",
        "CHOCH_UP": "Đảo đáy",
        "CHOCH_DN": "Đảo đỉnh",
        "NONE": "Chưa phá mức",
    }.get(plan.last_event, plan.last_event)
    return {
        "S_SMC": plan.action,
        "SMC bias": plan.bias.upper(),
        "SMC event": ev,
        "SMC PD": plan.pd,
        "SMC POI": plan.poi_label,
        "SMC nói": plan.plain,
        "SMC TF": plan.tf_label,
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
    return trade_figure(df, plan=plan, window="1Y", show_levels=False, title=title or "SMC khung lớn")


def _fnum(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().replace(",", "")
    if text in {"", "—", "-", "None", "nan"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def chart_view(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """1D/3M/6M/1Y/ALL = nến ngày. 1W/1M = gộp từ nến ngày."""
    work = df.copy()
    work["time"] = pd.to_datetime(work["time"])
    work = work.sort_values("time").reset_index(drop=True)
    if window == "1W":
        return _resample(work, "W-FRI")
    if window == "1M":
        return _resample(work, "ME")
    bars = {"1D": 30, "3M": 66, "6M": 126, "1Y": 252}.get(window)
    if bars:
        return work.tail(bars).reset_index(drop=True)
    return work


def trade_figure(
    df: pd.DataFrame,
    plan: Optional[SmcPlan] = None,
    levels: Optional[dict] = None,
    window: str = "6M",
    show_ema20: bool = True,
    show_ema50: bool = True,
    show_volume: bool = True,
    show_levels: bool = True,
    show_smc: bool = True,
    title: str = "",
):
    """Nến + EMA + volume + entry/SL/TP/kháng cự + SMC sóng lớn."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    daily = df.copy()
    daily["time"] = pd.to_datetime(daily["time"])
    view = chart_view(daily, window)
    if view.empty:
        fig = go.Figure()
        fig.update_layout(title="Không có nến")
        return fig

    close = float(view["close"].iloc[-1])
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22] if show_volume else [1],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=view["time"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="Giá",
            increasing_line_color="#3DDC97",
            increasing_fillcolor="#3DDC97",
            decreasing_line_color="#FF6B6B",
            decreasing_fillcolor="#FF6B6B",
        ),
        row=1, col=1,
    )
    if show_ema20 and len(view) >= 5:
        ema20 = view["close"].ewm(span=20, adjust=False).mean()
        fig.add_trace(
            go.Scatter(x=view["time"], y=ema20, name="EMA20", mode="lines", line=dict(color="#F5A524", width=1.6)),
            row=1, col=1,
        )
    if show_ema50 and len(view) >= 10:
        ema50 = view["close"].ewm(span=50, adjust=False).mean()
        fig.add_trace(
            go.Scatter(x=view["time"], y=ema50, name="EMA50", mode="lines", line=dict(color="#22D3EE", width=1.6)),
            row=1, col=1,
        )
    if show_volume:
        colors = ["#3DDC97" if c >= o else "#FF6B6B" for c, o in zip(view["close"], view["open"])]
        fig.add_trace(
            go.Bar(x=view["time"], y=view["volume"], name="Volume", marker_color=colors, marker_line_width=0),
            row=2, col=1,
        )

    y_lo = float(view["low"].min())
    y_hi = float(view["high"].max())
    drawn: list[float] = []

    def _near(price: float) -> bool:
        return any(abs(price - prev) / max(price, 1e-9) < 0.004 for prev in drawn)

    def _level(price, color, dash, text, width=1.1):
        if price is None:
            return
        price = float(price)
        if price <= 0 or _near(price):
            return
        if abs(price - close) / close > 0.22 and not (y_lo * 0.98 <= price <= y_hi * 1.02):
            return
        drawn.append(price)
        fig.add_hline(
            y=price,
            line_color=color,
            line_dash=dash,
            line_width=width,
            annotation_text=f"{text}  {price:.2f}",
            annotation_position="right",
            annotation_font_size=11,
            annotation_font_color=color,
            row=1, col=1,
        )

    levels = levels or {}
    if show_levels:
        resist = float(daily["high"].tail(20).max())
        _level(resist, "#A78BFA", "dash", "Resistance · 20D high")
        _level(_fnum(levels.get("Entry")), "#60A5FA", "dot", "Entry midpoint", 1.4)
        _level(_fnum(levels.get("SL T+")), "#FF6B6B", "dash", "Stop loss")
        _level(_fnum(levels.get("SL Hold")), "#FB7185", "dot", "SL Hold")
        _level(_fnum(levels.get("T+ T1")), "#3DDC97", "dash", "Take profit")
        _level(_fnum(levels.get("T+ T2")), "#86EFAC", "dot", "Take profit 2")
        _level(_fnum(levels.get("S1")), "#2DD4BF", "dash", "Support S1")
        _level(_fnum(levels.get("R1")), "#C4B5FD", "dot", "Resistance R1")

    if show_smc and plan is not None:
        fib_map = {r: p for r, p in plan.fibs}
        if 0.0 in fib_map and 0.236 in fib_map:
            top, bot = max(fib_map[0.0], fib_map[0.236]), min(fib_map[0.0], fib_map[0.236])
            if abs(((top + bot) / 2) - close) / close < 0.35:
                fig.add_hrect(y0=bot, y1=top, fillcolor="rgba(15,80,60,0.28)", line_width=0, row=1, col=1)
        if 0.618 in fib_map and 1.0 in fib_map:
            top, bot = max(fib_map[0.618], fib_map[1.0]), min(fib_map[0.618], fib_map[1.0])
            fig.add_hrect(
                y0=bot, y1=top,
                fillcolor="rgba(61,220,151,0.12)",
                line=dict(color="rgba(45,212,191,0.55)", width=1, dash="dash"),
                row=1, col=1,
            )
        if plan.has_long_setup and plan.entry_high > plan.entry_low:
            _level((plan.entry_low + plan.entry_high) / 2, "#93C5FD", "dot", "SMC entry")
            _level(plan.sl, "#FCA5A5", "dash", "SMC stop")
            _level(plan.tp1, "#4ADE80", "dash", "SMC TP1")
            _level(plan.tp2, "#86EFAC", "dot", "SMC TP2")
        elif plan.invalidation:
            _level(plan.invalidation, "#FF6B6B", "dash", "Cắt long")
        for ratio, price in plan.fibs:
            if ratio in (0.0, 1.0):
                continue
            label = "EQ" if ratio == 0.5 else f"{ratio:.3g}"
            _level(price, "#64748B" if ratio not in (0.618, 0.786) else "#F5C451", "dot", label, 1)

        t0 = pd.to_datetime(view["time"].iloc[0])
        t1 = pd.to_datetime(view["time"].iloc[-1])
        visible, prev = [], None
        for swing in plan.swings:
            when = pd.to_datetime(swing.time)
            if when < t0:
                prev = swing
            elif when <= t1:
                visible.append(swing)
        if prev is not None:
            visible = [prev] + visible
        if len(visible) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=[s.time for s in visible],
                    y=[s.price for s in visible],
                    mode="lines+markers",
                    name="SMC",
                    line=dict(color="#E5E7EB", width=1.4),
                    marker=dict(
                        size=8,
                        color=["#FF6B6B" if s.kind == "H" else "#3DDC97" for s in visible],
                        line=dict(width=0),
                    ),
                    hovertext=["Đỉnh lớn" if s.kind == "H" else "Đáy lớn" for s in visible],
                ),
                row=1, col=1,
            )
        if plan.events:
            event = plan.events[-1]
            when = pd.to_datetime(event.time)
            if t0 <= when <= t1:
                fig.add_vline(
                    x=event.time, line_dash="dash", line_color="#A78BFA",
                    annotation_text={"BOS_UP": "Phá đỉnh", "BOS_DN": "Phá đáy", "CHOCH_UP": "Đảo đáy", "CHOCH_DN": "Đảo đỉnh"}.get(event.kind, event.kind),
                    row=1, col=1,
                )

    pad = max((y_hi - y_lo) * 0.08, close * 0.01)
    extra = [p for p in drawn if y_lo - pad <= p <= y_hi + pad * 4]
    if extra:
        y_lo = min(y_lo, min(extra))
        y_hi = max(y_hi, max(extra))
    fig.update_yaxes(range=[y_lo - pad, y_hi + pad], row=1, col=1, gridcolor="#1E2A3F", tickfont=dict(size=11), side="right")
    if show_volume:
        fig.update_yaxes(title_text="VOLUME", row=2, col=1, showticklabels=False, gridcolor="#1E2A3F")
        fig.update_xaxes(gridcolor="#1E2A3F", row=2, col=1)
    fig.update_xaxes(gridcolor="#1E2A3F", rangeslider_visible=False, row=1, col=1)

    head = title or "Đồ thị"
    if show_smc and plan is not None:
        head = f"{title} · {window} · D {plan.bias.upper()} {plan.last_event} · {plan.pd} {plan.pd_pct:.0%}"
    fig.update_layout(
        title=dict(text=head, x=0, font=dict(size=13, color="#CBD5E1")),
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0E1624",
        height=720 if show_volume else 620,
        margin=dict(l=8, r=64, t=48, b=8),
        font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF5"),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=11)),
        hovermode="x unified",
        bargap=0.25,
    )
    return fig
