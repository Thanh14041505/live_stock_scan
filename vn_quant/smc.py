"""SMC khung lớn cho cổ phiếu VN (chỉ long).

Không đọc sóng 3 nến. Zigzag tối thiểu ~6–10% (và khung tuần/tháng) giống
các chart Super SMC: đỉnh/đáy lớn, BOS, CHoCH, Fibonacci 0–1 của nhịp vừa rồi.

Khuyến nghị viết tiếng thường:
  - Phá đỉnh (BOS lên)  → trend tăng còn, đừng đuổi, chờ hồi 50–78.6%.
  - Phá đáy (BOS xuống) → giảm chưa xong, không mua, đang cầm thì bán.
  - Đảo đáy (CHoCH lên) → đáy có thể xong; chỉ mua nếu còn gần đáy và có dòng tiền
    hoặc test đáy thành công.
  - Đảo đỉnh (CHoCH xuống) → đỉnh có thể xong, không mua, đang cầm thì bán.
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
    o = (
        x.resample(freq)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .dropna(subset=["close"])
        .reset_index()
    )
    return o


@dataclass
class Swing:
    i: int
    price: float
    kind: str  # H | L
    time: object


@dataclass
class Event:
    i: int
    kind: str  # BOS_UP BOS_DN CHOCH_UP CHOCH_DN
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
    fibs: list = field(default_factory=list)  # (ratio, price)


def major_zigzag(df: pd.DataFrame, min_pct: float) -> list[Swing]:
    """Đỉnh/đáy chỉ khi giá đảo chiều ít nhất min_pct. Bỏ nhiễu vài phiên."""
    if df is None or len(df) < 10:
        return []
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    times = df["time"].tolist()
    n = len(df)
    pivots: list[Swing] = []
    # seed
    direction = 1 if hi[5] >= hi[0] else -1
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
    # extreme đang chạy (chưa đảo đủ %) vẫn là đỉnh/đáy tạm để đo fib
    kind = "H" if direction == 1 else "L"
    if not pivots or pivots[-1].i != ext_i:
        pivots.append(Swing(ext_i, float(ext_p), kind, times[ext_i]))
    # bỏ đỉnh/đáy trùng loại liên tiếp — giữ cực trị
    alt: list[Swing] = []
    for s in pivots:
        if not alt:
            alt.append(s)
            continue
        if s.kind == alt[-1].kind:
            if s.kind == "H" and s.price >= alt[-1].price:
                alt[-1] = s
            elif s.kind == "L" and s.price <= alt[-1].price:
                alt[-1] = s
        else:
            alt.append(s)
    return alt


def _space_swings(swings: list[Swing], min_bars: int, min_pct: float) -> list[Swing]:
    """Gộp pivot quá gần — chart mẫu chỉ nối đỉnh/đáy cách nhau nhiều phiên."""
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
            if s.kind == "H" and s.price >= out[-1].price:
                out[-1] = s
            elif s.kind == "L" and s.price <= out[-1].price:
                out[-1] = s
            continue
        out.append(s)
    return out


def _events_from_swings(df: pd.DataFrame, swings: list[Swing]) -> tuple[list[Event], Bias]:
    """BOS = phá cùng chiều trend. CHoCH = phá ngược chiều (đảo)."""
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
            # tìm phiên đóng cửa vượt đỉnh cũ, sau đỉnh cũ và trước/đến đỉnh mới
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
        last = compact[-1].kind
        trend = "bull" if last.endswith("UP") else "bear"
    return compact, trend


def _tf_note(df: pd.DataFrame, freq: str, min_pct: float, name: str) -> str:
    try:
        w = _resample(df, freq)
    except Exception:
        return f"{name}: —"
    if len(w) < 8:
        return f"{name}: ít nến"
    sw = major_zigzag(w, min_pct)
    ev, bias = _events_from_swings(w, sw)
    bias_vi = {"bull": "tăng", "bear": "giảm", "range": "đi ngang"}.get(bias, "—")
    if not ev:
        return f"{name}: {bias_vi}"
    last = ev[-1]
    fresh = last.i >= len(w) - 6
    label = {
        "BOS_UP": "phá đỉnh",
        "BOS_DN": "phá đáy",
        "CHOCH_UP": "đảo đáy",
        "CHOCH_DN": "đảo đỉnh",
    }.get(last.kind, last.kind)
    age = "mới" if fresh else "cũ"
    return f"{name}: {bias_vi}, {label} ({age})"


def _flow_and_retest(df: pd.DataFrame, major_low: float, hint: Optional[dict]) -> tuple[bool, bool, str]:
    """Dòng tiền vào, hoặc test đáy lớn rồi giữ được."""
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    tail = slice(-6, None)
    up = float(v[tail][c[tail] >= o[tail]].sum()) if len(c) >= 6 else 0.0
    dn = float(v[tail][c[tail] < o[tail]].sum()) if len(c) >= 6 else 0.0
    price_up = c[-1] >= c[-6] if len(c) >= 6 else False
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
    # test đáy: 15 phiên chạm sát đáy lớn nhưng đóng cửa vẫn trên đáy
    window_lo = lo[-15:] if len(lo) >= 15 else lo
    window_c = c[-15:] if len(c) >= 15 else c
    touched = float(window_lo.min()) <= major_low * 1.03
    held = float(window_c[-1]) >= major_low * 0.995 and float(window_c.min()) >= major_low * 0.97
    retest = bool(touched and held)
    if retest:
        why.append("test đáy lớn rồi giữ được (không đóng cửa thủng)")
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
    del swing_n  # giữ signature cũ; sóng lớn không dùng fractal 3
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
    # Sàn VN: chỉ lấy nhịp đảo chiều khoảng 12% trở lên — bỏ sóng vài phiên.
    min_pct = float(np.clip(max(0.12, atr_pct * 4.0), 0.12, 0.18))
    swings = major_zigzag(work, min_pct)
    swings = _space_swings(swings, min_bars=10, min_pct=min_pct)
    if len(swings) < 4:
        swings = _space_swings(major_zigzag(work, 0.08), min_bars=8, min_pct=0.08)
    events, bias = _events_from_swings(work, swings)
    last_event = events[-1].kind if events else "NONE"
    fresh = bool(events) and events[-1].i >= len(work) - 25
    # Phá rồi lấy lại: không còn tính là phá đáy/đảo đỉnh.
    if fresh and last_event in ("BOS_DN", "CHOCH_DN") and close > events[-1].price * 1.015:
        fresh = False
    if fresh and last_event in ("BOS_UP", "CHOCH_UP") and close < events[-1].price * 0.985:
        fresh = False

    hs = [s for s in swings if s.kind == "H"]
    ls = [s for s in swings if s.kind == "L"]
    if not hs or not ls:
        return None

    # Nhịp fib = 2 swing đối diện gần nhất (đỉnh ↔ đáy), giống thước fib trên chart mẫu
    a, b = swings[-2], swings[-1]
    leg_top = max(a.price, b.price)
    leg_bot = min(a.price, b.price)
    if leg_top <= leg_bot:
        leg_top = float(work["high"].tail(60).max())
        leg_bot = float(work["low"].tail(60).min())
    span = max(leg_top - leg_bot, 1e-9)
    fibs = _fib_prices(leg_top, leg_bot)
    fib_map = {r: p for r, p in fibs}
    # 0 = đỉnh nhịp, 1 = đáy nhịp. depth 0 ở đỉnh, 1 ở đáy
    depth = float(np.clip((leg_top - close) / span, 0, 1))
    eq = fib_map[0.5]
    if depth >= 0.55:
        pd_zone = "DISCOUNT"  # gần đáy nhịp
    elif depth <= 0.45:
        pd_zone = "PREMIUM"  # gần đỉnh nhịp
    else:
        pd_zone = "EQ"

    major_low = float(ls[-1].price)
    major_high = float(hs[-1].price)
    flow, retest, flow_why = _flow_and_retest(work, major_low, flow_hint)
    near_low = depth >= 0.62 or close <= major_low * 1.06
    in_golden = 0.50 <= depth <= 0.86  # hồi về 50–78.6% (và sâu hơn một chút)

    week = _tf_note(work, "W-FRI", max(0.08, min_pct), "Tuần")
    month = _tf_note(work, "ME", 0.12, "Tháng")
    tf_label = f"{week} · {month}"
    higher_tf_blocking = ("phá đáy (mới)" in week) or ("đảo đỉnh (mới)" in week) or ("phá đáy (mới)" in month)

    # vùng mua gần đáy nhịp: từ đáy đến fib 0.618 (tức depth 1.0 → 0.618)
    zone_low = fib_map[1.0]
    zone_high = fib_map[0.618]
    if zone_high < zone_low:
        zone_low, zone_high = zone_high, zone_low
    demand = Zone(
        "BULL_OB",
        i0=int(ls[-1].i),
        i1=int(ls[-1].i),
        low=float(zone_low),
        high=float(zone_high),
        time0=ls[-1].time,
        time1=ls[-1].time,
    )

    buf = max(atr * 0.4, close * 0.006)
    sl = min(major_low, zone_low) - buf
    if sl >= close:
        sl = close - buf
    # TP luôn trên giá
    tp1 = fib_map[0.382] if fib_map[0.382] > close * 1.01 else fib_map[0.236]
    if tp1 <= close:
        tp1 = max(major_high, close * 1.04)
    tp2 = major_high if major_high > tp1 else tp1 + max(atr * 2, span * 0.25)
    if tp2 <= tp1:
        tp2 = tp1 * 1.04
    entry_mid = (zone_low + zone_high) / 2
    # entry phải không nằm trên đầu giá — nếu vùng đáy đã ở dưới, giữ nguyên để chờ hồi
    has_long = entry_mid <= close * 1.015 and sl < min(close, zone_low)
    if zone_high > close * 1.01:
        # giá đang dưới cả vùng fib — không có entry hợp lệ phía trên
        zone_high = min(zone_high, close)
        zone_low = min(zone_low, zone_high)
        has_long = zone_low < close and sl < zone_low
    risk = max(entry_mid - sl, 1e-9)
    reward = tp1 - entry_mid
    rr = float(reward / risk) if reward > 0 else 0.0

    invalidation = sl
    structure = {
        "bull": "Sóng lớn đang tăng (đỉnh sau cao hơn, đáy sau cao hơn)",
        "bear": "Sóng lớn đang giảm (đỉnh sau thấp hơn, đáy sau thấp hơn)",
        "range": "Sóng lớn đi ngang",
    }[bias]

    plain = _plain(last_event) if fresh else _plain("NONE")
    if not fresh and last_event != "NONE":
        plain += " Lần phá mức trước đã cũ, không dùng để vào lệnh nữa."

    action: Action = "WATCH"
    tactics: list[str] = []

    if last_event == "BOS_DN" and fresh:
        action = "EXIT"
        has_long = False
        tactics = [
            "Phá đáy lớn: không mua. Đang cầm thì bán, cắt dưới đáy vừa thủng.",
            f"Mốc cắt long: {major_low:.2f}.",
            "Chờ đáy mới và có đảo đáy (CHoCH lên) rồi mới nghĩ đến mua.",
        ]
    elif last_event == "CHOCH_DN" and fresh:
        action = "EXIT"
        has_long = False
        tactics = [
            "Đảo đỉnh: chuỗi tăng gãy. Không mua mới. Đang cầm thì bán hoặc hạ mạnh tỷ trọng.",
            f"Đáy nhịp vừa mất quanh {major_low:.2f}.",
        ]
    elif last_event == "CHOCH_UP" and fresh:
        if near_low and (flow or retest) and not higher_tf_blocking and rr >= 1.5 and has_long:
            action = "BUY"
            tactics = [
                "Đảo đáy + còn sát đáy + có xác nhận (dòng tiền hoặc test đáy).",
                f"Mua trong vùng {zone_low:.2f}–{zone_high:.2f} (gần đáy nhịp, fib 61.8–100%).",
                f"Cắt lỗ nếu đóng cửa dưới {sl:.2f}. Chốt một phần tại {tp1:.2f}, phần còn lại kỳ vọng {tp2:.2f}.",
                f"Xác nhận: {flow_why}.",
            ]
        else:
            action = "WATCH"
            has_long = has_long and not higher_tf_blocking
            why_no = []
            if not near_low:
                why_no.append("giá đã chạy xa đáy")
            if not (flow or retest):
                why_no.append("chưa có dòng tiền / chưa test đáy thành công")
            if higher_tf_blocking:
                why_no.append("khung tuần hoặc tháng vừa phá đáy / đảo đỉnh")
            if rr < 1.5:
                why_no.append("lãi/lỗ kỳ vọng chưa đủ")
            tactics = [
                "Có đảo đáy nhưng chưa đủ điều kiện mua: " + (", ".join(why_no) or "chờ thêm") + ".",
                "Không mua đuổi. Chỉ mua khi giá còn trong vùng gần đáy và có dòng tiền.",
            ]
            if has_long:
                tactics.append(f"Vùng chờ: {zone_low:.2f}–{zone_high:.2f}. Cắt lỗ {sl:.2f}. Mục tiêu {tp1:.2f} rồi {tp2:.2f}.")
    elif last_event == "BOS_UP" and fresh:
        # phá đỉnh — không đuổi; mua nếu đã hồi về golden
        if in_golden and (flow or retest) and not higher_tf_blocking and rr >= 1.5 and has_long and close <= zone_high * 1.02:
            action = "BUY"
            tactics = [
                "Đã phá đỉnh trước đó và nay hồi về 50–78.6% của nhịp tăng, có dòng tiền hoặc giữ đáy nhịp.",
                f"Mua {zone_low:.2f}–{zone_high:.2f}. Cắt lỗ {sl:.2f}. Mục tiêu {tp1:.2f} / {tp2:.2f}.",
                f"Xác nhận: {flow_why}.",
            ]
        else:
            action = "WATCH"
            tactics = [
                "Phá đỉnh: xu hướng tăng còn, nhưng không mua đuổi ở vùng giá cao.",
                f"Chờ hồi về {fib_map[0.5]:.2f}–{fib_map[0.786]:.2f} (50–78.6%). Cắt lỗ dưới {sl:.2f}. Mục tiêu trên giá: {tp1:.2f} rồi {tp2:.2f}.",
            ]
            if higher_tf_blocking:
                tactics.append("Khung tuần/tháng đang xấu — ưu tiên đứng ngoài dù ngày vừa phá đỉnh.")
                has_long = False
    else:
        # Không có phá đỉnh/đáy MỚI trên ngày — không hô bán chỉ vì sóng đang giảm.
        if higher_tf_blocking:
            action = "EXIT"
            has_long = False
            plain = "Khung tuần hoặc tháng vừa phá đáy hoặc đảo đỉnh. Không mua. Nếu đang cầm thì bán."
            tactics = [
                "Tín hiệu bán nằm ở khung tuần/tháng, không phải vài phiên ngày.",
                f"Mốc cắt tham chiếu: {major_low:.2f}.",
                tf_label,
            ]
        elif bias == "bear":
            action = "WATCH"
            has_long = False
            tactics = [
                "Sóng lớn vẫn giảm nhưng chưa thủng đáy mới. Không mua.",
                "Chỉ bán khi có phá đáy (BOS) hoặc đảo đỉnh (CHoCH). Chưa tới thì đứng ngoài.",
            ]
        elif bias == "bull" and in_golden and (flow or retest) and rr >= 1.5 and has_long:
            action = "WATCH"
            tactics = [
                "Giá gần đáy một nhịp tăng và có dòng tiền, nhưng chưa phá đỉnh/đảo đáy mới.",
                "Chưa mua một mình — cần các chiến thuật kia cùng chiều hoặc một lần phá đỉnh rõ.",
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

    depth_pct = depth * 100
    thesis = " · ".join([
        plain,
        structure,
        f"Giá ở {depth_pct:.0f}% từ đỉnh nhịp về đáy (0%=đỉnh, 100%=đáy)",
        tf_label,
        f"Dòng tiền: {flow_why}",
    ])

    poi_label = (
        f"Vùng gần đáy nhịp (fib 61.8–100%): {zone_low:.2f}–{zone_high:.2f}"
        if has_long else
        "Không mở long"
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
        sl = f"{plan.sl:.2f}"
        tp1 = f"{plan.tp1:.2f}"
        tp2 = f"{plan.tp2:.2f}"
        rr = f"1:{plan.rr:.1f}"
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
    import plotly.graph_objects as go

    work = df.tail(220).reset_index(drop=True)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=work["time"],
                open=work["open"],
                high=work["high"],
                low=work["low"],
                close=work["close"],
                name="Giá",
                increasing_line_color="#3DDC97",
                decreasing_line_color="#FF6B6B",
            )
        ]
    )
    # đường sóng lớn
    sx = [s.time for s in plan.swings if s.time in set(work["time"]) or True]
    # thời gian có thể lệch kiểu — lọc theo vị trí nếu time khớp được
    times = list(work["time"])
    line_x, line_y = [], []
    for s in plan.swings:
        if s.time in times or str(s.time) in {str(t) for t in times}:
            line_x.append(s.time)
            line_y.append(s.price)
    if len(line_x) >= 2:
        fig.add_trace(
            go.Scatter(
                x=line_x, y=line_y, mode="lines+markers",
                line=dict(color="#60A5FA", width=2),
                marker=dict(size=7, color="#60A5FA"),
                name="Sóng lớn",
                hovertext=["Đỉnh lớn" if s.kind == "H" else "Đáy lớn" for s in plan.swings if s.time in line_x or True][:len(line_x)],
            )
        )
    for r, price in plan.fibs:
        fig.add_hline(
            y=price,
            line_color="#334155" if r not in (0.5, 0.618, 0.786) else "#14532d",
            line_dash="dot",
            annotation_text=f"{r:.3g} ({price:.2f})",
            annotation_font_size=10,
            annotation_font_color="#94A3B8",
        )
    if plan.zones:
        z = plan.zones[-1]
        fig.add_hrect(
            y0=z.low, y1=z.high,
            fillcolor="rgba(61,220,151,0.16)",
            line_width=0,
            annotation_text="vùng gần đáy",
            annotation_font_size=11,
        )
    if plan.events:
        e = plan.events[-1]
        fig.add_vline(
            x=e.time, line_dash="dash", line_color="#A78BFA",
            annotation_text={
                "BOS_UP": "Phá đỉnh",
                "BOS_DN": "Phá đáy",
                "CHOCH_UP": "Đảo đáy",
                "CHOCH_DN": "Đảo đỉnh",
            }.get(e.kind, e.kind),
        )
    if plan.has_long_setup:
        fig.add_hline(y=plan.sl, line_color="#FF6B6B", annotation_text="Cắt lỗ")
        fig.add_hline(y=plan.tp1, line_color="#3DDC97", annotation_text="Chốt 1")
        fig.add_hline(y=plan.tp2, line_dash="dash", line_color="#3DDC97", annotation_text="Chốt 2")
    else:
        fig.add_hline(y=plan.invalidation, line_color="#FF6B6B", annotation_text="Cắt long")
    fig.update_layout(
        title=title or "SMC khung lớn",
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0E1624",
        height=680,
        xaxis_rangeslider_visible=False,
        margin=dict(l=12, r=12, t=48, b=12),
        font=dict(family="IBM Plex Sans, sans-serif", color="#E8EDF5"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1E2A3F")
    fig.update_yaxes(gridcolor="#1E2A3F")
    return fig
