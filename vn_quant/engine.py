# VN Quant Engine — extracted from notebooks (Claude v1.1 + VSA/Ichimoku/RSI-DG)
from __future__ import annotations

import warnings
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — CONFIG                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Tất cả tham số điều chỉnh được tập trung tại đây — không cần sửa code
# trong các class. Khi muốn thay đổi behavior, chỉ override QuantConfig.
#
# Ví dụ:  cfg = QuantConfig(capital=200_000_000, risk_per_trade=0.005)
#         engine = QuantEngine(cfg=cfg)

@dataclass
class QuantConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    lookback_short:     int   = 50      # Phiên ngắn hạn cho indicators T+
                                        # 40 đủ tính ATR14, Stoch, EMA20 mà
                                        # không kéo dài về quá khứ
    lookback_long:      int   = 600     # Phiên dài hạn cho SMA50/SMA200
                                        # SMA200 cần tối thiểu 200 điểm data
    sources:            list  = field(default_factory=lambda: ["KBS", "VCI"])
    delay_sec:          float = 2    # Delay giữa mỗi mã tránh rate-limit
    vp_bins:            int   = 35      # ✏️ v5.2: 8→30
                                        # 30 bins trên 35 ngày ≈ zone 0.3%
                                        # Đủ chi tiết xác định POC/VAH/VAL
                                        # mà không bị quá thưa (8) hay nhiễu (50+)

    # ── Scoring thresholds ────────────────────────────────────────────────────
    min_score_swing:    float = 5.0     # Điểm tối thiểu để vào watchlist T+
    min_score_hold:     float = 6.0     # Điểm tối thiểu Hold (khắt khe hơn)
    min_rr:             float = 1.5     # R:R tối thiểu chung

    # ── Liquidity Filter (VN-specific) ────────────────────────────────────────
    min_avg_vol_20d:    int   = 50_000  # Tối thiểu 50K cổ/ngày trung bình 20d
    min_avg_val_20d:    float = 10e6    # Tối thiểu 10 tỷ VNĐ/ngày

    # ── Position Sizing ───────────────────────────────────────────────────────
    capital:            float = 100_000_000
    risk_per_trade:     float = 0.01    # 1% NAV tối đa mỗi lệnh
    max_position_pct:   float = 0.20    # Tối đa 20% NAV / 1 mã
    kelly_fraction:     float = 0.25    # Conservative Kelly (1/4 full Kelly)

    # ── Smart Entry ───────────────────────────────────────────────────────────
    entry_pullback_pct: float = 0.35     # % pullback từ giá hiện tại cho entry
    entry_atr_mult:     float = 0.35     # Multiplier ATR cho entry dip

    # ── Partial TP ────────────────────────────────────────────────────────────
    partial_tp_pct:     float = 0.3     # Chốt 30% (1/3) tại T1

    # ── R:R Gate ─────────────────────────────────────────────────────────────
    min_rr_swing:       float = 2.0     # R:R tối thiểu cho T+
    min_rr_hold:        float = 3.0     # R:R tối thiểu cho Hold

    # ── Volume Dry-up ─────────────────────────────────────────────────────────
    dryup_vol_ratio:    float = 0.70    # Vol < 70% MA20 → cạn cung
    dryup_sessions:     int   = 7       # Quan sát 7 phiên liên tiếp
    dryup_range_ratio:  float = 0.75    # Biên độ HL < 75% ATR → nền hẹp
    dryup_bonus:        float = 2.0     # Bonus score tối đa khi dry-up

    # ── Win Rate ─────────────────────────────────────────────────────────────
    wr_base:            float = 45.0    # Win rate base (%) trước điều chỉnh
                                        # 45% = tỷ lệ thực tế thị trường VN

    # ── MomentumEngine (v5.3) ─────────────────────────────────────────────────
    bb_period:          int   = 20      # Chu kỳ Bollinger Bands
    bb_std:             float = 2.0     # Số std dev cho BB
    bb_squeeze_pct:     float = 15.0    # Percentile bandwidth để xác định squeeze
    bb_squeeze_lookback: int  = 60      # Lookback tính percentile squeeze
    div_lookback:       int   = 15      # Nến lookback tìm swing pivot (VN: 15 > US: 10)
    div_min_bars:       int   = 5       # Khoảng cách tối thiểu giữa 2 pivot
    hidden_acc_cmf_min: float = 0.03    # CMF tăng > 3% để xác nhận hidden accumulation

    # ── IchimokuEngine (v5.4) ─────────────────────────────────────────────────
    # Tham số chuẩn Ichimoku gốc Nhật (9/26/52)
    ichi_tenkan:        int   = 9       # Đường chuyển đổi — short-term momentum
    ichi_kijun:         int   = 26      # Đường cơ sở — medium-term trend/S/R
    ichi_senkou_b:      int   = 52      # Senkou Span B — long-term equilibrium
    ichi_cloud_merge_pct: float = 1.5   # % để merge 2 S/R gần nhau thành 1 vùng

    # ── T+ Target Filter (v5.4) ───────────────────────────────────────────────
    # Chỉ trade khi biên độ đủ để đạt lợi nhuận tối thiểu
    min_tp_pct:         float = 9.0     # T1 target tối thiểu 9% so với entry
    min_atr_pct:        float = 1.5     # ATR% tối thiểu: mã biên hẹp không đủ
    swing_t1_mult:      float = 2.0     # T1 = entry + risk × 2.0  (tăng từ 1.5)
    swing_t2_mult:      float = 3.5     # T2 = entry + risk × 3.5  (tăng từ 2.5)
    hold_t1_mult:       float = 2.5     # Hold T1 (giữ nguyên)
    hold_t2_mult:       float = 4.0     # Hold T2 (giữ nguyên)

    # ── ScoreCoeffs — thay magic numbers trong ScoringEngine (v5.4) ──────────
    # Volume ratio piecewise breakpoints
    sc_vol_low_slope:   float = 4.0     # vol ≤ 0.8 → score = vol × slope
    sc_vol_mid_base:    float = 3.5     # vol [0.8,1.3) base
    sc_vol_mid_slope:   float = 6.0     # vol [0.8,1.3) slope
    sc_vol_hi_base:     float = 6.0     # vol [1.3,2.2) base
    sc_vol_hi_slope:    float = 3.5     # vol [1.3,2.2) slope
    sc_vol_vhi_base:    float = 8.5     # vol ≥ 2.2 base
    sc_vol_vhi_slope:   float = 1.2     # vol ≥ 2.2 slope
    # CMF: range [-0.25, +0.25] → [0,10]
    sc_cmf_shift:       float = 0.25
    sc_cmf_scale:       float = 0.055
    # MACD histogram → score multiplier
    sc_macd_hist_mult:  float = 480.0
    # EMA % distance → score
    sc_ema_shift:       float = 6.0
    sc_ema_scale:       float = 1.1
    # Force index
    sc_force_shift:     float = 1.0
    sc_force_scale:     float = 4.8
    # RS score
    sc_rs_mult:         float = 4.2
    sc_rs_base:         float = 1.8

    # ── Dao Găm & MA Signals (v5.5 NEW) ──────────────────────────────────────
    dao_gam_tolerance_pct: float = 0.5   # % dung sai để tính là "chạm" Kijun
    ma_touch_tolerance_pct: float = 1.0  # % dung sai để tính là "chạm" MA50/200
    ma_short_period_1:  int = 9          # MA ngắn hạn 1
    ma_short_period_2:  int = 10         # MA ngắn hạn 2

    # ── DMI/ADX & Aroon (v5.6 NEW) ────────────────────────────────────────────
    dmi_period:      int   = 14
    aroon_period:    int   = 25
    adx_trend_min:   float = 25.0   # ADX ≥ 25 = xu hướng đủ mạnh để tin cậy
    adx_weak_min:    float = 20.0   # ADX 20-25 = xu hướng yếu, cẩn trọng

    # ── Composite Index RSI Divergence (v5.7 NEW) ─────────────────────────────
    # Port từ Pine Script "Composite Index Divergence Indicator" (z3bazi / Constance Brown).
    # Tên tham số ↔ input của script gốc:
    cd_rsi_len:      int   = 14     # rsi_length           — RSI chính
    cd_mom_len:      int   = 9      # rsi_mom_length       — momentum của RSI chính
    cd_rsi_ma_len:   int   = 3      # rsi_ma_length        — RSI nhanh (chu kỳ 3)
    cd_sma_len:      int   = 3      # ma_length            — SMA làm mượt RSI nhanh
    cd_fast_len:     int   = 13     # fastLength           — SMA nhanh của Composite (đường tín hiệu)
    cd_slow_len:     int   = 33     # slowLength           — SMA chậm của Composite
    cd_lbl:          int   = 5      # lbL  — số nến TRÁI khi xác định pivot
    cd_lbr:          int   = 5      # lbR  — số nến PHẢI = độ trễ xác nhận pivot (phiên)
    cd_range_min:    int   = 5      # rangeLower — khoảng cách tối thiểu giữa 2 pivot
    cd_range_max:    int   = 60     # rangeUpper — khoảng cách tối đa giữa 2 pivot
    cd_use_close:    bool  = False  # useclose — True: so giá đóng cửa; False: dùng low/high (bóng nến)
    # Phần bổ sung (không có trong script gốc):
    cd_signal_window: int  = 3      # Phân kỳ được coi là "MỚI" nếu xác nhận trong N phiên gần nhất
    cd_replace_legacy_rsi_div: bool = True
                                    # True: dùng phân kỳ RSI-Composite thay cho rsi_bull/bear_div
                                    #       cũ của MomentumEngine (tránh cộng điểm 2 lần). False: giữ cũ (A/B test)
    dg_near_pct:     float = 2.0    # Cảnh báo "sắp về Dao găm" khi giá cách Kijun ≤ N% phía trên



# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST — giữ nguyên từ v5
# ═══════════════════════════════════════════════════════════════════════════════


VN100 = ['FPT', 'VCB', 'HPG', 'TCB', 'ACB', 'BID', 'MWG', 'SSI', 'VND', 'CTD', 'VHM', 'VIC', 'VRE', 'PNJ', 'MSN',
                   'GAS', 'PLX', 'SAB', 'STB', 'SHB', 'LPB', 'VPB', 'MBB', 'HDB', 'TPB', 'OCB', 'SSB', 'VIB', 'CTG', 'POW',
                   'GVR', 'DGC', 'DPM', 'BSR', 'PVD', 'PVT', 'VCG', 'VSC', 'HSG', 'NKG', 'BCM', 'KDH', 'DXG', 'NLG', 'TCH',
                   'DIG', 'PDR', 'KBC', 'CRE', 'VPI', 'HDG', 'AGG', 'HT1', 'AAA', 'DBC', 'NVL', 'VHC', 'MSH', 'CII', 'VGI',
                   'ANV', 'SMC', 'HAG', 'DLG', 'BFC', 'PTB', 'TLH', 'PHR', 'DCM', 'LAS', 'VIX', 'MSR', 'GEL', 'HUT', 'GEG',
                   'GEX', 'FTS', 'HCM', 'CTS', 'MBS', 'VCI', 'ORS', 'BSI', 'BAF', 'BMP', 'BVH', 'BWE', 'CMG', 'CTR', 'DGW',
                   'DSE', 'DXS', 'EIB', 'EVF', 'FRT', 'GEE', 'GMD', 'HDC', 'HHV', 'IMP', 'KDC', 'KOS', 'MSB', 'NAB', 'NT2',
                   'PAN', 'PC1', 'REE', 'SBT', 'SCS', 'SIP', 'SJS', 'SZC', 'VGC', 'VJC', 'VNM', 'VPL', 'CEO', 'SHS', 'PVS',
                   'MST', 'VC3', 'IDC', 'VTZ', 'VFS', 'PVC', 'C69', 'TNG', 'VGS', 'KSF', 'PVB', 'BVS', 'CTP', 'FID', 'VTV',
                   'PLC', 'OCH', 'PVI', 'NTP', 'IPA', 'DXP', 'VIW', 'TVN', 'OIL', 'ACV', 'HNG', 'BVB', 'FOX', 'DRI', 'DDV',
                   'AAS', 'VEA', 'TV1', 'ABB', 'VGT', 'QTP', 'MZG', 'PHP', 'SBS', 'STH', 'F88', 'HNM', 'FOC', 'DSH', 'BMS',
                   'QNS', 'TOS', 'XMC', 'BNA', 'VC7', 'VC2', 'DTD', 'UNI', 'NBC', 'NAG', 'CAP', 'PSI', 'VGP', 'PPT', 'PCH',
                   'NVB', 'L40', 'APS', 'VHE', 'API', 'PVG', 'IDJ', 'LDP', 'CMS', 'DST', 'AAV', 'AMS', 'BIG', 'VTD', 'G36',
                   'DSH', 'DDB', 'CLI', 'PIV', 'CLX', 'TTG', 'DVN', 'PXL', 'MPC', 'ABW', 'PGB', 'VBB', 'NCG', 'ALC', 'KCB',
                   'GCF', 'KLB', 'VAB', 'C32', 'CIG', 'HSL', 'RYG', 'VTO', 'HII', 'TCM', 'PPC', 'ELC', 'TCI', 'SHI', 'DAH',
                   'LDG', 'HQC', 'HHP', 'SCR', 'LCG', 'QCG', 'TTH', 'NRC', 'TVC', 'TIG', 'SVN', 'DL1', 'KIP', 'CTX', 'APF',
                   'SBB', 'VNP', 'HVN', 'TLG', 'PET', 'VOS']


# Base weights — normalized về 100, regime sẽ điều chỉnh dynamic
BASE_WEIGHTS = {
    "volume_ratio": 11, "cmf": 9,  "obv": 5, "force": 3, "inst_flow": 4,
    "rsi": 9,  "ema": 6, "stoch": 5, "macd": 5,
    "vp_position": 11, "trend": 9, "rs_score": 4,
    "vsa": 9,
    "momentum_div": 9,   # v5.3: divergence
    "ichimoku": 11,      # v5.4: Ichimoku multi-TF (weight cao vì 3TF = reliable)
    "dmi_aroon": 7,        # ✏️ v5.6 — vừa phải, dưới Ichimoku (đa TF toàn diện hơn)
}

REGIME_WEIGHT_DELTA = {
    "bull":     {"volume_ratio": +3, "trend": +4, "rsi": -2, "vp_position": +2,
                 "vsa": +2, "ichimoku": +3, "dmi_aroon": +2},
    "sideways": {"cmf": +3, "obv": +3, "vp_position": +4, "trend": -3,
                 "momentum_div": +2, "ichimoku": +2, "dmi_aroon": -2},  # ADX tự nhiên thấp khi sideway → giảm tin cậy
    "bear":     {"cmf": +5, "trend": -5, "rsi": +3, "volume_ratio": -3,
                 "vsa": +3, "momentum_div": +3, "ichimoku": +4, "dmi_aroon": +2},
    "panic":    {"cmf": +5, "force": +5, "volume_ratio": +5, "rsi": -5,
                 "vsa": +4, "momentum_div": +4, "ichimoku": +5, "dmi_aroon": +3},
}

REGIME_SCORE_MULT = {"bull": 1.0, "sideways": 0.95, "bear": 0.85, "panic": 0.70}
REGIME_WR_MULT    = {"bull": 1.15, "sideways": 1.0, "bear": 0.80, "panic": 0.65}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — DATA LAYER                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Nhiệm vụ: Fetch OHLCV từ vnstock, chuẩn hóa tên cột, xử lý missing data.
#
# Vấn đề thực tế ở VN:
#   - API vnstock có v3 và v4 với signature khác nhau → dùng fallback
#   - Ngày lễ, ngày không giao dịch → forward fill giá, volume = 0
#   - Một số mã trả về tên cột khác nhau tùy source (KBS vs VCI)
#
# Design choice: không raise exception, return None nếu data không hợp lệ
# → caller (pipeline) tự quyết định skip hay retry

class DataLayer:
    """
    Fetch và chuẩn hóa OHLCV data từ vnstock.

    Hỗ trợ hai version API vnstock (v3/v4) với cơ chế fallback tự động.
    Xử lý missing data theo chuẩn: ffill giá (giữ giá cuối hợp lệ),
    volume=0 khi không giao dịch (tránh nhầm với phiên bình thường).

    Output luôn là DataFrame đã chuẩn hóa với cột:
    time, open, high, low, close, volume
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def fetch(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch data với fallback strategy:
        1. Thử vnstock v4 (Quote API)
        2. Thử vnstock v3 (Vnstock API)
        3. Thử source khác trong sources list
        → Return None nếu tất cả fail hoặc data < 20 phiên
        """
        end   = (datetime.today() + timedelta(1)).strftime("%Y-%m-%d")
        # Buffer +45 ngày để bù ngày lễ và ensure đủ lookback
        start = (datetime.today() - timedelta(days=days + 1)).strftime("%Y-%m-%d")
        for src in self.cfg.sources:
            for fn in (self._v4, self._v3):
                try:
                    df = fn(symbol, start, end, src)
                    if df is not None:
                        clean = self._clean(df, days)
                        if clean is not None and len(clean) >= 20:
                            return clean
                except Exception:
                    print(f'Fail rồi cu: {Exception}')
                    continue
        return None

    def _v4(self, sym, start, end, src):
        from vnstock import Quote
        q = Quote(symbol=sym, source=src)
        try:    return q.history(start=start, end=end, interval="1D")
        except TypeError:
            return q.history(start_date=start, end_date=end, interval="1D")

    def _v3(self, sym, start, end, src):
        from vnstock import Vnstock
        stk = Vnstock().stock(symbol=sym, source=src)
        try:    return stk.quote.history(start=start, end=end, interval="1D")
        except TypeError:
            return stk.quote.history(start_date=start, end_date=end, interval="1D")

    def _clean(self, df: pd.DataFrame, days: int) -> Optional[pd.DataFrame]:
        """
        Chuẩn hóa và làm sạch data:
        - Normalize tên cột về lowercase
        - Map alias (tradingdate→time, o→open, ...)
        - ffill/bfill giá để lấp ngày lễ
        - volume=0 thay vì NaN (ngày không GD ≠ thiếu data)
        - Loại bỏ row close=0 (data lỗi, không phải ngày lễ)
        - Lấy tail(days) để đảm bảo đúng lookback
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.rename(columns={
            "tradingdate": "time", "date": "time",
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        })
        if not {"time", "open", "high", "low", "close", "volume"}.issubset(df.columns):
            return None
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].ffill().bfill()
        df["volume"] = df["volume"].fillna(0)
        df = df[df["close"] > 0].sort_values("time").reset_index(drop=True)
        return df.tail(days).reset_index(drop=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — REGIME ENGINE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Tại sao cần phát hiện Regime?
#   RSI=80 trong bull market → vẫn có thể mua tiếp
#   RSI=60 trong bear market → có thể đã là đỉnh
#   Cùng một setup kỹ thuật, kết quả khác nhau hoàn toàn tùy thị trường.
#
# Approach: đo VNINDEX, không đo từng mã
#   - EMA10/EMA30 cross → trend ngắn hạn
#   - SMA200 → trend dài hạn
#   - Momentum 5 phiên → tốc độ di chuyển
#   - ATR% → mức biến động → phân biệt Panic vs Bear bình thường
#
# Output: 4 regime + dynamic weights + score multiplier + win rate multiplier
#   Bull:     weights thiên momentum, multiplier 1.0
#   Sideways: weights thiên mean-reversion (VP, CMF), multiplier 0.95
#   Bear:     siết filter, giảm score 15%, multiplier 0.85
#   Panic:    chỉ giao dịch cực thận trọng, multiplier 0.70

class RegimeEngine:
    """
    Phát hiện Market Regime từ VNINDEX và tính Dynamic Weights.

    Thuật toán:
    1. Tính EMA10, EMA30, SMA200 của VNINDEX
    2. Đo momentum 5 phiên (mom5)
    3. Đo ATR% để phân biệt Panic (biến động cao + giảm mạnh)
    4. Rule-based classification → 4 regime
    5. Điều chỉnh BASE_WEIGHTS theo regime (REGIME_WEIGHT_DELTA)
    6. Normalize weights về tổng = 100

    Tại sao rule-based thay vì ML?
    - Data VNINDEX chỉ có từ 2000, không đủ cho supervised learning
    - Rule-based dễ audit, dễ debug, không overfit
    - Regime thay đổi chậm (tuần/tháng), không cần model phức tạp
    """

    def __init__(self, data_layer: DataLayer):
        self.dl      = data_layer
        self._regime = "sideways"
        self._ctx    = {}

    def detect(self, df_override: pd.DataFrame = None, quiet: bool = False) -> dict:
        if not quiet:
            print("  📊 VN-Index regime...", end=" ", flush=True)
        df = df_override if df_override is not None else self.dl.fetch("VNINDEX", days=60)
        if df is None or len(df) < 30:
            if not quiet: print("⚠ neutral")
            self._regime = "sideways"
            self._ctx = {
                "regime": "sideways", "score_mult": 0.95,
                "label": "⚪ Không xác định", "vnindex": 0.0,
                "wr_mult": 1.0,
            }
            return self._ctx

        c     = df["close"]
        ema10 = c.ewm(span=10, adjust=False).mean()
        ema30 = c.ewm(span=30, adjust=False).mean()
        ma200 = c.rolling(200).mean() if len(c) >= 200 else ema30
        atr   = pd.concat([
            df["high"] - df["low"],
            (df["high"] - c.shift()).abs(),
            (df["low"]  - c.shift()).abs()
        ], axis=1).max(axis=1).rolling(14).mean()

        trend_up    = ema10.iloc[-1] > ema30.iloc[-1]
        above_ma200 = c.iloc[-1] > ma200.iloc[-1]
        mom5        = (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] * 100 if len(c) >= 6 else 0.0
        atr_norm    = atr.iloc[-1] / c.iloc[-1] * 100   # ATR% — chuẩn hóa theo giá
        high_vol    = atr_norm > 2.5                     # >2.5% là biến động cao bất thường

        # Thứ tự if-elif quan trọng: Panic > Bear > Bull > Sideways
        if   high_vol and mom5 < -2.0:                regime, label, ctx = "panic",    "🔴 PANIC — thận trọng tối đa",  1.5
        elif not trend_up and mom5 < -1.0:            regime, label, ctx = "bear",     "🔴 Downtrend — hạ tỷ trọng",    2.5
        elif trend_up and above_ma200 and mom5 > 1.0: regime, label, ctx = "bull",     "🟢 Uptrend — thuận chiều mua",  8.0
        elif trend_up and mom5 > -1.0:                regime, label, ctx = "sideways", "🟡 Uptrend nghỉ / tích lũy",   6.0
        else:                                          regime, label, ctx = "sideways", "🟡 Sideways — chọn lọc",       4.5

        mult = REGIME_SCORE_MULT[regime]
        print(f"✓  {c.iloc[-1]:,.2f}  |  {label}  |  5d:{mom5:+.1f}%  ATR:{atr_norm:.1f}%")

        self._regime = regime
        self._ctx = {
            "regime":        regime,
            "label":         label,
            "context_score": ctx,
            "score_mult":    mult,
            "wr_mult":       REGIME_WR_MULT[regime],
            "mom5":          round(mom5, 2),
            "vnindex":       round(float(c.iloc[-1]), 2),
            "atr_pct":       round(atr_norm, 2),
        }
        if not quiet:
            print(f"✓  {c.iloc[-1]:,.2f}  |  {label}  |  5d:{mom5:+.1f}%  ATR:{atr_norm:.1f}%")
        return self._ctx

    def get_dynamic_weights(self) -> dict:
        """
        Áp dụng delta lên BASE_WEIGHTS theo regime, normalize về tổng=100.

        Ví dụ Bull regime:
        - volume_ratio +3: momentum đang mạnh, volume quan trọng hơn
        - trend +5: uptrend rõ ràng, trọng số trend tăng
        - rsi -2: RSI cao không còn là cảnh báo mua đuổi trong bull
        """
        w = dict(BASE_WEIGHTS)
        delta = REGIME_WEIGHT_DELTA.get(self._regime, {})
        for k, d in delta.items():
            if k in w:
                w[k] = max(0, w[k] + d)
        total = sum(w.values())
        return {k: round(v / total * 100, 1) for k, v in w.items()}

    def get_min_score(self, mode: str, cfg: QuantConfig) -> float:
        """
        Nâng min_score khi thị trường xấu để lọc chặt hơn.
        context_score cao (≥7) = thị trường tốt → giữ ngưỡng gốc
        context_score thấp (<3.5) = thị trường xấu → tăng ngưỡng +1.5
        """
        base = cfg.min_score_swing if mode == "swing" else cfg.min_score_hold
        ctx  = self._ctx.get("context_score", 5.0)
        if   ctx >= 7.0: return base
        elif ctx >= 5.0: return base + 0.5
        elif ctx >= 3.5: return base + 1.0
        else:            return base + 1.5

    @property
    def regime(self) -> str:
        return self._regime

    @property
    def ctx(self) -> dict:
        return self._ctx

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — VSA ENGINE  (NEW v5.2)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Volume Spread Analysis (VSA) — đọc dấu chân Smart Money
#
# Tại sao chỉ volume_ratio và CMF chưa đủ?
#   - volume_ratio chỉ biết "nhiều hay ít", không biết "mua hay bán"
#   - CMF là lagging indicator (tính 21 phiên), phản ứng chậm
#   - Smart Money có thể giả tạo volume (pump & dump):
#     * Nổ volume MUA thực sự: volume lớn + spread rộng + đóng cửa CAO
#     * Xả hàng núp bóng:      volume lớn + spread rộng + đóng cửa THẤP
#     → Chỉ nhìn volume sẽ không phân biệt được
#
# Ba thành phần VSA:
#   1. Volume   : vol_ratio = volume / MA(volume, 20) → chuẩn hóa
#   2. Spread   : spread_ratio = (H-L) / MA(H-L, 20) → chuẩn hóa
#   3. Close Pos: (close - low) / spread → 0=đóng thấp nhất, 1=đóng cao nhất
#                 Đây là key để phân biệt mua thật vs xả hàng
#
# 5 trạng thái và logic:
#   🚀 NỔ VOL    : vol>1.5x AND spread>1.2x AND close_pos>0.7 AND roc>1.5%
#                  → Lực mua áp đảo, đóng gần đỉnh, breakout thực
#   🩸 CẠN VOL   : vol<0.6x AND spread<0.85x AND close_pos≥0.4 AND roc>-1.5%
#                  → Cung cạn, không ai bán, nền trước breakout
#   🔨 RÚT CHÂN  : vol>1.3x AND spread>1.3x AND close_pos>0.65 AND roc>-2%
#                  → Vol lớn nhưng đóng cao → rũ hàng yếu tay xong gom lại
#                  (v5.2 fix: roc > -0.02 thay vì < 0.01 để loại nến đỏ sâu)
#   ⚠️ PHÂN PHỐI : vol>1.5x AND close_pos<0.35 AND spread>1.0x AND roc<2%
#                  → Vol lớn nhưng đóng thấp → Smart Money đang xả
#   🥀 CẠN CẦU  : vol<0.6x AND close_pos<0.4 AND roc>0
#                  → Giá tăng nhẹ không có lực đỡ → breakout sẽ thất bại
#
# Implementation: np.select vectorized — O(n) trên toàn series, không loop
# → Có thể tính cho 1000 phiên trong microseconds

class VSAEngine:
    """
    Volume Spread Analysis — phát hiện hành vi Smart Money.

    Đọc 3 chiều cùng lúc: Volume × Spread × ClosePosition
    để phân biệt mua thật / xả hàng / rũ nền / cạn cung.

    Tất cả tính toán vectorized với numpy, không dùng loop Python.
    """

    @staticmethod
    def detect(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Tính VSA signal cho toàn bộ DataFrame.

        Parameters
        ----------
        df       : DataFrame với cột open, high, low, close, volume
        lookback : Cửa sổ rolling để tính MA volume và MA spread (mặc định 20)

        Returns
        -------
        pd.Series : VSA signal tại mỗi phiên (string label)
        """
        if df is None or len(df) < lookback + 2:
            return pd.Series(["—"] * len(df), index=df.index if df is not None else None)

        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # ── 1. Chuẩn hóa Volume ───────────────────────────────────────────────
        # vol_ratio > 1 → phiên này giao dịch nhiều hơn trung bình
        # Dùng replace(0, nan) tránh chia cho 0 khi mã ngưng GD
        avg_vol   = v.rolling(lookback).mean().replace(0, np.nan)
        vol_ratio = v / avg_vol

        # ── 2. Chuẩn hóa Spread (biên độ nến H-L) ───────────────────────────
        # spread_ratio > 1 → nến hôm nay rộng hơn trung bình
        spread      = h - l
        avg_spread  = spread.rolling(lookback).mean().replace(0, np.nan)
        spread_ratio = spread / avg_spread

        # ── 3. Vị trí đóng cửa trong nến (Close Position) ────────────────────
        # 0.0 = đóng tại đáy nến (bearish)
        # 0.5 = đóng giữa nến (neutral)
        # 1.0 = đóng tại đỉnh nến (bullish)
        # Xử lý doji (spread=0) → gán 0.5 (neutral)
        close_pos = np.where(spread == 0, 0.5, (c - l) / (spread + 1e-9))

        # ── 4. Rate of Change (1 phiên) ───────────────────────────────────────
        # Để lọc: nến tăng hay giảm bao nhiêu %
        roc = c.pct_change()

        # ── 5. Quy tắc VSA (thứ tự ưu tiên từ trên xuống) ───────────────────
        # np.select trả về choice tương ứng với condition đầu tiên = True
        # Nếu không match → default "—"

        conditions = [
            # 🚀 NỔ VOL: Lực mua áp đảo
            # Vol lớn (>1.5x) + Spread rộng (>1.2x) + Đóng gần đỉnh (>0.7) + Giá tăng (>1.5%)
            (vol_ratio > 1.5) & (spread_ratio > 1.2) & (close_pos > 0.7) & (roc > 0.015),

            # 🩸 CẠN VOL: Cung cạn kiệt, tiền thông minh đang gom âm thầm
            # Vol nhỏ (<0.6x) + Spread hẹp (<0.85x) + Không bị bán tháo (close_pos≥0.4)
            # + Không có roc âm sâu (<-1.5%) để loại panic sell
            (vol_ratio < 0.6) & (spread_ratio < 0.85) & (close_pos >= 0.4) & (roc > -0.015),

            # 🔨 RÚT CHÂN / GOM HÀNG: Rũ yếu tay xong đỡ lại
            # Vol lớn (>1.3x) + Spread rộng (>1.3x) + Đóng cao (>0.65) + KHÔNG phải nến đỏ sâu
            # v5.2 fix: roc > -0.02 (thay roc < 0.01) để loại nến đỏ giảm >2%
            (vol_ratio > 1.3) & (spread_ratio > 1.3) & (close_pos > 0.65) & (roc > -0.02),

            # ⚠️ PHÂN PHỐI: Smart Money đang xả hàng
            # Vol lớn (>1.5x) + Đóng thấp (<0.35 thân nến) + Spread rộng (>1.0x)
            # + Giá không tăng mạnh (<2%) → volume là để xả không phải mua
            (vol_ratio > 1.5) & (close_pos < 0.35) & (spread_ratio > 1.0) & (roc < 0.02),

            # 🥀 CẠN CẦU: Breakout fake — giá tăng không có người mua đỡ
            # Vol nhỏ (<0.6x) + Đóng thấp trong nến (<0.4) + Giá vẫn tăng nhẹ
            # → Người bán không nhiều nhưng người mua cũng không, breakout thiếu sức
            (vol_ratio < 0.6) & (close_pos < 0.4) & (roc > 0),
        ]

        choices = ["🚀 NỔ VOL", "🩸 CẠN VOL", "🔨 RÚT CHÂN", "⚠️ PHÂN PHỐI", "🥀 CẠN CẦU"]

        result = np.select(conditions, choices, default="—")
        return pd.Series(result, index=df.index)

    @staticmethod
    def get_latest(df: pd.DataFrame, lookback: int = 20) -> str:
        """Trả về VSA signal của phiên gần nhất."""
        signals = VSAEngine.detect(df, lookback)
        if signals is None or len(signals) == 0:
            return "—"
        return str(signals.iloc[-1])

    @staticmethod
    def to_score(vsa_signal: str) -> float:
        """
        Map VSA signal → điểm số để tích hợp vào ScoringEngine.

        Scale 0-10 để consistent với các indicator khác:
        - NỔ VOL:    8.5 (cực kỳ tích cực)
        - RÚT CHÂN:  8.0 (tích cực, setup gom hàng)
        - CẠN VOL:   7.0 (tích cực, nền tốt)
        - Neutral:   5.0 (không có tín hiệu rõ)
        - CẠN CẦU:   3.5 (tiêu cực nhẹ)
        - PHÂN PHỐI: 1.5 (tiêu cực mạnh)
        """
        return {
            "🚀 NỔ VOL":    8.5,
            "🔨 RÚT CHÂN":  8.0,
            "🩸 CẠN VOL":   7.0,
            "—":            5.0,
            "🥀 CẠN CẦU":  3.5,
            "⚠️ PHÂN PHỐI": 1.5,
        }.get(vsa_signal, 5.0)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — INDICATOR ENGINE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Tính toàn bộ chỉ báo kỹ thuật từ OHLCV thuần Python/numpy/pandas.
# Không dùng thư viện ta hay pandas_ta để tránh dependency.
#
# Nhóm chỉ báo:
#   Volume   : volume_ratio, OBV, CMF, Force Index
#   Momentum : RSI14, MACD(12,26,9), Stochastic(14,3)
#   Trend    : EMA20, EMA50, SMA50, SMA200 (long_trend)
#   Structure: Volume Profile (POC/VAH/VAL), Bollinger Band %B
#   Relative : RS so với VNINDEX (20 phiên)
#   Flow     : Institutional flow estimate (AD line 5/20)
#   VSA      : (tách riêng, call VSAEngine.get_latest)
#
# Tất cả output là float/bool/str được pack vào dict `raw`
# để ScoringEngine chấm điểm và SignalBuilder tính levels

class IndicatorEngine:
    """
    Tính toàn bộ chỉ báo kỹ thuật từ OHLCV.

    Không dùng thư viện external (ta, pandas_ta) — tất cả tính từ
    công thức gốc để kiểm soát hoàn toàn và tránh dependency issues.
    """

    @staticmethod
    def volume_profile(df: pd.DataFrame, n_bins: int = 30) -> dict:
        """
        Volume Profile — tìm vùng giá giao dịch nhiều nhất.

        Thuật toán:
        1. Chia range giá (Low_min → High_max) thành n_bins vùng bằng nhau
        2. Gán mỗi phiên vào bin tương ứng theo Typical Price = (H+L+C)/3
        3. Cộng dồn volume vào mỗi bin
        4. POC = bin có volume cao nhất
        5. Value Area = tập hợp bin chiếm 68.2% tổng volume (1 std dev)
           → VAH = đỉnh Value Area, VAL = đáy Value Area

        Tại sao n_bins=30?
        - 8 bins trên 35 phiên: zone rộng ~1.5% → entry/SL không chính xác
        - 30 bins: zone ~0.3-0.5% → đủ để đặt lệnh với độ chính xác cao
        - 50+ bins: nhiễu, nhiều bin trống, POC không ổn định

        vp_signal:
        - breakout    : giá > VAH × 1.005 → đang breakout
        - below_value : giá < VAL × 0.995 → dưới vùng giá trị
        - at_poc      : giá sát POC ± 1.5% → điểm cân bằng
        - value_area  : giá trong vùng giá trị → bình thường

        Parameters
        ----------
        df     : DataFrame OHLCV
        n_bins : Số bin chia (mặc định 30, v5.2 từ 8)
        """
        if len(df) < 25:
            return {"poc": None, "vah": None, "val": None, "vp_signal": "unknown"}
        try:
            lo, hi = df["low"].min(), df["high"].max()
            sz     = (hi - lo) / n_bins
            edges  = np.arange(lo, hi + sz + 1e-9, sz)
            df2    = df.copy()
            df2["tp"]  = (df2["high"] + df2["low"] + df2["close"]) / 3
            df2["bin"] = pd.cut(df2["tp"], bins=edges, labels=edges[:-1], include_lowest=True)
            vp     = df2.groupby("bin", observed=True)["volume"].sum()
            poc    = float(vp.idxmax())
            srt    = vp.sort_values(ascending=False)
            va     = srt[srt.cumsum() <= vp.sum() * 0.682].index
            vah    = float(va.max()) if len(va) else poc
            val    = float(va.min()) if len(va) else poc
            close  = float(df["close"].iloc[-1])
            noise  = poc * 0.015
            if   close > vah * 1.005:       sig = "breakout"
            elif close < val * 0.995:       sig = "below_value"
            elif abs(close - poc) <= noise: sig = "at_poc"
            else:                           sig = "value_area"
            return {"poc": round(poc, 2), "vah": round(vah, 2), "val": round(val, 2), "vp_signal": sig}
        except Exception:
            return {"poc": None, "vah": None, "val": None, "vp_signal": "unknown"}

    @staticmethod
    def long_trend(df_long: pd.DataFrame) -> dict:
        """
        Phân loại xu hướng dài hạn dựa trên vị trí giá vs SMA50/SMA200.

        Market Structure:
        - uptrend  : Close > SMA50 > SMA200, SMA200 không dốc xuống
        - weak_up  : Close > SMA50 nhưng SMA50 < SMA200 (đang phục hồi)
        - sideway  : Close giữa SMA50 và SMA200, không rõ hướng
        - downtrend: Close < SMA50 < SMA200 hoặc Close < SMA50 với slope âm

        Tham số slope:
        - slope50  = (SMA50 hiện tại - SMA50 cách 10 phiên) / SMA50 cách 10 phiên
        - slope200 = (SMA200 hiện tại - SMA200 cách 20 phiên) / SMA200 cách 20 phiên
        → Đo tốc độ thay đổi của đường MA để biết đang tăng hay đang chết đứng

        Fallback cho mã <200 phiên data: chỉ dùng SMA50
        """
        if df_long is None or len(df_long) < 60:
            return {"trend_long": "sideway", "sma50": None, "sma200": None}
        c       = df_long["close"]
        sma50_s = c.rolling(50).mean()
        sma50   = float(sma50_s.iloc[-1])
        close   = float(c.iloc[-1])
        past50  = float(sma50_s.iloc[-10]) if len(sma50_s) >= 10 else sma50
        slope50 = (sma50 - past50) / (past50 + 1e-9)
        dist50  = (close - sma50) / (sma50 + 1e-9)
        sma200 = None
        if len(c) >= 200:
            sma200_s = c.rolling(200).mean()
            sma200   = float(sma200_s.iloc[-1])
            past200  = float(sma200_s.iloc[-20]) if len(sma200_s) >= 20 else sma200
            slope200 = (sma200 - past200) / (past200 + 1e-9)
            if close > sma50 and close > sma200:
                trend = "uptrend" if sma50 > sma200 and slope200 > -0.005 else "weak_up"
            elif close < sma50 and close < sma200:
                trend = "downtrend"
            elif close > sma50 and close < sma200:
                trend = "weak_up" if slope50 > 0.005 and dist50 > 0.015 else "sideway"
            elif close < sma50 and close > sma200:
                trend = "downtrend" if dist50 < -0.04 or slope50 < -0.01 else "sideway"
            else:
                trend = "sideway"
        else:
            if   close > sma50: trend = "weak_up"   if slope50 > 0 else "sideway"
            elif close < sma50: trend = "downtrend"  if slope50 < 0 or dist50 < -0.04 else "sideway"
            else:               trend = "sideway"
        return {
            "trend_long": trend,
            "sma50":  round(sma50, 2),
            "sma200": round(sma200, 2) if sma200 else None,
        }

    @staticmethod
    def relative_strength(df_stk: pd.DataFrame, df_vni: pd.DataFrame, period: int = 20) -> float:
        """
        Relative Strength so với VNINDEX.

        RS = (return_stock - return_vni) / max(|return_vni|, 1%)
        → Chuẩn hóa theo biến động index để có thể so sánh cross-sector

        RS > 0 : mã mạnh hơn thị trường
        RS = 1 : mã tăng gấp đôi index
        RS < 0 : mã yếu hơn thị trường

        Dùng rolling(3) smooth để giảm nhiễu ngày cuối tháng.
        Cap tại [-10, 10] tránh outlier khi index gần 0%.
        """
        if df_stk is None or df_vni is None: return 1.0
        if len(df_stk) < period or len(df_vni) < period: return 1.0
        try:
            s_s = df_stk["close"].rolling(3).mean()
            v_s = df_vni["close"].rolling(3).mean()
            rs  = s_s.iloc[-1] / s_s.iloc[-period] - 1
            rv  = v_s.iloc[-1] / v_s.iloc[-period] - 1
            d   = max(abs(rv), 0.01)
            sgn = np.sign(rv) if abs(rv) >= 0.01 else 1.0
            return float(max(min((rs / d) * sgn, 10.0), -10.0))
        except:
            return 1.0

    @staticmethod
    def money_flow(df: pd.DataFrame) -> dict:
        """
        Ước tính dòng tiền tổ chức từ Accumulation/Distribution line.

        Thuật toán:
        1. CLV (Close Location Value) = ((C-L)-(H-C))/(H-L)
           → +1 nếu đóng tại đỉnh, -1 nếu đóng tại đáy
        2. AD line = cumsum(CLV × Volume)
           → Tăng = tiền vào, giảm = tiền ra
        3. EMA5 vs EMA20 của AD line → ngắn hạn so với trung hạn
        4. inst_score 2-9 dựa trên combo inst_up + ad_roc

        Smart Money proxy:
        - cs = (C-L)/(H-L) rolling → mức đóng cửa trung bình 5/20 phiên
        - cs5 > cs20 + cs5 > 0.55 → tiền thông minh đang đẩy giá lên
        """
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        clv    = ((c - l) - (h - c)) / (h - l + 1e-9)
        ad     = (clv * v).cumsum()
        ad5    = ad.ewm(span=5,  adjust=False).mean()
        ad20   = ad.ewm(span=20, adjust=False).mean()
        inst_up = bool(ad5.iloc[-1] > ad20.iloc[-1])
        ad_roc  = float((ad.iloc[-1] - ad.iloc[-6]) / (v.rolling(20).mean().iloc[-1] * 5 + 1e-9)) if len(df) >= 6 else 0

        if   inst_up and ad_roc > 0.3: inst_score = 9.0
        elif inst_up and ad_roc > 0:   inst_score = 7.0
        elif inst_up:                  inst_score = 5.5
        elif ad_roc > 0:               inst_score = 4.0
        else:                          inst_score = 2.0

        cs     = (c - l) / (h - l + 1e-9)
        cs5    = float(cs.rolling(5).mean().iloc[-1])
        cs20   = float(cs.rolling(20).mean().iloc[-1])
        smart  = bool(cs5 > cs20 and cs5 > 0.55)

        vol_ma = v.rolling(20).mean()
        tr     = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr    = tr.rolling(14).mean()
        vr     = v / (vol_ma + 1e-9)
        hl     = (h - l) / (atr + 1e-9)
        b_acc  = bool(vr.iloc[-1] > 2.0 and hl.iloc[-1] < 1.5 and c.iloc[-1] > c.iloc[-2])
        b_brk  = bool(vr.iloc[-1] > 2.5 and (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] > 0.01)

        n = sum([inst_up, smart, b_acc or b_brk])
        if   b_brk: label = "🏦 Tổ chức đẩy"
        elif b_acc: label = "🔍 Tích lũy lặng lẽ"
        elif n >= 2: label = "💰 Tiền lớn vào"
        elif n == 1: label = "🔄 Trung tính"
        else:        label = "📤 Tiền lớn rút"

        return {"inst_flow": inst_score, "inst_flow_up": inst_up,
                "smart_money": smart, "block_accum": b_acc, "block_break": b_brk, "mf_label": label}

    @staticmethod
    def compute_all(df: pd.DataFrame, vp: dict, trend: dict,
                    rs: float, mf: dict, vsa_signal: str) -> dict:
        """
        Tổng hợp tất cả indicators vào dict `raw`.

        Bao gồm: volume metrics, OBV, CMF, Force Index, RSI,
        EMA20/50, MACD, Stochastic, ATR, Bollinger %B,
        VP signals, Trend, RS, Money Flow, VSA (v5.2 mới).
        """
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        R = {"close": round(float(c.iloc[-1]), 2), "volume": int(v.iloc[-1])}

        # ── Volume metrics ────────────────────────────────────────────────────
        vol_ma20 = v.rolling(20).mean().iloc[-1]
        R["volume_ratio"] = round(v.iloc[-1] / vol_ma20, 2) if vol_ma20 > 0 else 0.0
        R["avg_vol_20d"]  = int(vol_ma20)
        R["avg_val_20d"]  = round(float(vol_ma20 * c.rolling(20).mean().iloc[-1]), 0)

        # ── OBV (On-Balance Volume) ───────────────────────────────────────────
        # OBV signal 1 = OBV đang tăng (EMA5 > EMA20)
        # OBV slope = tốc độ thay đổi OBV → momentum của dòng tiền
        obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
        R["obv_signal"] = 1 if obv.rolling(5).mean().iloc[-1] > obv.rolling(20).mean().iloc[-1] else -1
        if len(df) >= 6:
            y  = obv.iloc[-6:].values.astype(float)
            sl = float(np.polyfit(np.arange(6, dtype=float), y, 1)[0])
            R["obv_slope"] = round(sl / (abs(float(obv.iloc[-6])) + 1e-9) * 100, 2)
        else:
            R["obv_slope"] = 0.0

        # ── CMF (Chaikin Money Flow) ──────────────────────────────────────────
        # CMF > 0.05: tiền đang vào, CMF < -0.05: tiền đang ra
        mfv = ((c - l) - (h - c)) / (h - l + 1e-9) * v
        R["cmf"] = round(float((mfv.rolling(21).sum() / v.rolling(21).sum().replace(0, np.nan)).iloc[-1]), 4)

        # ── Force Index ───────────────────────────────────────────────────────
        # Force = close_change × volume → đo lực di chuyển
        # Chuẩn hóa theo MA20 của lực để so sánh cross-mã
        fi   = (c.diff() * v).ewm(span=13, adjust=False).mean()
        fima = fi.abs().rolling(20).mean().iloc[-1]
        R["force_index"] = round(float(fi.iloc[-1]) / (fima + 1e-9), 4)

        # ── Money Flow metrics ────────────────────────────────────────────────
        for k in ["inst_flow", "inst_flow_up", "smart_money", "block_accum", "block_break", "mf_label"]:
            R[k] = mf.get(k)

        # ── RSI (14 phiên) ────────────────────────────────────────────────────
        delta = c.diff()
        g  = delta.clip(lower=0).rolling(14).mean()
        ls = (-delta.clip(upper=0)).rolling(14).mean()
        R["rsi"] = round(float((100 - 100 / (1 + g / ls.replace(0, np.nan))).iloc[-1]), 2)

        # ── EMA 20/50 ─────────────────────────────────────────────────────────
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        R["ema_pct"] = round(float((c.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1] * 100), 2)
        R["ema20"]   = round(float(ema20.iloc[-1]), 2)
        R["ema50"]   = round(float(ema50.iloc[-1]), 2)

        # ── MACD (12, 26, 9) ──────────────────────────────────────────────────
        # macd_hist_pct: histogram chuẩn hóa theo giá (để so sánh cross-mã)
        # macd_cross_up: histogram vừa chuyển từ âm sang dương → buy signal
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        ml    = ema12 - ema26
        ms    = ml.ewm(span=9, adjust=False).mean()
        mh    = ml - ms
        R["macd_hist_pct"]  = round(float(mh.iloc[-1]) / (float(c.iloc[-1]) + 1e-9) * 100, 4)
        R["macd_cross_up"]  = bool(mh.iloc[-1] > 0 and mh.iloc[-2] <= 0)
        # Lưu macd_hist series vào raw để MomentumEngine dùng
        # (chỉ tail 40 để tránh bloat dict)
        R["_macd_hist_series"] = mh.tail(40).values.tolist()

        # ── Stochastic (14, 3) ────────────────────────────────────────────────
        # stoch_cross_up: %K vừa cắt lên %D từ dưới → signal mua
        l14 = l.rolling(14).min()
        h14 = h.rolling(14).max()
        sk  = (c - l14) / (h14 - l14 + 1e-9) * 100
        sd  = sk.rolling(3).mean()
        R["stoch_k"]        = round(float(sk.iloc[-1]), 2)
        R["stoch_d"]        = round(float(sd.iloc[-1]), 2)
        R["stoch_cross_up"] = bool(sk.iloc[-1] > sd.iloc[-1] and sk.iloc[-2] <= sd.iloc[-2])

        # ── ATR (Average True Range, 14 phiên) ───────────────────────────────
        # atr_ratio: ATR% = ATR/Close × 100 → mức biến động tương đối
        # atr_abs: giá trị ATR tuyệt đối (VNĐ/điểm) → dùng tính SL/TP
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        R["atr_ratio"] = round(float(tr.rolling(14).mean().iloc[-1]) / (float(c.iloc[-1]) + 1e-9) * 100, 2)
        R["atr_abs"]   = round(float(tr.rolling(14).mean().iloc[-1]), 2)

        # ── Bollinger Band %B ─────────────────────────────────────────────────
        # %B = 0 → tại lower band, %B = 1 → tại upper band
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        R["bb_pctb"] = round(float((c - (sma20 - 2 * std20)).iloc[-1]) / (float(4 * std20.iloc[-1]) + 1e-9), 4)

        # ── Volume Profile & Trend ────────────────────────────────────────────
        R.update({k: vp.get(k) for k in ["vp_signal", "poc", "vah", "val"]})
        R["trend_long"] = trend.get("trend_long", "sideway")
        R["sma50"]      = trend.get("sma50")
        R["sma200"]     = trend.get("sma200")
        R["rs"]         = round(float(rs), 2) if isinstance(rs, (int, float)) else 1.0


        # ── VSA (v5.2) ────────────────────────────────────────────────────────
        R["vsa_signal"] = vsa_signal

        # Lưu RSI series để MomentumEngine tìm divergence
        rsi_full = 100 - 100 / (1 + g / ls.replace(0, np.nan))
        R["_rsi_series"]   = rsi_full.tail(40).values.tolist()
        R["_close_series"] = c.tail(40).values.tolist()
        R["_low_series"]   = l.tail(40).values.tolist()
        R["_high_series"]  = h.tail(40).values.tolist()

        return R

    @staticmethod
    def dmi_aroon(df_long: pd.DataFrame, cfg: QuantConfig) -> dict:
        """
        DMI (+DI/-DI/ADX) + Aroon (Up/Down/Osc) — v5.6 NEW.

        Khác bản tham khảo ở 3 điểm:
        1. Tính trên df_long (400 phiên) thay vì df ngắn — Wilder's smoothing
           (EWM alpha=1/period) cần đủ warm-up để hội tụ, ADX là smoothed-của-
           smoothed nên hội tụ càng chậm hơn RSI/MACD thường.
        2. Aroon dùng sliding_window_view + argmax/argmin numpy thuần —
           vectorized thật, không dùng rolling().apply(lambda).
        3. Tie-breaking argmax ưu tiên đỉnh/đáy GẦN NHẤT khi trùng giá (đảo
           mảng trước khi argmax) — khớp convention các platform chart phổ biến,
           tránh lệch kết quả khi giá VN hay trùng nhau do bước giá làm tròn.
        """
        default = {
            "plus_di": 25.0, "minus_di": 25.0, "adx": 20.0,
            "aroon_up": 50.0, "aroon_down": 50.0, "aroon_osc": 0.0,
            "dmi_bullish": False, "dmi_strength": "none",
            "aroon_state": "neutral", "dmi_aroon_label": "—",
        }
        p, ap = cfg.dmi_period, cfg.aroon_period
        if df_long is None or len(df_long) < max(p * 3, ap + 1):
            return default

        h, l, c = df_long["high"], df_long["low"], df_long["close"]

        # ── DMI / ADX ─────────────────────────────────────────────────────────
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        up_move, down_move = h - h.shift(1), l.shift(1) - l
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_s  = tr.ewm(alpha=1/p, adjust=False).mean()
        pdm_s = pd.Series(pos_dm, index=df_long.index).ewm(alpha=1/p, adjust=False).mean()
        ndm_s = pd.Series(neg_dm, index=df_long.index).ewm(alpha=1/p, adjust=False).mean()

        plus_di_s  = (pdm_s / tr_s.replace(0, np.nan)) * 100
        minus_di_s = (ndm_s / tr_s.replace(0, np.nan)) * 100
        di_sum  = plus_di_s + minus_di_s
        di_diff = (plus_di_s - minus_di_s).abs()
        adx_s = ((di_diff / di_sum.replace(0, np.nan)) * 100).ewm(alpha=1/p, adjust=False).mean()

        plus_di  = float(plus_di_s.iloc[-1])  if not pd.isna(plus_di_s.iloc[-1])  else 25.0
        minus_di = float(minus_di_s.iloc[-1]) if not pd.isna(minus_di_s.iloc[-1]) else 25.0
        adx      = float(adx_s.iloc[-1])      if not pd.isna(adx_s.iloc[-1])      else 20.0

        # ── Aroon (vectorized, tie-break ưu tiên đỉnh/đáy gần nhất) ──────────
        window = ap + 1
        h_arr, l_arr = h.values.astype(float), l.values.astype(float)
        if len(h_arr) >= window:
            hw_rev = sliding_window_view(h_arr, window)[:, ::-1]  # cột 0 = phiên gần nhất
            lw_rev = sliding_window_view(l_arr, window)[:, ::-1]
            days_since_high = hw_rev.argmax(axis=1)[-1]
            days_since_low  = lw_rev.argmin(axis=1)[-1]
            aroon_up   = ((ap - days_since_high) / ap) * 100
            aroon_down = ((ap - days_since_low)  / ap) * 100
        else:
            aroon_up = aroon_down = 50.0
        aroon_osc = aroon_up - aroon_down

        # ── Tags ──────────────────────────────────────────────────────────────
        dmi_bullish = bool(plus_di > minus_di)
        if   adx >= cfg.adx_trend_min: dmi_strength = "strong"
        elif adx >= cfg.adx_weak_min:  dmi_strength = "weak"
        else:                          dmi_strength = "none"

        if   aroon_up >= 70 and aroon_down <= 30: aroon_state = "strong_bull"
        elif aroon_down >= 70 and aroon_up <= 30: aroon_state = "strong_bear"
        elif aroon_osc > 50:                      aroon_state = "emerging_bull"
        elif aroon_osc < -50:                     aroon_state = "emerging_bear"
        else:                                     aroon_state = "neutral"

        dir_icon = "🟢" if dmi_bullish else "🔴"
        str_txt  = {"strong": "Mạnh", "weak": "Yếu", "none": "Sideway"}[dmi_strength]
        aroon_txt = {"strong_bull": "Aroon Bull mạnh", "strong_bear": "Aroon Bear mạnh",
                     "emerging_bull": "Aroon Bull mới", "emerging_bear": "Aroon Bear mới",
                     "neutral": "Aroon trung tính"}[aroon_state]

        return {
            "plus_di": round(plus_di, 1), "minus_di": round(minus_di, 1), "adx": round(adx, 1),
            "aroon_up": round(aroon_up, 1), "aroon_down": round(aroon_down, 1),
            "aroon_osc": round(aroon_osc, 1),
            "dmi_bullish": dmi_bullish, "dmi_strength": dmi_strength,
            "aroon_state": aroon_state,
            "dmi_aroon_label": f"{dir_icon} DMI {str_txt} | {aroon_txt}",
        }

    @staticmethod
    def ma_signals(df: pd.DataFrame, df_long: pd.DataFrame, cfg: QuantConfig) -> dict:
        """
        Tín hiệu MA ngắn hạn (9/10) vs MA dài hạn (50/200) — v5.5 NEW.

        - lost_short_trend: đóng cửa dưới CẢ MA9 VÀ MA10 → cảnh báo sớm mất
          xu hướng tăng ngắn hạn (nhạy hơn EMA20 đang dùng trong ScoringEngine)
        - touch_ma_support: low 3 phiên gần nhất chạm MA50 hoặc MA200 rồi
          đóng cửa bật lên trên → hỗ trợ dài hạn còn hiệu lực
        """
        result = {
            "ma9": None, "ma10": None, "sma50_chk": None, "sma200_chk": None,
            "lost_short_trend": False,
            "touch_ma_support":  False,
            "ma_support_level":  None,
            "ma_signal_label":   "—",
        }
        try:
            c = df["close"]
            if len(c) >= cfg.ma_short_period_2:
                ma9  = c.rolling(cfg.ma_short_period_1).mean().iloc[-1]
                ma10 = c.rolling(cfg.ma_short_period_2).mean().iloc[-1]
                close_now = float(c.iloc[-1])
                result["ma9"], result["ma10"] = round(float(ma9), 2), round(float(ma10), 2)
                result["lost_short_trend"] = bool(close_now < ma9 and close_now < ma10)
        except Exception:
            pass

        try:
            if df_long is not None and len(df_long) >= 55:
                cl, l = df_long["close"], df_long["low"]
                close_now  = float(cl.iloc[-1])
                sma50      = float(cl.rolling(50).mean().iloc[-1])
                sma200     = float(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else None
                tol        = cfg.ma_touch_tolerance_pct / 100
                low_recent = float(l.tail(3).min())

                result["sma50_chk"]  = round(sma50, 2)
                result["sma200_chk"] = round(sma200, 2) if sma200 else None

                touched_50  = bool(low_recent <= sma50 * (1 + tol) and close_now > sma50)
                touched_200 = bool(sma200 and low_recent <= sma200 * (1 + tol) and close_now > sma200)

                if touched_50 or touched_200:
                    result["touch_ma_support"] = True
                    result["ma_support_level"] = round(sma200 if touched_200 else sma50, 2)

                lost = result["lost_short_trend"]
                if lost and (touched_50 or touched_200):
                    result["ma_signal_label"] = "⚠️ Mất MA9/10 nhưng test MA50/200"
                elif lost:
                    result["ma_signal_label"] = "🔴 Mất MA9/10 — hết xu hướng ngắn"
                elif touched_200:
                    result["ma_signal_label"] = "🟢 Test MA200 — hỗ trợ dài hạn"
                elif touched_50:
                    result["ma_signal_label"] = "🟢 Test MA50 — hỗ trợ trung hạn"
        except Exception:
            pass

        return result

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5b — MOMENTUM ENGINE  (NEW v5.3)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Feature Engineering tách biệt: tính BB, Divergence, Hidden Accumulation.
# Output merge vào raw dict trước khi vào ScoringEngine/SignalBuilder.
#
# THIẾT KẾ:
#   1. BB & Squeeze: bandwidth percentile-based, không hardcode threshold
#   2. Divergence: fractal swing pivot detection → so sánh giá vs indicator
#      - Dùng lookback 15 nến cho VN (market ít nến hơn, nhiễu nhiều hơn)
#      - Min 5 nến giữa 2 pivot để tránh false pivot
#      - Chỉ confirm nếu 2 pivot đủ rõ ràng (chênh > 0.3%)
#   3. Hidden Accumulation: giá flat/giảm nhẹ nhưng CMF đang tăng
#      → Tiền lớn đang gom âm thầm không để lộ qua giá

class MomentumEngine:
    """
    Feature Engineering cho Momentum + Volatility + Divergence.

    Tách biệt hoàn toàn khỏi SignalBuilder — chỉ tính toán và trả về
    dict feature để nhét vào `raw`. SignalBuilder không làm toán ở đây.

    Output keys:
    - bb_upper, bb_mid, bb_lower, bb_bandwidth
    - is_bb_squeeze (bool)
    - rsi_bull_div, rsi_bear_div (bool)
    - macd_bull_div, macd_bear_div (bool)
    - hidden_accumulation (bool)
    - vol_dry_up (bool) — định nghĩa nhanh 3 phiên, bổ sung VolumeDryupDetector
    - div_label (str) — mô tả trạng thái divergence để hiển thị
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    # ── Helpers: tìm Swing Pivot ──────────────────────────────────────────────
    @staticmethod
    def _find_swing_lows(series: np.ndarray, lookback: int = 5, min_bars: int = 5) -> list:
        """
        Tìm danh sách (index, value) của các Swing Low trong series.

        Định nghĩa Swing Low: giá trị tại i thấp hơn tất cả trong cửa sổ
        [i-lookback, i+lookback] — fractal pivot chắc chắn hơn so sánh ±2 nến.

        Parameters
        ----------
        series   : mảng giá (close / low)
        lookback : bán kính cửa sổ kiểm tra (default 5, VN dùng từ cfg.div_lookback)
        min_bars : khoảng cách tối thiểu giữa 2 pivot để tránh noise

        Returns
        -------
        list[(index, value)] — sắp xếp theo index tăng dần
        """
        pivots = []
        n = len(series)
        for i in range(lookback, n - lookback):
            window = series[max(0, i - lookback): i + lookback + 1]
            # Pivot Low: giá trị tại i là min của cả cửa sổ
            if series[i] == window.min():
                # Kiểm tra khoảng cách với pivot trước
                if pivots and (i - pivots[-1][0]) < min_bars:
                    # Nếu gần hơn min_bars, giữ pivot thấp hơn
                    if series[i] < pivots[-1][1]:
                        pivots[-1] = (i, series[i])
                else:
                    pivots.append((i, series[i]))
        return pivots

    @staticmethod
    def _find_swing_highs(series: np.ndarray, lookback: int = 5, min_bars: int = 5) -> list:
        """
        Tìm danh sách (index, value) của các Swing High trong series.
        Logic đối xứng với _find_swing_lows.
        """
        pivots = []
        n = len(series)
        for i in range(lookback, n - lookback):
            window = series[max(0, i - lookback): i + lookback + 1]
            if series[i] == window.max():
                if pivots and (i - pivots[-1][0]) < min_bars:
                    if series[i] > pivots[-1][1]:
                        pivots[-1] = (i, series[i])
                else:
                    pivots.append((i, series[i]))
        return pivots

    def _detect_divergence(
        self,
        price_series: np.ndarray,
        indicator_series: np.ndarray,
        mode: str = "bullish",          # "bullish" hoặc "bearish"
        min_price_chg_pct: float = 0.3, # Chênh lệch tối thiểu giữa 2 pivot (%)
    ) -> bool:
        """
        Phát hiện phân kỳ (Divergence) giữa giá và indicator.

        Bullish Divergence (Phân kỳ dương — tín hiệu MUA):
          - Giá tạo đáy MỚI THẤP HƠN (Lower Low)
          - Indicator tại đáy đó CAO HƠN (Higher Low)
          → Lực bán đang yếu đi, chuẩn bị đảo chiều

        Bearish Divergence (Phân kỳ âm — tín hiệu BÁN):
          - Giá tạo đỉnh MỚI CAO HƠN (Higher High)
          - Indicator tại đỉnh đó THẤP HƠN (Lower High)
          → Momentum đang suy yếu dù giá còn tăng

        Điều kiện đủ:
          1. Tìm được ít nhất 2 pivot gần nhất
          2. Chênh lệch giá giữa 2 pivot > min_price_chg_pct%
             (tránh noise khi giá gần bằng nhau)
          3. Chỉ xét pivot trong 2/3 cuối series (tránh pivot quá cũ)
          4. Pivot thứ 2 (gần hơn) phải trong 10 nến cuối để còn relevant

        Parameters
        ----------
        price_series     : mảng giá (low cho bullish, high cho bearish)
        indicator_series : mảng RSI hoặc MACD Histogram
        mode             : "bullish" hoặc "bearish"
        min_price_chg_pct: ngưỡng chênh lệch tối thiểu giữa 2 pivot

        Returns
        -------
        bool : True nếu phát hiện divergence, False nếu không
        """
        if len(price_series) < self.cfg.div_lookback * 2 + 2:
            return False
        if len(price_series) != len(indicator_series):
            return False

        # Làm sạch NaN
        valid_mask = ~(np.isnan(price_series) | np.isnan(indicator_series))
        if valid_mask.sum() < self.cfg.div_lookback * 2:
            return False

        lookback  = max(3, self.cfg.div_lookback // 3)  # Bán kính pivot ngắn hơn div_lookback
        min_bars  = self.cfg.div_min_bars

        # Chỉ xét 2/3 cuối để tránh pivot lịch sử quá cũ
        start_idx = len(price_series) // 3
        price_win = price_series[start_idx:]
        ind_win   = indicator_series[start_idx:]

        try:
            if mode == "bullish":
                # Tìm Swing Low của giá
                pivots = self._find_swing_lows(price_win, lookback, min_bars)
            else:
                # Tìm Swing High của giá
                pivots = self._find_swing_highs(price_win, lookback, min_bars)
        except Exception:
            return False

        # Cần ít nhất 2 pivot
        if len(pivots) < 2:
            return False

        # Lấy 2 pivot gần nhất
        p1_idx, p1_val = pivots[-2]
        p2_idx, p2_val = pivots[-1]

        # Pivot thứ 2 phải đủ gần (trong 10 nến cuối)
        if (len(price_win) - 1 - p2_idx) > 10:
            return False

        # Chênh lệch giá tối thiểu để tránh noise
        price_chg_pct = abs(p2_val - p1_val) / (abs(p1_val) + 1e-9) * 100
        if price_chg_pct < min_price_chg_pct:
            return False

        # Lấy giá trị indicator tại 2 pivot
        ind1 = ind_win[p1_idx]
        ind2 = ind_win[p2_idx]

        # Kiểm tra NaN indicator tại pivot
        if np.isnan(ind1) or np.isnan(ind2):
            return False

        if mode == "bullish":
            # Giá Lower Low + Indicator Higher Low → Phân kỳ dương
            return bool(p2_val < p1_val and ind2 > ind1)
        else:
            # Giá Higher High + Indicator Lower High → Phân kỳ âm
            return bool(p2_val > p1_val and ind2 < ind1)

    def compute(self, df: pd.DataFrame, raw: dict) -> dict:
        """
        Tính toàn bộ momentum features và trả về dict.

        Parameters
        ----------
        df  : DataFrame OHLCV
        raw : dict indicators đã tính từ IndicatorEngine.compute_all()
              (dùng _rsi_series, _macd_hist_series, _close_series, ...)

        Returns
        -------
        dict với tất cả momentum features để merge vào raw
        """
        result = {
            # Defaults — đảm bảo không có KeyError phía downstream
            "bb_upper":           None,
            "bb_mid":             None,
            "bb_lower":           None,
            "bb_bandwidth":       None,
            "is_bb_squeeze":      False,
            "rsi_bull_div":       False,
            "rsi_bear_div":       False,
            "macd_bull_div":      False,
            "macd_bear_div":      False,
            "hidden_accumulation": False,
            "vol_dry_up":         False,
            "div_label":          "—",
        }

        if df is None or len(df) < self.cfg.bb_period + 5:
            return result

        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # ── 1. Bollinger Bands ────────────────────────────────────────────────
        # SMA20 ± 2 std — công thức chuẩn John Bollinger
        try:
            period = self.cfg.bb_period
            std_m  = self.cfg.bb_std
            sma    = c.rolling(period).mean()
            std    = c.rolling(period).std()
            upper  = sma + std_m * std
            lower  = sma - std_m * std
            bw     = (upper - lower) / (sma + 1e-9)  # Bandwidth chuẩn hóa theo giá

            result["bb_upper"] = round(float(upper.iloc[-1]), 2)
            result["bb_mid"]   = round(float(sma.iloc[-1]),   2)
            result["bb_lower"] = round(float(lower.iloc[-1]), 2)
            result["bb_bandwidth"] = round(float(bw.iloc[-1]), 4)

            # ── BB Squeeze: bandwidth hiện tại < percentile 15% của 60 phiên ──
            # Percentile-based thay vì hardcode threshold vì mỗi mã có
            # mức volatility nền khác nhau (blue-chip vs penny stock khác nhau)
            lookback_sq = min(self.cfg.bb_squeeze_lookback, len(bw.dropna()))
            if lookback_sq >= 20:
                bw_window = bw.dropna().tail(lookback_sq).values
                threshold = np.percentile(bw_window, self.cfg.bb_squeeze_pct)
                result["is_bb_squeeze"] = bool(float(bw.iloc[-1]) < threshold)
        except Exception:
            pass  # Giữ default False, không raise

        # ── 2. Divergence RSI ─────────────────────────────────────────────────
        # Dùng _rsi_series và _low_series/_high_series đã được lưu trong raw
        # bởi IndicatorEngine để tránh tính lại RSI
        try:
            rsi_arr   = np.array(raw.get("_rsi_series", []), dtype=float)
            low_arr   = np.array(raw.get("_low_series",  []), dtype=float)
            high_arr  = np.array(raw.get("_high_series", []), dtype=float)

            if len(rsi_arr) >= self.cfg.div_lookback * 2:
                result["rsi_bull_div"] = self._detect_divergence(
                    low_arr, rsi_arr, mode="bullish"
                )
                result["rsi_bear_div"] = self._detect_divergence(
                    high_arr, rsi_arr, mode="bearish"
                )
        except Exception:
            pass

        # ── 3. Divergence MACD Histogram ─────────────────────────────────────
        # MACD Histogram phân kỳ nhạy hơn RSI, confirm sớm hơn
        # Dùng _macd_hist_series đã lưu trong raw
        try:
            mh_arr  = np.array(raw.get("_macd_hist_series", []), dtype=float)
            low_arr = np.array(raw.get("_low_series",  []), dtype=float)
            high_arr= np.array(raw.get("_high_series", []), dtype=float)

            if len(mh_arr) >= self.cfg.div_lookback * 2:
                result["macd_bull_div"] = self._detect_divergence(
                    low_arr, mh_arr, mode="bullish"
                )
                result["macd_bear_div"] = self._detect_divergence(
                    high_arr, mh_arr, mode="bearish"
                )
        except Exception:
            pass

        # ── 4. Hidden Accumulation ────────────────────────────────────────────
        # Điều kiện: giá không tăng (sideway/giảm nhẹ trong 5 phiên)
        #            nhưng CMF đang tăng và dương
        # → Tiền lớn đang gom mà không để giá tăng để mua thêm
        try:
            if len(c) >= 6:
                price_5d_chg  = (float(c.iloc[-1]) - float(c.iloc[-6])) / (float(c.iloc[-6]) + 1e-9)
                cmf_now       = float(raw.get("cmf", 0))
                # Tính CMF 5 phiên trước để so sánh slope
                mfv_s = ((c - l) - (h - c)) / (h - l + 1e-9) * v
                cmf_5d_ago = float(
                    (mfv_s.rolling(21).sum() / v.rolling(21).sum().replace(0, np.nan)).iloc[-6]
                ) if len(df) >= 27 else 0.0

                cmf_rising   = (cmf_now - cmf_5d_ago) > self.cfg.hidden_acc_cmf_min
                price_flat   = -0.03 <= price_5d_chg <= 0.01  # Giá giảm ít hoặc flat
                cmf_positive = cmf_now > 0

                result["hidden_accumulation"] = bool(price_flat and cmf_rising and cmf_positive)
        except Exception:
            pass

        # ── 5. Vol Dry-up nhanh (3 phiên) ────────────────────────────────────
        # Bổ sung cho VolumeDryupDetector (7 phiên) — xác nhận nhanh hơn
        # Không replace, dùng song song: 3 phiên = setup entry gấp hơn
        try:
            if len(v) >= 23:
                vol_ma20   = float(v.rolling(20).mean().iloc[-1])
                vol_avg_3d = float(v.iloc[-3:].mean())
                result["vol_dry_up"] = bool(vol_avg_3d < vol_ma20 * 0.50)
        except Exception:
            pass

        # ── 6. Divergence Label tổng hợp ─────────────────────────────────────
        try:
            rbd = result["rsi_bull_div"]
            mbd = result["macd_bull_div"]
            rrd = result["rsi_bear_div"]
            mrd = result["macd_bear_div"]

            if rbd and mbd:    result["div_label"] = "💚 Phân kỳ dương kép (RSI+MACD)"
            elif rbd:          result["div_label"] = "🟢 RSI Phân kỳ dương"
            elif mbd:          result["div_label"] = "🟩 MACD Phân kỳ dương"
            elif rrd and mrd:  result["div_label"] = "🔴 Phân kỳ âm kép (RSI+MACD)"
            elif rrd:          result["div_label"] = "🟠 RSI Phân kỳ âm"
            elif mrd:          result["div_label"] = "🟡 MACD Phân kỳ âm"
            elif result["hidden_accumulation"]: result["div_label"] = "🔍 Tích lũy ẩn (CMF↑)"
            elif result["is_bb_squeeze"]:       result["div_label"] = "🗜 BB Squeeze — Chờ nổ"
            else:              result["div_label"] = "—"
        except Exception:
            pass

        return result

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5c — ICHIMOKU ENGINE  (NEW v5.4)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Ichimoku Kinko Hyo — "Nhìn thoáng qua thấy cân bằng"
#
# PHONG CÁCH TRỊNH PHÁT — Multi-Timeframe (Ngày / Tuần / Tháng):
#   - Không chỉ nhìn Daily, phải xem W và M để biết S/R "thực sự"
#   - Vùng Kumo W/M là "tường thành" — giá vượt qua hoặc nằm dưới là quyết định
#   - Kijun Daily = điểm cân bằng ngắn hạn → entry zone tối ưu khi pullback
#   - Kijun Weekly = target tự nhiên cho T+ (thường cách 8-15% từ vùng hỗ trợ)
#   - 3TF đều bullish (giá > Kumo D/W/M) = "Triple confirmation" = mua mạnh
#
# 7 TÍN HIỆU CHÍNH:
#   1. above_cloud_d/w/m  : giá trên Kumo theo từng TF
#   2. below_cloud_d/w/m  : giá dưới Kumo → tránh mua
#   3. in_cloud_d         : giá đang trong mây → vùng không rõ ràng, né
#   4. tenkan_cross_up    : Tenkan cắt lên Kijun Daily → tín hiệu mua ngắn
#   5. strong_cloud_d     : SpanA > SpanB → mây xanh = uptrend
#   6. cloud_thickness_pct: % dày mây → S/R mạnh hay yếu
#   7. chikou_above       : Chikou Span > giá 26 phiên trước → confirm uptrend
#
# S/R TỔ HỢP 3TF (hội tụ = mạnh hơn):
#   Lấy tất cả levels Kijun + SpanA + SpanB từ D/W/M
#   → Merge những level gần nhau (< 1.5%) thành 1 vùng
#   → Sắp xếp theo vị trí so với giá hiện tại → support/resist list
#
# TÍNH WEEKLY/MONTHLY:
#   - Resample OHLCV ngày thành tuần/tháng trực tiếp từ df long
#   - Dùng agg: open=first, high=max, low=min, close=last, volume=sum
#   - Ichimoku W/M tính trên dữ liệu đã resample (cần ≥52 tuần cho W,
#     ≥6 tháng cho M)

class IchimokuEngine:
    """
    Ichimoku Kinko Hyo Multi-Timeframe — Daily, Weekly, Monthly.

    Phong cách Trịnh Phát: hội tụ S/R đa khung thời gian là điểm vào/ra
    tối ưu. Mây Tuần và Tháng là "tường thành" — quan trọng hơn Daily.

    Output: dict với tất cả levels + signals để nhét vào raw dict.
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    # ── Core: tính Ichimoku trên 1 DataFrame OHLCV ────────────────────────────
    def _calc_ichimoku(self, df: pd.DataFrame,
                       t: int = 9, k: int = 26, sb: int = 52) -> dict:
        """
        Tính 5 đường Ichimoku chuẩn trên DataFrame bất kỳ (D/W/M).

        Parameters: t=Tenkan, k=Kijun, sb=Senkou B period.
        Returns dict với các giá trị tại phiên gần nhất (iloc[-1]).

        Lưu ý về shift:
        - Senkou A/B được shift(k) về tương lai để tạo mây.
        - Để lấy mây "hiện tại" (tương ứng với giá hôm nay),
          ta đọc giá trị iloc[-1] của series đã shift — pandas shift
          điền NaN ở cuối khi shift dương, nên ta cần đọc iloc[-k-1]
          của series CHƯA shift. Đây là cách đúng để tránh off-by-one.
        """
        if df is None or len(df) < 2:
            return {}

        h, l, c = df["high"], df["low"], df["close"]

        # Tenkan-sen (chuyển đổi — 9 chu kỳ)
        tenkan_s = (h.rolling(t).max() + l.rolling(t).min()) / 2

        # Kijun-sen (cơ sở — 26 chu kỳ)
        kijun_s  = (h.rolling(k).max() + l.rolling(k).min()) / 2

        # Senkou Span A = (Tenkan + Kijun) / 2, shift k về tương lai
        # Để lấy mây hiện tại: đọc iloc[-k-1] của series chưa shift
        span_a_s  = (tenkan_s + kijun_s) / 2

        # Senkou Span B = (max52 + min52) / 2, shift k về tương lai
        span_b_s   = (h.rolling(sb).max() + l.rolling(sb).min()) / 2

        # Vị trí đọc mây hiện tại (phải lùi k phiên do mây đi trước k phiên)
        target_idx = -k - 1

        # --- FIX LỖI NaN TẠI ĐÂY ---
        # Kiểm tra nếu đủ chiều dài và vị trí lùi về không bị NaN
        if len(span_a_s) >= abs(target_idx) and not pd.isna(span_a_s.iloc[target_idx]):
            span_a_now = float(span_a_s.iloc[target_idx])
        else:
            # Fallback: lấy giá trị tính toán được gần nhất (cuối cùng)
            valid_a = span_a_s.dropna()
            span_a_now = float(valid_a.iloc[-1]) if not valid_a.empty else float(c.iloc[-1])

        if len(span_b_s) >= abs(target_idx) and not pd.isna(span_b_s.iloc[target_idx]):
            span_b_now = float(span_b_s.iloc[target_idx])
        else:
            # Fallback tương tự cho Span B
            valid_b = span_b_s.dropna()
            span_b_now = float(valid_b.iloc[-1]) if not valid_b.empty else float(c.iloc[-1])
        # ---------------------------

        # Chikou Span = Close so với giá 26 phiên trước
        chikou_price_26ago = float(c.iloc[target_idx]) if len(c) >= abs(target_idx) else float(c.iloc[0])

        tenkan_now = float(tenkan_s.iloc[-1]) if not pd.isna(tenkan_s.iloc[-1]) else float(c.iloc[-1])
        kijun_now  = float(kijun_s.iloc[-1]) if not pd.isna(kijun_s.iloc[-1]) else float(c.iloc[-1])
        close_now  = float(c.iloc[-1])

        # Vị thế giá vs Kumo
        kumo_top    = max(span_a_now, span_b_now)
        kumo_bottom = min(span_a_now, span_b_now)
        above_cloud = bool(close_now > kumo_top)
        below_cloud = bool(close_now < kumo_bottom)
        in_cloud    = not above_cloud and not below_cloud

        # Độ dày Kumo (%) — càng dày, S/R càng mạnh
        cloud_thick_pct = round(abs(span_a_now - span_b_now) / (kumo_bottom + 1e-9) * 100, 2)

        # Màu mây: SpanA > SpanB = xanh (bullish), SpanA < SpanB = đỏ (bearish)
        strong_cloud = bool(span_a_now > span_b_now)

        return {
            "tenkan":           round(tenkan_now,  2),
            "kijun":            round(kijun_now,   2),
            "span_a":           round(span_a_now,  2),
            "span_b":           round(span_b_now,  2),
            "kumo_top":         round(kumo_top,    2),
            "kumo_bottom":      round(kumo_bottom, 2),
            "above_cloud":      above_cloud,
            "below_cloud":      below_cloud,
            "in_cloud":         in_cloud,
            "cloud_thick_pct":  cloud_thick_pct,
            "strong_cloud":     strong_cloud,
            "chikou_above":     bool(close_now > chikou_price_26ago),
            "tenkan_above_kijun": bool(tenkan_now > kijun_now),
        }

    # ── Resample ngày → tuần/tháng ────────────────────────────────────────────
    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, freq: str) -> Optional[pd.DataFrame]:
        """
        Resample OHLCV ngày thành Weekly ('W') hoặc Monthly ('MS').

        Dùng 'time' column làm index. Cần ít nhất:
        - Weekly: 60 tuần (~280 ngày giao dịch)
        - Monthly: 7 tháng (~140 ngày giao dịch)

        Returns None nếu không đủ dữ liệu.
        """
        if df is None or len(df) < 5:
            return None
        try:
            df2 = df.copy()
            df2["time"] = pd.to_datetime(df2["time"])
            df2 = df2.set_index("time").sort_index()
            resampled = df2.resample(freq).agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["close"])
            # Loại tuần/tháng chưa hoàn chỉnh (nến chưa đóng)
            resampled = resampled.iloc[:-1] if len(resampled) > 1 else resampled
            return resampled.reset_index() if len(resampled) >= 10 else None
        except Exception:
            return None

    # ── Merge S/R levels gần nhau ─────────────────────────────────────────────
    def _merge_levels(self, levels: list, merge_pct: float = 1.5) -> list:
        """
        Gộp các levels gần nhau thành 1 vùng (cluster).

        Nếu 2 levels cách nhau < merge_pct%, gộp thành trung bình có trọng số
        (level xuất hiện nhiều TF → trọng số cao hơn).

        Returns danh sách levels đã merge, sắp xếp tăng dần.
        """
        if not levels:
            return []
        # levels = [(price, tf_count), ...]
        sorted_lvl = sorted(levels, key=lambda x: x[0])
        merged = []
        group  = [sorted_lvl[0]]

        for lvl in sorted_lvl[1:]:
            last = group[-1]
            # Kiểm tra khoảng cách với phần tử cuối group
            if abs(lvl[0] - last[0]) / (last[0] + 1e-9) * 100 <= merge_pct:
                group.append(lvl)
            else:
                # Tổng hợp group: weighted average theo tf_count
                total_w = sum(g[1] for g in group)
                avg_p   = sum(g[0] * g[1] for g in group) / total_w
                merged.append((round(avg_p, 2), total_w))
                group = [lvl]

        # Flush group cuối
        total_w = sum(g[1] for g in group)
        avg_p   = sum(g[0] * g[1] for g in group) / total_w
        merged.append((round(avg_p, 2), total_w))
        return sorted(merged, key=lambda x: x[0])

    # ── Main entry point ──────────────────────────────────────────────────────
    def compute(self, df_short: pd.DataFrame, df_long: pd.DataFrame) -> dict:
        """
        Tính Ichimoku cho 3 TF và tổng hợp S/R levels.

        Parameters
        ----------
        df_short : DataFrame OHLCV ngắn hạn (40 phiên) — dùng cho Daily
        df_long  : DataFrame OHLCV dài hạn (280 phiên) — dùng để resample W/M

        Returns
        -------
        dict với tất cả Ichimoku features để merge vào raw.
        """
        result = {
            # Daily
            "ichi_tenkan_d":      None, "ichi_kijun_d":    None,
            "ichi_span_a_d":      None, "ichi_span_b_d":   None,
            "ichi_above_cloud_d": False,"ichi_below_cloud_d": False,
            "ichi_in_cloud_d":    False,"ichi_strong_cloud_d": False,
            "ichi_cloud_thick_d": 0.0, "ichi_chikou_above_d": False,
            "ichi_tk_cross_up_d": False,
            # Weekly
            "ichi_kijun_w":       None, "ichi_span_a_w":   None,
            "ichi_span_b_w":      None,
            "ichi_above_cloud_w": False,"ichi_below_cloud_w": False,
            "ichi_strong_cloud_w":False,
            # Monthly
            "ichi_kijun_m":       None, "ichi_span_a_m":   None,
            "ichi_span_b_m":      None,
            "ichi_above_cloud_m": False,"ichi_below_cloud_m": False,
            "ichi_strong_cloud_m":False,
            # Tổng hợp
            "ichi_bull_3tf":      False,  # Tất cả 3TF đều bullish
            "ichi_bear_3tf":      False,  # Tất cả 3TF đều bearish
            "ichi_support_1":     None,   # Hỗ trợ mạnh nhất (gần giá nhất)
            "ichi_support_2":     None,   # Hỗ trợ thứ 2
            "ichi_support_3":     None,   # Hỗ trợ thứ 3 (xa hơn)
            "ichi_resist_1":      None,   # Kháng cự gần nhất
            "ichi_resist_2":      None,   # Kháng cự thứ 2
            "ichi_resist_3":      None,   # Kháng cự thứ 3
            "ichi_label":         "—",    # Mô tả tổng quan
            "ichi_score":         5.0,    # Sub-score để đưa vào ScoringEngine
        }

        t = self.cfg.ichi_tenkan
        k = self.cfg.ichi_kijun
        s = self.cfg.ichi_senkou_b

        # ── 1. Daily Ichimoku ─────────────────────────────────────────────────
        # Dùng df_long để có đủ 52+ phiên cho Senkou B
        d = self._calc_ichimoku(df_long, t, k, s)
        if d:
            result.update({
                "ichi_tenkan_d":      d["tenkan"],
                "ichi_kijun_d":       d["kijun"],
                "ichi_span_a_d":      d["span_a"],
                "ichi_span_b_d":      d["span_b"],
                "ichi_above_cloud_d": d["above_cloud"],
                "ichi_below_cloud_d": d["below_cloud"],
                "ichi_in_cloud_d":    d["in_cloud"],
                "ichi_strong_cloud_d":d["strong_cloud"],
                "ichi_cloud_thick_d": d["cloud_thick_pct"],
                "ichi_chikou_above_d":d["chikou_above"],
                "ichi_tk_cross_up_d": d["tenkan_above_kijun"],
            })

        # ── 2. Weekly Ichimoku ────────────────────────────────────────────────
        # Resample df_long → weekly, tính Ichimoku với tham số chuẩn (9/26/52)
        # Weekly cần ≥ 52+26 = 78 tuần → df_long 280 ngày chỉ ~56 tuần
        # Giảm xuống 4/13/26 cho Weekly (giữ nguyên tỷ lệ)
        # Trịnh Phát dùng 9/26/52 ngày tương đương ~2/5/10 tuần
        df_w = self._resample_ohlcv(df_long, "W-FRI")  # Tuần kết thúc thứ Sáu
        if df_w is not None and len(df_w) >= 26:
            # Dùng 9/26/52 trên weekly data (khoảng 9 tuần = ~2 tháng)
            tw, kw, sw = 9, 26, min(52, len(df_w) - 1)
            w = self._calc_ichimoku(df_w, tw, kw, sw)
            if w:
                result.update({
                    "ichi_kijun_w":       w["kijun"],
                    "ichi_span_a_w":      w["span_a"],
                    "ichi_span_b_w":      w["span_b"],
                    "ichi_above_cloud_w": w["above_cloud"],
                    "ichi_below_cloud_w": w["below_cloud"],
                    "ichi_strong_cloud_w":w["strong_cloud"],
                })

        # ── 3. Monthly Ichimoku ───────────────────────────────────────────────
        # Monthly: 280 ngày ≈ 13-14 tháng — đủ cho Ichimoku ngắn (6/13/26M)
        # Tham số gốc 9/26/52 ngày ~ 0.4/1.2/2.4 tháng → quá ngắn
        # Dùng 3/6/12 tháng (gần chuẩn của Trịnh Phát cho khung tháng)
        df_m = self._resample_ohlcv(df_long, "MS")  # Month Start
        if df_m is not None and len(df_m) >= 13:
            tm, km, sm = 3, 6, min(12, len(df_m) - 1)
            m = self._calc_ichimoku(df_m, tm, km, sm)
            if m:
                result.update({
                    "ichi_kijun_m":       m["kijun"],
                    "ichi_span_a_m":      m["span_a"],
                    "ichi_span_b_m":      m["span_b"],
                    "ichi_above_cloud_m": m["above_cloud"],
                    "ichi_below_cloud_m": m["below_cloud"],
                    "ichi_strong_cloud_m":m["strong_cloud"],
                })

        # ── 4. Triple Confirmation ────────────────────────────────────────────
        # Bull 3TF: giá trên Kumo cả 3 TF → xác nhận uptrend toàn diện
        bull_3tf = (result["ichi_above_cloud_d"] and
                    result["ichi_above_cloud_w"] and
                    result["ichi_above_cloud_m"])
        bear_3tf = (result["ichi_below_cloud_d"] and
                    result["ichi_below_cloud_w"] and
                    result["ichi_below_cloud_m"])
        result["ichi_bull_3tf"] = bull_3tf
        result["ichi_bear_3tf"] = bear_3tf

        # ── 5. Tổng hợp S/R levels từ 3TF ────────────────────────────────────
        # Thu thập tất cả levels có ý nghĩa (Kijun + SpanA + SpanB của D/W/M)
        # tf_count = số TF đề cập đến level đó → dùng làm trọng số merge
        close = float(df_long["close"].iloc[-1]) if df_long is not None else 0
        all_levels = []

        def _add(val, tf_w):
            if val and not np.isnan(val) and val > 0:
                all_levels.append((float(val), tf_w))

        # Daily: weight 1
        _add(result.get("ichi_kijun_d"),  1)
        _add(result.get("ichi_span_a_d"), 1)
        _add(result.get("ichi_span_b_d"), 1)
        # Weekly: weight 2 (quan trọng hơn Daily)
        _add(result.get("ichi_kijun_w"),  2)
        _add(result.get("ichi_span_a_w"), 2)
        _add(result.get("ichi_span_b_w"), 2)
        # Monthly: weight 3 (quan trọng nhất)
        _add(result.get("ichi_kijun_m"),  3)
        _add(result.get("ichi_span_a_m"), 3)
        _add(result.get("ichi_span_b_m"), 3)

        if all_levels:
            merged = self._merge_levels(all_levels, self.cfg.ichi_cloud_merge_pct)
            # Phân loại support (< giá) và resist (> giá)
            supports = [(p, w) for p, w in merged if p < close * 0.999]
            resists  = [(p, w) for p, w in merged if p > close * 1.001]
            # Hỗ trợ: lấy gần nhất trước (sort giảm dần)
            supports_sorted = sorted(supports, key=lambda x: x[0], reverse=True)
            # Kháng cự: lấy gần nhất trước (sort tăng dần)
            resists_sorted  = sorted(resists,  key=lambda x: x[0])

            for i, key in enumerate(["ichi_support_1", "ichi_support_2", "ichi_support_3"]):
                if i < len(supports_sorted):
                    result[key] = supports_sorted[i][0]
            for i, key in enumerate(["ichi_resist_1", "ichi_resist_2", "ichi_resist_3"]):
                if i < len(resists_sorted):
                    result[key] = resists_sorted[i][0]

        # ── 6. Ichimoku Score (sub-score cho ScoringEngine) ───────────────────
        # Tính sub-score 0-10 dựa trên vị thế và tín hiệu Ichimoku
        ichi_sc = 5.0  # Neutral baseline
        if result["ichi_bull_3tf"]:        ichi_sc += 3.5   # Triple bull = cực mạnh
        elif result["ichi_above_cloud_d"]:  ichi_sc += 2.0   # Daily above cloud
        if result["ichi_above_cloud_w"]:    ichi_sc += 1.0   # Weekly confirmation
        if result["ichi_above_cloud_m"]:    ichi_sc += 0.5   # Monthly confirmation
        if result["ichi_tk_cross_up_d"]:    ichi_sc += 1.0   # Tenkan > Kijun
        if result["ichi_strong_cloud_d"]:   ichi_sc += 0.5   # Mây xanh
        if result["ichi_chikou_above_d"]:   ichi_sc += 0.5   # Chikou xác nhận
        # Penalty
        if result["ichi_bear_3tf"]:         ichi_sc -= 3.5
        elif result["ichi_below_cloud_d"]:  ichi_sc -= 2.0
        if result["ichi_below_cloud_w"]:    ichi_sc -= 1.0
        if result["ichi_in_cloud_d"]:       ichi_sc -= 0.5   # Trong mây = không rõ
        result["ichi_score"] = round(max(0.5, min(10.0, ichi_sc)), 2)

        # ── 7. Label tổng quan ────────────────────────────────────────────────
        try:
            if bull_3tf:
                label = "☀️ Ichimoku 3TF Bull — Mua mạnh"
            elif result["ichi_above_cloud_d"] and result["ichi_above_cloud_w"]:
                label = "🌤 Trên Kumo D+W — Uptrend"
            elif result["ichi_above_cloud_d"] and result["ichi_tk_cross_up_d"]:
                label = "🟢 Trên Kumo Daily + TK>KJ"
            elif result["ichi_in_cloud_d"]:
                label = "☁️ Trong Mây — Chờ breakout"
            elif result["ichi_below_cloud_d"] and result["ichi_below_cloud_w"]:
                label = "🔴 Dưới Kumo D+W — Downtrend"
            elif result["ichi_below_cloud_d"]:
                label = "🟠 Dưới Kumo Daily — Thận trọng"
            else:
                label = "—"
            result["ichi_label"] = label
        except Exception:
            pass

        return result

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5d — DAO GĂM DETECTOR  (NEW v5.5)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# "Dao găm" — nến có bóng dưới (wick) xuyên xuống chạm Kijun-sen nhưng đóng
# cửa bật lên trên đường này. Đây là dạng pin-bar/hammer test hỗ trợ:
# Smart Money đẩy giá xuống thử phản ứng thị trường tại vùng cân bằng,
# nếu lực mua đủ mạnh để đóng cửa trên Kijun → hỗ trợ được xác nhận.
#
# Check theo 3 khung:
#   - Daily : low thấp nhất trong 3 phiên gần nhất vs Kijun Daily
#   - Weekly: low thấp nhất TRONG TUẦN HIỆN TẠI (kể cả tuần chưa đóng) vs Kijun Weekly
#   - Monthly: low thấp nhất TRONG THÁNG HIỆN TẠI vs Kijun Monthly
#
# Lưu ý: dùng tuần/tháng "hiện tại" (chưa đóng) vì nếu chờ tuần/tháng đóng
# xong mới confirm thì đã mất điểm entry — dao găm cần xác nhận real-time.

class DaoGamDetector:
    """
    Phát hiện "Dao găm" — test hỗ trợ Kijun-sen thành công theo 3 khung Ichimoku.

    Output: dict bool cho từng TF + label tổng hợp để hiển thị.
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def detect(self, df_long: pd.DataFrame,
               kijun_d: float, kijun_w: float, kijun_m: float,
               close: float) -> dict:
        result = {
            "dao_gam_d": False, "dao_gam_w": False, "dao_gam_m": False,
            "dao_gam_label": "—",
        }
        if df_long is None or len(df_long) < 3:
            return result

        tol = self.cfg.dao_gam_tolerance_pct / 100

        # ── Daily: 3 phiên gần nhất ───────────────────────────────────────────
        try:
            if kijun_d:
                low_min_d = float(df_long["low"].tail(3).min())
                result["dao_gam_d"] = bool(
                    low_min_d <= kijun_d * (1 + tol) and close > kijun_d
                )
        except Exception:
            pass

        # ── Weekly: tuần hiện tại (kể cả chưa đóng nến tuần) ─────────────────
        try:
            if kijun_w:
                df2 = df_long.copy()
                df2["time"] = pd.to_datetime(df2["time"])
                cur_week = df2["time"].max().to_period("W-FRI")
                mask = df2["time"].dt.to_period("W-FRI") == cur_week
                low_min_w = float(df2.loc[mask, "low"].min())
                result["dao_gam_w"] = bool(
                    low_min_w <= kijun_w * (1 + tol) and close > kijun_w
                )
        except Exception:
            pass

        # ── Monthly: tháng hiện tại ───────────────────────────────────────────
        try:
            if kijun_m:
                df2 = df_long.copy()
                df2["time"] = pd.to_datetime(df2["time"])
                cur_month = df2["time"].max().to_period("M")
                mask = df2["time"].dt.to_period("M") == cur_month
                low_min_m = float(df2.loc[mask, "low"].min())
                result["dao_gam_m"] = bool(
                    low_min_m <= kijun_m * (1 + tol) and close > kijun_m
                )
        except Exception:
            pass

        # ── Label tổng hợp ────────────────────────────────────────────────────
        tfs = []
        if result["dao_gam_d"]: tfs.append("D")
        if result["dao_gam_w"]: tfs.append("W")
        if result["dao_gam_m"]: tfs.append("M")
        if len(tfs) >= 2:
            result["dao_gam_label"] = f"🗡️🗡️ Dao găm {'+'.join(tfs)} — Hỗ trợ mạnh"
        elif tfs:
            result["dao_gam_label"] = f"🗡️ Dao găm {tfs[0]} — Test hỗ trợ"
        else:
            result["dao_gam_label"] = "—"

        return result

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5e — COMPOSITE INDEX RSI DIVERGENCE ENGINE  (NEW v5.7)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# NGUỒN Ý TƯỞNG
#   Port từ Pine Script "Composite Index Divergence Indicator" (z3bazi, dựa trên
#   Composite Index của Constance Brown) sang Python/pandas. Mục tiêu: phát hiện
#   PHÂN KỲ RSI tự động, theo đúng logic script gốc chứ không dùng phiên bản
#   phân kỳ đơn giản trong MomentumEngine (chỉ 40 nến, pivot dễ lệch).
#
# COMPOSITE INDEX (osc) — dao động dùng để so sánh với giá
#   osc = RSI14 momentum(9)  +  SMA3( RSI3 )
#         └─ rsi_delta ─────┘    └─ rsi_sma ─┘
#   - rsi_delta = RSI14 hôm nay − RSI14 cách 9 phiên  (đà của RSI)
#   - rsi_sma   = trung bình 3 phiên của RSI chu kỳ 3  (RSI cực nhạy, làm mượt)
#   Ngoài ra có 2 đường SMA13 / SMA33 của osc (đường tín hiệu, giống MACD).
#
# PIVOT (đỉnh/đáy của osc) — có ĐỘ TRỄ xác nhận
#   Một bar là pivot đáy nếu osc tại đó THẤP NHẤT trong cửa sổ
#   [5 bar bên trái ... 5 bar bên phải] (lbL=5, lbR=5).
#   => Pivot chỉ được XÁC NHẬN sau lbR=5 phiên. Đây là thiết kế của script gốc:
#      không repaint, nhưng tín hiệu luôn đến muộn 5 phiên so với đáy thật.
#      Trong code: "pivot_idx" = bar đáy/đỉnh thật, "confirm_idx" = pivot_idx + lbR
#      = bar mà tín hiệu chính thức xuất hiện (bar đầu tiên có thể hành động).
#
# 4 LOẠI PHÂN KỲ (so sánh pivot hiện tại với pivot LIỀN TRƯỚC cùng phía)
#   ┌──────────┬───────────────┬──────────────┬─────────────────────────────┐
#   │ Mã loại  │ Giá           │ Composite    │ Ý nghĩa                     │
#   ├──────────┼───────────────┼──────────────┼─────────────────────────────┤
#   │ bull_reg │ đáy thấp hơn  │ đáy cao hơn  │ PK DƯƠNG thường: đảo chiều ↑│
#   │ bull_hid │ đáy cao hơn   │ đáy thấp hơn │ PK dương ẩn: uptrend tiếp ↑ │
#   │ bear_reg │ đỉnh cao hơn  │ đỉnh thấp hơn│ PK ÂM thường: đảo chiều ↓   │
#   │ bear_hid │ đỉnh thấp hơn │ đỉnh cao hơn │ PK âm ẩn: downtrend tiếp ↓  │
#   └──────────┴───────────────┴──────────────┴─────────────────────────────┘
#   Giá được lấy TẠI CHÍNH bar pivot của osc (low cho đáy, high cho đỉnh;
#   hoặc close nếu cfg.cd_use_close=True) — đúng như script gốc.
#
# ĐIỀU KIỆN KHOẢNG CÁCH (_inRange trong script gốc)
#   Khoảng cách giữa 2 pivot phải nằm trong [cd_range_min, cd_range_max] bar
#   (5–60). Công thức chính xác của script: bars = confirm_now − confirm_prev − 1.
#   Loại bỏ cặp pivot quá sát (nhiễu) hoặc quá xa (không còn liên quan).
#
# TÍN HIỆU "MỚI" (fresh) VÀ VÔ HIỆU HÓA (invalid) — phần bổ sung so với script
#   - fresh  : phân kỳ vừa được xác nhận trong vòng cfg.cd_signal_window phiên
#              (mặc định 3). Phân kỳ cũ hơn không còn được coi là tín hiệu hiện tại.
#   - invalid: sau khi xác nhận, giá ĐÓNG CỬA xuyên qua mức pivot
#              (bull: đóng dưới đáy pivot; bear: đóng trên đỉnh pivot)
#              → luận điểm đảo chiều đã thất bại → không còn fresh.
#
# OUTPUT (dict, prefix "cd_") — merge vào `raw`:
#   cd_valid, cd_osc, cd_fast, cd_slow, cd_osc_above_fast, cd_rsi14
#   cd_{kind}_fresh / _age / _pivot_date / _prev_date / _pivot_px / _invalid
#       với kind ∈ {bull_reg, bull_hid, bear_reg, bear_hid}
#   cd_label (nhãn hiển thị), cd_note (giải thích chi tiết), cd_bias
#
# TÍCH HỢP: xem merge_into_raw() — mặc định GHI ĐÈ rsi_bull_div / rsi_bear_div
#   của MomentumEngine bằng phân kỳ THƯỜNG của engine này, để toàn bộ logic
#   downstream (Scoring, WinRate, smart_entry, levels, classify_horizon,
#   PositionAdvisor) tự dùng tín hiệu tốt hơn mà KHÔNG bị cộng điểm 2 lần.
#   Bản cũ được giữ trong raw["rsi_bull_div_legacy"] / ["rsi_bear_div_legacy"].
#   Phân kỳ ẨN không đưa vào Score/Win% (chưa có dữ liệu kiểm chứng trọng số) —
#   chỉ hiển thị và dùng trong DivergenceAdvisor.
#
# KIỂM CHỨNG: engine có forward_stats() / baseline_stats() và hàm
#   divergence_study() (Cell 15) để đo tỷ lệ thắng THỰC TẾ của từng loại phân kỳ
#   trên dữ liệu lịch sử so với xác suất nền — đừng tin con số "75%" khi chưa chạy.

class CompositeDivergenceEngine:
    """
    Phân kỳ RSI dựa trên Composite Index (Constance Brown), port từ Pine Script.

    Cách dùng nhanh:
        eng = CompositeDivergenceEngine(QuantConfig())
        cd  = eng.compute(df_long)     # df_long: OHLCV ngày, cột time/open/high/low/close/volume
        cd["cd_label"], cd["cd_bull_reg_fresh"], ...
    """

    KINDS   = ("bull_reg", "bull_hid", "bear_reg", "bear_hid")
    KIND_VI = {"bull_reg": "PK dương thường", "bull_hid": "PK dương ẩn",
               "bear_reg": "PK âm thường",    "bear_hid": "PK âm ẩn"}
    KIND_IC = {"bull_reg": "🟢", "bull_hid": "🟩", "bear_reg": "🔴", "bear_hid": "🟠"}

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    # ══════════════════════════════════════════════════════════════════════════
    # 1. COMPOSITE INDEX
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _rsi_wilder(close: pd.Series, length: int) -> pd.Series:
        """
        RSI chuẩn Wilder — giống hàm rsi() của Pine Script (RMA có "seed").
          - Trung bình gain/loss đầu tiên = SMA của `length` biến động đầu tiên
          - Các phiên sau: avg = (avg_trước × (length−1) + giá_trị_mới) / length
            (tương đương EWM alpha = 1/length bắt đầu từ giá trị seed).
        LƯU Ý: khác RSI trong IndicatorEngine.compute_all (dùng SMA đơn giản nên ra
        số hơi khác). Ảnh hưởng của seed tan dần theo cấp số nhân, nên engine này
        được cấp df_long (≈400 phiên) chứ không phải df ngắn 40 phiên.
        """
        delta = close.astype(float).diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        if len(close) <= length:
            return pd.Series(np.nan, index=close.index)

        def rma(x: pd.Series) -> pd.Series:
            s = x.copy()
            seed = x.iloc[1: length + 1].mean()      # SMA của `length` biến động đầu tiên
            s.iloc[: length] = np.nan                # bỏ phần trước seed
            s.iloc[length]   = seed                  # đặt seed tại bar thứ `length`
            return s.ewm(alpha=1 / length, adjust=False).mean()   # từ seed trở đi: RMA

        avg_gain, avg_loss = rma(gain), rma(loss)
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        # Không có phiên giảm nào trong cửa sổ → RSI = 100 (tránh NaN do chia 0)
        return rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)

    def composite_index(self, close: pd.Series):
        """
        Trả về (osc, fast_sma, slow_sma, rsi14) — bốn pd.Series cùng index.
        osc = (RSI14 − RSI14 cách 9 phiên) + SMA3(RSI3)
        """
        c = self.cfg
        rsi_main  = self._rsi_wilder(close, c.cd_rsi_len)              # RSI14
        rsi_delta = rsi_main - rsi_main.shift(c.cd_mom_len)            # mom(RSI14, 9)
        rsi_fast  = self._rsi_wilder(close, c.cd_rsi_ma_len)           # RSI3
        rsi_sma   = rsi_fast.rolling(c.cd_sma_len).mean()              # SMA3(RSI3)
        osc       = rsi_delta + rsi_sma
        fast      = osc.rolling(c.cd_fast_len).mean()                  # SMA13(osc)
        slow      = osc.rolling(c.cd_slow_len).mean()                  # SMA33(osc)
        return osc, fast, slow, rsi_main

    # ══════════════════════════════════════════════════════════════════════════
    # 2. PIVOT
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _pivot_flags(values: np.ndarray, left: int, right: int, side: str) -> np.ndarray:
        """
        Đánh dấu các bar là pivot. Trả về mảng bool, True tại vị trí PIVOT THẬT
        (chưa cộng độ trễ xác nhận).

        Bar i là pivot đáy ("low") nếu values[i] là nhỏ nhất trong cửa sổ
        [i-left, i+right]; pivot đỉnh ("high") nếu lớn nhất.
        Khi có nhiều bar bằng nhau, chọn bar ĐẦU TIÊN (bên trái nhất) để tránh
        sinh 2 pivot liền kề — xấp xỉ hành vi pivotlow/pivothigh của TradingView.
        Cửa sổ chứa NaN (giai đoạn warm-up của osc) bị bỏ qua.
        """
        n = len(values)
        flags = np.zeros(n, dtype=bool)
        w = left + right + 1
        if n < w:
            return flags
        win   = sliding_window_view(values, w)          # mỗi hàng = 1 cửa sổ w phần tử
        valid = ~np.isnan(win).any(axis=1)
        pos   = win.argmin(axis=1) if side == "low" else win.argmax(axis=1)  # vị trí cực trị (đầu tiên)
        is_pv = valid & (pos == left)                   # cực trị nằm đúng giữa cửa sổ
        flags[left: left + len(win)] = is_pv            # cửa sổ j có tâm là bar j+left
        return flags

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TÌM CÁC PHÂN KỲ TRONG TOÀN BỘ LỊCH SỬ
    # ══════════════════════════════════════════════════════════════════════════
    def find_events(self, df: pd.DataFrame, osc: pd.Series) -> list:
        """
        Quét toàn bộ lịch sử, trả về danh sách sự kiện phân kỳ (sắp theo confirm_idx).
        Mỗi sự kiện là dict:
            kind          : bull_reg | bull_hid | bear_reg | bear_hid
            pivot_idx     : chỉ số bar pivot (đáy/đỉnh thật)
            prev_idx      : chỉ số bar pivot liền trước
            confirm_idx   : pivot_idx + lbR — bar tín hiệu được xác nhận
            pivot_px      : giá tại pivot (low/high hoặc close)
            prev_px       : giá tại pivot trước
            osc_now, osc_prev : giá trị Composite tại 2 pivot
        """
        c = self.cfg
        o = osc.values.astype(float)
        if c.cd_use_close:
            px_low = px_high = df["close"].values.astype(float)
        else:
            px_low, px_high = df["low"].values.astype(float), df["high"].values.astype(float)

        events = []
        for side in ("low", "high"):
            px    = px_low if side == "low" else px_high
            flags = self._pivot_flags(o, c.cd_lbl, c.cd_lbr, side)
            pivs  = np.flatnonzero(flags)                # các bar pivot, tăng dần

            # Duyệt từng CẶP pivot liên tiếp (pivot trước → pivot hiện tại),
            # giống valuewhen(plFound, ..., 1) trong Pine: luôn so với pivot LIỀN TRƯỚC.
            for prev_p, cur_p in zip(pivs[:-1], pivs[1:]):
                # _inRange: bars = confirm_now − confirm_prev − 1 (= cur_p − prev_p − 1)
                bars = cur_p - prev_p - 1
                if not (c.cd_range_min <= bars <= c.cd_range_max):
                    continue
                o_now, o_prev   = o[cur_p], o[prev_p]
                px_now, px_prev = px[cur_p], px[prev_p]
                base = {"pivot_idx": int(cur_p), "prev_idx": int(prev_p),
                        "confirm_idx": int(cur_p + c.cd_lbr),
                        "pivot_px": float(px_now), "prev_px": float(px_prev),
                        "osc_now": float(o_now), "osc_prev": float(o_prev)}
                if side == "low":
                    if px_now < px_prev and o_now > o_prev:      # giá đáy thấp hơn, osc đáy cao hơn
                        events.append({**base, "kind": "bull_reg"})
                    if px_now > px_prev and o_now < o_prev:      # giá đáy cao hơn, osc đáy thấp hơn
                        events.append({**base, "kind": "bull_hid"})
                else:
                    if px_now > px_prev and o_now < o_prev:      # giá đỉnh cao hơn, osc đỉnh thấp hơn
                        events.append({**base, "kind": "bear_reg"})
                    if px_now < px_prev and o_now > o_prev:      # giá đỉnh thấp hơn, osc đỉnh cao hơn
                        events.append({**base, "kind": "bear_hid"})
        events.sort(key=lambda e: e["confirm_idx"])
        return events

    # ══════════════════════════════════════════════════════════════════════════
    # 4. TÍN HIỆU HIỆN TẠI (phiên cuối cùng của df)
    # ══════════════════════════════════════════════════════════════════════════
    def _empty(self) -> dict:
        """Giá trị mặc định — đảm bảo downstream không bị KeyError khi thiếu dữ liệu."""
        r = {"cd_valid": False, "cd_osc": None, "cd_fast": None, "cd_slow": None,
             "cd_osc_above_fast": False, "cd_rsi14": None,
             "cd_label": "—", "cd_note": "", "cd_bias": "neutral"}
        for k in self.KINDS:
            r.update({f"cd_{k}_fresh": False, f"cd_{k}_age": None, f"cd_{k}_invalid": False,
                      f"cd_{k}_pivot_date": None, f"cd_{k}_prev_date": None, f"cd_{k}_pivot_px": None})
        return r

    @staticmethod
    def _fmt_date(df: pd.DataFrame, idx: int) -> str:
        """Định dạng ngày dd/mm của bar idx (rơi về chỉ số bar nếu không có cột time)."""
        try:
            return pd.to_datetime(df["time"].iloc[idx]).strftime("%d/%m")
        except Exception:
            return f"#{idx}"

    def compute(self, df: pd.DataFrame) -> dict:
        """
        Tính phân kỳ tại phiên cuối của df.
        df nên là df_long (≥ ~150 phiên; càng dài RSI Wilder càng hội tụ).
        """
        res, c = self._empty(), self.cfg
        min_bars = c.cd_rsi_len * 3 + c.cd_mom_len + c.cd_lbl + c.cd_lbr + 10
        if df is None or len(df) < min_bars:
            return res
        try:
            osc, fast, slow, rsi14 = self.composite_index(df["close"])
            events = self.find_events(df, osc)
            last   = len(df) - 1
            closes = df["close"].values.astype(float)

            res.update({"cd_valid": True,
                        "cd_osc":  round(float(osc.iloc[-1]), 2),
                        "cd_fast": round(float(fast.iloc[-1]), 2) if not pd.isna(fast.iloc[-1]) else None,
                        "cd_slow": round(float(slow.iloc[-1]), 2) if not pd.isna(slow.iloc[-1]) else None,
                        "cd_rsi14": round(float(rsi14.iloc[-1]), 2)})
            res["cd_osc_above_fast"] = bool(res["cd_fast"] is not None and res["cd_osc"] > res["cd_fast"])

            notes, labels = [], []
            for k in self.KINDS:
                evs = [e for e in events if e["kind"] == k]
                if not evs:
                    continue
                e   = evs[-1]                                   # sự kiện GẦN NHẤT của loại này
                age = last - e["confirm_idx"]                   # đã xác nhận cách đây bao nhiêu phiên
                after = closes[e["confirm_idx"] + 1:]           # các phiên đóng cửa sau xác nhận
                # Vô hiệu hóa: đóng cửa xuyên qua mức pivot sau khi đã xác nhận
                if k.startswith("bull"):
                    invalid = bool(len(after) and after.min() < e["pivot_px"])
                else:
                    invalid = bool(len(after) and after.max() > e["pivot_px"])
                fresh = bool(age <= c.cd_signal_window and not invalid)

                res[f"cd_{k}_age"]        = int(age)
                res[f"cd_{k}_invalid"]    = invalid
                res[f"cd_{k}_fresh"]      = fresh
                res[f"cd_{k}_pivot_date"] = self._fmt_date(df, e["pivot_idx"])
                res[f"cd_{k}_prev_date"]  = self._fmt_date(df, e["prev_idx"])
                res[f"cd_{k}_pivot_px"]   = round(e["pivot_px"], 2)

                if fresh:
                    when = "hôm nay" if age == 0 else f"{age} phiên trước"
                    labels.append(f"{self.KIND_IC[k]} {self.KIND_VI[k]} (xác nhận {when}; "
                                  f"{'đáy' if k.startswith('bull') else 'đỉnh'} {res[f'cd_{k}_pivot_date']})")
                    notes.append(self._explain(k, e, res[f"cd_{k}_pivot_date"], res[f"cd_{k}_prev_date"]))

            res["cd_label"] = " + ".join(labels) if labels else "—"
            res["cd_note"]  = " | ".join(notes)
            bull = res["cd_bull_reg_fresh"] or res["cd_bull_hid_fresh"]
            bear = res["cd_bear_reg_fresh"] or res["cd_bear_hid_fresh"]
            res["cd_bias"] = "mixed" if (bull and bear) else "bull" if bull else "bear" if bear else "neutral"
        except Exception:
            # Không raise: lỗi tính toán 1 mã không được làm hỏng cả lượt scan
            return self._empty()
        return res

    @staticmethod
    def _explain(kind: str, e: dict, d_now: str, d_prev: str) -> str:
        """Câu giải thích cụ thể để người đọc (hoặc AI khác) tự đối chiếu trên chart."""
        side = "đáy" if kind.startswith("bull") else "đỉnh"
        px_dir  = {"bull_reg": "thấp hơn", "bull_hid": "cao hơn",
                   "bear_reg": "cao hơn",  "bear_hid": "thấp hơn"}[kind]
        osc_dir = {"bull_reg": "cao hơn",  "bull_hid": "thấp hơn",
                   "bear_reg": "thấp hơn", "bear_hid": "cao hơn"}[kind]
        return (f"{CompositeDivergenceEngine.KIND_VI[kind]}: {side} giá {d_now} ({e['pivot_px']:.2f}) "
                f"{px_dir} {side} trước {d_prev} ({e['prev_px']:.2f}) nhưng {side} RSI-Composite "
                f"{osc_dir} ({e['osc_now']:.1f} vs {e['osc_prev']:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. GHÉP VÀO raw
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _rebuild_div_label(raw: dict) -> str:
        """
        Dựng lại raw["div_label"] (nhãn cột 'Phân kỳ') theo đúng thứ tự ưu tiên của
        MomentumEngine nhưng dùng cờ RSI mới. MACD/hidden-accumulation/BB squeeze giữ nguyên.
        """
        rbd, mbd = raw.get("rsi_bull_div"), raw.get("macd_bull_div")
        rrd, mrd = raw.get("rsi_bear_div"), raw.get("macd_bear_div")
        if   rbd and mbd:  return "💚 Phân kỳ dương kép (RSI+MACD)"
        elif rbd:          return "🟢 RSI Phân kỳ dương"
        elif mbd:          return "🟩 MACD Phân kỳ dương"
        elif rrd and mrd:  return "🔴 Phân kỳ âm kép (RSI+MACD)"
        elif rrd:          return "🟠 RSI Phân kỳ âm"
        elif mrd:          return "🟡 MACD Phân kỳ âm"
        elif raw.get("hidden_accumulation"): return "🔍 Tích lũy ẩn (CMF↑)"
        elif raw.get("is_bb_squeeze"):       return "🗜 BB Squeeze — Chờ nổ"
        return "—"

    def merge_into_raw(self, raw: dict, cd: dict) -> None:
        """
        Gộp kết quả vào dict raw (sửa tại chỗ).
        Nếu cfg.cd_replace_legacy_rsi_div=True và engine tính được (cd_valid):
          rsi_bull_div ← cd_bull_reg_fresh ; rsi_bear_div ← cd_bear_reg_fresh
        (bản cũ lưu ở *_legacy) rồi dựng lại div_label.
        Đặt cd_replace_legacy_rsi_div=False để so sánh A/B hai cách tính.
        """
        raw.update(cd)
        if self.cfg.cd_replace_legacy_rsi_div and cd.get("cd_valid"):
            raw["rsi_bull_div_legacy"] = raw.get("rsi_bull_div", False)
            raw["rsi_bear_div_legacy"] = raw.get("rsi_bear_div", False)
            raw["rsi_bull_div"] = bool(cd["cd_bull_reg_fresh"])
            raw["rsi_bear_div"] = bool(cd["cd_bear_reg_fresh"])
            raw["div_label"]    = self._rebuild_div_label(raw)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. KIỂM CHỨNG LỊCH SỬ (dùng bởi divergence_study, Cell 15)
    # ══════════════════════════════════════════════════════════════════════════
    def forward_stats(self, df: pd.DataFrame, horizons=(3, 5, 10)) -> pd.DataFrame:
        """
        Với MỌI phân kỳ trong lịch sử, đo kết quả nếu hành động NGAY khi xác nhận.

        Giả định thực tế (không look-ahead):
          - Tín hiệu biết được ở ĐÓNG CỬA bar confirm_idx → vào lệnh ở giá MỞ CỬA phiên kế tiếp.
          - ret_h  = (close sau h phiên kể từ phiên vào lệnh) / entry − 1   (đơn vị %)
            h tối thiểu 3 để phù hợp T+2 (mua phiên 1, bán được từ phiên 3).
          - mfe    = mức lãi thuận chiều tối đa (%) trong max(horizons) phiên (dùng high/low)
          - mae    = mức lỗ ngược chiều tối đa (%) trong cùng khoảng
          Với phân kỳ ÂM (kỳ vọng giá giảm): dấu ngược lại (mfe = mức giảm tối đa).
        Chưa tính phí, thuế, trần/sàn, trượt giá.
        """
        if df is None or len(df) < 80:
            return pd.DataFrame()
        osc, *_ = self.composite_index(df["close"])
        events  = self.find_events(df, osc)
        o, h, l, c = (df[x].values.astype(float) for x in ("open", "high", "low", "close"))
        n, hmax, rows = len(df), max(horizons), []
        for e in events:
            t = e["confirm_idx"]
            if t + 1 >= n or not o[t + 1] > 0:
                continue                                       # chưa có phiên kế tiếp để vào lệnh
            entry = o[t + 1]
            sign  = 1 if e["kind"].startswith("bull") else -1
            row = {"kind": e["kind"], "confirm_date": self._fmt_date(df, t),
                   "entry": round(entry, 2), "full_window": bool(t + hmax < n)}
            for hz in horizons:
                row[f"ret_{hz}"] = (c[t + hz] / entry - 1) * 100 if t + hz < n else np.nan
            end = min(t + hmax, n - 1)
            hi, lo = h[t + 1: end + 1].max(), l[t + 1: end + 1].min()
            row["mfe"] = ((hi / entry - 1) if sign > 0 else (1 - lo / entry)) * 100
            row["mae"] = ((1 - lo / entry) if sign > 0 else (hi / entry - 1)) * 100
            rows.append(row)
        return pd.DataFrame(rows)

    def baseline_stats(self, df: pd.DataFrame, horizons=(3, 5, 10)) -> dict:
        """
        Xác suất NỀN: nếu mua vào mở cửa MỌI phiên (không cần tín hiệu) thì tỷ lệ
        lãi sau h phiên là bao nhiêu? Cần để so sánh — một tín hiệu "thắng 60%" chẳng
        có giá trị nếu cứ mua bừa cũng thắng 55%. Trả về {hz: (n, n_pos)} để cộng dồn.
        """
        o, c = df["open"].values.astype(float), df["close"].values.astype(float)
        out = {}
        for hz in horizons:
            if len(df) <= hz + 1:
                out[hz] = (0, 0, 0.0); continue
            ret = (c[hz:] / o[1: len(df) - hz + 1] - 1) * 100   # ret[t] = close[t+hz] / open[t+1] − 1
            ret = ret[np.isfinite(ret)]
            out[hz] = (len(ret), int((ret > 0).sum()), float(ret.sum()))
        return out

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — VOLUME DRY-UP DETECTOR                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Wyckoff Spring / Volume Dry-up — một trong những setup mạnh nhất VN
#
# Lý thuyết Wyckoff:
#   Phase A: Selling Climax — bán tháo với volume lớn
#   Phase B: Accumulation — gom hàng âm thầm, volume giảm dần
#   Phase C: Spring — thử thách đáy cuối (volume bùng sau nền khô)
#   Phase D: Markup — bắt đầu sóng tăng
#
# Detector này focus vào Phase B→C:
#   - Volume khô dần (cung cạn)
#   - Biên độ nến hẹp dần (không ai giao dịch)
#   - Không có nến đỏ mạnh kèm volume (không phân phối)
#   → Khi 3 điều kiện thỏa = nền đang hoàn tất
#
# Spring detection:
#   Volume đột tăng >1.5× MA20 SAU ít nhất n-1 phiên khô
#   + Giá đóng cửa tăng → Smart Money vừa kích hoạt

class VolumeDryupDetector:
    """
    Phát hiện nền cạn cung (Volume Dry-up) — Wyckoff Phase B/C.

    3 thành phần kiểm tra:
    1. Vol 5 phiên gần < 70% MA20 (cung cạn kiệt)
    2. Biên độ nến hẹp hơn ATR14 × 75% (nền tích lũy yên tĩnh)
    3. Không có nến đỏ mạnh kèm volume cao (không phân phối)

    Bonus: Wyckoff Spring — volume đột tăng sau nền khô, giá đóng cao.

    Output gồm: flag, strength 0-3, label mô tả, bonus score 0-2 điểm.
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def detect(self, df: pd.DataFrame) -> dict:
        if len(df) < 25:
            return self._empty()

        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        n = self.cfg.dryup_sessions  # mặc định 5 phiên

        # ── 1. Volume khô ────────────────────────────────────────────────────
        vol_ma20     = v.rolling(20).mean()
        recent_vols  = v.iloc[-n:]
        recent_ma    = vol_ma20.iloc[-n:]
        vol_ratios   = recent_vols.values / (recent_ma.values + 1e-9)
        vol_dry      = bool(np.all(vol_ratios < self.cfg.dryup_vol_ratio))
        avg_vol_ratio = float(np.mean(vol_ratios))

        # ── 2. Nền hẹp ───────────────────────────────────────────────────────
        tr    = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        recent_ranges = (h.iloc[-n:] - l.iloc[-n:]).values
        avg_range     = float(np.mean(recent_ranges))
        narrow_base   = bool(avg_range < atr14 * self.cfg.dryup_range_ratio)

        # Kiểm tra thêm: biên độ có đang thu hẹp dần không?
        if n >= 4:
            first_half  = float(np.mean(recent_ranges[:n//2]))
            second_half = float(np.mean(recent_ranges[n//2:]))
            narrowing   = bool(second_half < first_half * 0.85)
        else:
            narrowing = narrow_base

        # ── 3. Không phân phối ───────────────────────────────────────────────
        recent_o   = df["open"].iloc[-n:].values if "open" in df.columns else df["close"].iloc[-n:].values
        recent_c   = c.iloc[-n:].values
        recent_v   = v.iloc[-n:].values
        recent_vma = vol_ma20.iloc[-n:].values
        no_distrib = True
        for i in range(n):
            bearish_body  = (recent_o[i] - recent_c[i]) / (recent_o[i] + 1e-9) > 0.015
            high_vol_bar  = recent_v[i] > recent_vma[i] * 1.2
            if bearish_body and high_vol_bar:
                no_distrib = False
                break

        # ── 4. Wyckoff Spring ─────────────────────────────────────────────────
        last_vol    = float(v.iloc[-1])
        last_ma20   = float(vol_ma20.iloc[-1])
        prev_dry    = bool(np.all(vol_ratios[:-1] < self.cfg.dryup_vol_ratio))
        spring_flag = bool(prev_dry and last_vol > last_ma20 * 1.5 and c.iloc[-1] > c.iloc[-2])

        # ── Tổng hợp ─────────────────────────────────────────────────────────
        strength    = sum([vol_dry, narrow_base or narrowing, no_distrib])
        dry_up_flag = strength >= 2

        if   spring_flag:   label = "🚀 Wyckoff Spring — Bứt phá nền"
        elif strength == 3: label = "💎 Cạn cung hoàn hảo — Chờ kích hoạt"
        elif strength == 2: label = "📦 Đang tích lũy — Vol khô dần"
        elif vol_dry:       label = "🔇 Vol thấp — Chưa rõ xu hướng"
        else:               label = "—"

        if   spring_flag:   bonus = min(self.cfg.dryup_bonus, 2.0)
        elif strength == 3: bonus = self.cfg.dryup_bonus * 0.8
        elif strength == 2: bonus = self.cfg.dryup_bonus * 0.5
        else:               bonus = 0.0

        return {
            "dry_up_flag":     dry_up_flag,
            "dry_up_strength": strength,
            "dry_up_label":    label,
            "spring_flag":     spring_flag,
            "vol_dry":         vol_dry,
            "narrow_base":     narrow_base or narrowing,
            "no_distrib":      no_distrib,
            "avg_vol_ratio":   round(avg_vol_ratio, 2),
            "avg_range_ratio": round(avg_range / (atr14 + 1e-9), 2),
            "bonus_score":     round(bonus, 2),
        }

    def _empty(self) -> dict:
        return {
            "dry_up_flag": False, "dry_up_strength": 0,
            "dry_up_label": "—", "spring_flag": False,
            "vol_dry": False, "narrow_base": False, "no_distrib": False,
            "avg_vol_ratio": 1.0, "avg_range_ratio": 1.0, "bonus_score": 0.0,
        }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — WIN RATE ESTIMATOR                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Tại sao không dùng backtest để tính win rate?
#   1. vnstock không có tick data đủ sâu cho rolling backtest
#   2. Market VN thay đổi cấu trúc nhanh (circuit breaker T+0 pilot,
#      margin calls, HNX vs HOSE behavior khác nhau)
#   3. Backtest overfitted không honest bằng heuristic model có lý luận
#
# Approach: Pattern-based heuristic + Regime adjustment
#
# Công thức:
#   win_rate = base (45%) + Σ(pattern_delta) × regime_mult
#   Clamp: [25%, 82%] — không bao giờ "chắc chắn" hay "không có cơ hội"
#
# Pattern table: 19 patterns, mỗi pattern +/- điểm % vào win rate
# Regime multiplier:
#   Bull × 1.15 → cùng setup thắng nhiều hơn khi thị trường tốt
#   Panic × 0.65 → cùng setup thắng ít hơn khi thị trường hoảng loạn

class WinRateEstimator:
    """
    Ước tính xác suất thắng dựa trên Pattern + Regime Multiplier.

    Không phải backtest — là heuristic model:
    - Mỗi pattern kỹ thuật đóng góp +/- % vào win rate base
    - Regime điều chỉnh tổng thể theo chất lượng thị trường
    - Output: win_rate_est (%), confidence label, danh sách pattern active

    Scale: base 45% (thực tế VN), cap [25%, 82%]
    v5.3: Thêm 6 divergence patterns, BB squeeze breakout pattern.
    """

    _PATTERNS = {
        # VP Signals
        "vp_breakout":    {"weight": 8.0,  "check": lambda r, d: r.get("vp_signal") == "breakout"},
        "vp_at_poc":      {"weight": 5.0,  "check": lambda r, d: r.get("vp_signal") == "at_poc"},
        "vp_value_area":  {"weight": 3.0,  "check": lambda r, d: r.get("vp_signal") == "value_area"},
        # RSI Zone
        "rsi_oversold":   {"weight": 7.0,  "check": lambda r, d: r.get("rsi", 50) < 35},
        "rsi_healthy":    {"weight": 4.0,  "check": lambda r, d: 38 <= r.get("rsi", 50) <= 55},
        "rsi_overbought": {"weight": -6.0, "check": lambda r, d: r.get("rsi", 50) > 75},
        # Volume & Flow
        "vol_surge":      {"weight": 6.0,  "check": lambda r, d: r.get("volume_ratio", 1) > 2.0},
        "vol_decent":     {"weight": 3.0,  "check": lambda r, d: 1.3 <= r.get("volume_ratio", 1) <= 2.0},
        "cmf_positive":   {"weight": 5.0,  "check": lambda r, d: r.get("cmf", 0) > 0.05},
        "inst_flow_up":   {"weight": 4.0,  "check": lambda r, d: bool(r.get("inst_flow_up", False))},
        # VSA patterns
        "vsa_nos_vol":    {"weight": 9.0,  "check": lambda r, d: r.get("vsa_signal") == "🚀 NỔ VOL"},
        "vsa_rut_chan":   {"weight": 7.0,  "check": lambda r, d: r.get("vsa_signal") == "🔨 RÚT CHÂN"},
        "vsa_can_vol":    {"weight": 5.0,  "check": lambda r, d: r.get("vsa_signal") == "🩸 CẠN VOL"},
        "vsa_phan_phoi":  {"weight": -8.0, "check": lambda r, d: r.get("vsa_signal") == "⚠️ PHÂN PHỐI"},
        "vsa_can_cau":    {"weight": -4.0, "check": lambda r, d: r.get("vsa_signal") == "🥀 CẠN CẦU"},
        # Trend
        "trend_up":       {"weight": 7.0,  "check": lambda r, d: r.get("trend_long") == "uptrend"},
        "trend_weak_up":  {"weight": 3.0,  "check": lambda r, d: r.get("trend_long") == "weak_up"},
        "trend_down":     {"weight": -8.0, "check": lambda r, d: r.get("trend_long") == "downtrend"},
        # Momentum
        "macd_cross_up":  {"weight": 5.0,  "check": lambda r, d: bool(r.get("macd_cross_up", False))},
        "stoch_cross_up": {"weight": 4.0,  "check": lambda r, d: bool(r.get("stoch_cross_up", False))},
        "rs_strong":      {"weight": 4.0,  "check": lambda r, d: r.get("rs", 1) > 1.5},
        "rs_weak":        {"weight": -3.0, "check": lambda r, d: r.get("rs", 1) < 0.5},
        # Wyckoff
        "spring":         {"weight": 9.0,  "check": lambda r, d: bool(d.get("spring_flag", False))},
        "dryup_perfect":  {"weight": 6.0,  "check": lambda r, d: d.get("dry_up_strength", 0) >= 3 and not d.get("spring_flag")},
        "dryup_forming":  {"weight": 3.0,  "check": lambda r, d: d.get("dry_up_strength", 0) == 2},
        # ── v5.3: Divergence & BB patterns ─────────────────────────────────
        # RSI bullish divergence: lực bán cạn, xác suất thắng tăng mạnh
        "rsi_bull_div":   {"weight": 7.0,  "check": lambda r, d: bool(r.get("rsi_bull_div", False))},
        # MACD bullish divergence: momentum đang quay đầu trung hạn
        "macd_bull_div":  {"weight": 6.0,  "check": lambda r, d: bool(r.get("macd_bull_div", False))},
        # BB Squeeze + Breakout: nổ sau tích lũy, xác suất cao
        "bb_squeeze_brk": {"weight": 5.0,
                           "check": lambda r, d: bool(r.get("is_bb_squeeze", False)) and r.get("vp_signal") == "breakout"},
        # Hidden Accumulation: tiền lớn đang gom, setup tích cực
        "hidden_accum":   {"weight": 4.0,  "check": lambda r, d: bool(r.get("hidden_accumulation", False))},
        # RSI bearish divergence: momentum yếu, rủi ro cao
        "rsi_bear_div":   {"weight": -7.0, "check": lambda r, d: bool(r.get("rsi_bear_div", False))},
        # MACD bearish divergence: trend đang mất động lực
        "macd_bear_div":  {"weight": -5.0, "check": lambda r, d: bool(r.get("macd_bear_div", False))},

        # ── v5.4: Ichimoku patterns ──────────────────────────────────────────
        # Triple confirmation (3TF Bull) — xác suất thắng cao nhất
        "ichi_bull_3tf":  {"weight": 8.0,  "check": lambda r, d: bool(r.get("ichi_bull_3tf", False))},
        # Tenkan cắt lên Kijun Daily + trên Kumo = tín hiệu mua ngắn hạn
        "ichi_tk_cross":  {"weight": 5.0,
                           "check": lambda r, d: bool(r.get("ichi_tk_cross_up_d", False)) and
                                                bool(r.get("ichi_above_cloud_d", False))},
        # Trên Kumo D+W (2 trong 3 TF) — tích cực
        "ichi_dw_bull":   {"weight": 4.0,
                           "check": lambda r, d: bool(r.get("ichi_above_cloud_d", False)) and
                                                bool(r.get("ichi_above_cloud_w", False)) and
                                                not bool(r.get("ichi_bull_3tf", False))},
        # Chikou xác nhận — confirm thêm
        "ichi_chikou":    {"weight": 3.0,  "check": lambda r, d: bool(r.get("ichi_chikou_above_d", False))},
        # Trong mây — vùng không chắc chắn
        "ichi_in_cloud":  {"weight": -3.0, "check": lambda r, d: bool(r.get("ichi_in_cloud_d", False))},
        # Dưới Kumo Daily — downtrend, rủi ro cao
        "ichi_below_d":   {"weight": -6.0, "check": lambda r, d: bool(r.get("ichi_below_cloud_d", False))},
        # Dưới Kumo D+W — downtrend mạnh
        "ichi_bear_3tf":  {"weight": -9.0, "check": lambda r, d: bool(r.get("ichi_bear_3tf", False))},
        # ── v5.6: DMI/Aroon patterns ──────────────────────────────────────────
        "dmi_strong_bull":     {"weight": 7.0,  "check": lambda r, d: bool(r.get("dmi_bullish")) and r.get("dmi_strength") == "strong"},
        "dmi_strong_bear":     {"weight": -8.0, "check": lambda r, d: not r.get("dmi_bullish", True) and r.get("dmi_strength") == "strong"},
        "dmi_no_trend":        {"weight": -3.0, "check": lambda r, d: r.get("dmi_strength") == "none"},
        "aroon_strong_bull":   {"weight": 6.0,  "check": lambda r, d: r.get("aroon_state") == "strong_bull"},
        "aroon_strong_bear":   {"weight": -7.0, "check": lambda r, d: r.get("aroon_state") == "strong_bear"},
        "aroon_emerging_bull": {"weight": 3.0,  "check": lambda r, d: r.get("aroon_state") == "emerging_bull"},
    }


    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def estimate(self, raw: dict, dryup: dict, regime_mult: float) -> dict:
        base          = self.cfg.wr_base
        pattern_delta = 0.0
        triggered     = []
        for name, info in self._PATTERNS.items():
            try:
                if info["check"](raw, dryup):
                    pattern_delta += info["weight"]
                    if info["weight"] > 0:
                        triggered.append(name)
            except Exception:
                pass
        adjusted = (base + pattern_delta) * regime_mult
        win_rate = max(25.0, min(82.0, adjusted))
        if   len(triggered) >= 6 and win_rate >= 65: conf = "🟢 Cao"
        elif len(triggered) >= 4 and win_rate >= 55: conf = "🟡 Trung bình"
        elif len(triggered) >= 2 and win_rate >= 45: conf = "🟠 Thấp"
        else:                                         conf = "🔴 Rất thấp"
        return {
            "win_rate_est":     round(win_rate, 1),
            "win_rate_label":   f"{win_rate:.0f}% ({conf})",
            "wr_confidence":    conf,
            "wr_patterns":      len(triggered),
            "wr_pattern_names": triggered[:5],
        }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — SCORING ENGINE                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Chấm điểm 0-10 cho từng indicator, tổng hợp có trọng số.
#
# Design:
#   - Mỗi indicator → điểm sub-score 0-10
#   - Nhân với dynamic weight (đã normalize về tổng=100)
#   - Nhân regime_mult (0.7 - 1.0) → tổng score bị kéo xuống khi thị trường xấu
#   - Cộng bonus từ VolumeDryupDetector (0-2 điểm)
#   - Score cuối = (Σ sub_score × weight / 100) × regime_mult + bonus
#
# VSA integration (v5.2):
#   VSAEngine.to_score(vsa_signal) → sub-score 0-10
#   Weight "vsa" = 10 (trong BASE_WEIGHTS)
#   → Thay thế một phần cho volume_ratio trong việc đánh giá dòng tiền
#
# Mode swing vs hold:
#   RSI scoring khác nhau:
#   - Swing: RSI 28-45 điểm cao nhất (mua đáy sau retracement)
#   - Hold:  RSI 48-58 điểm cao nhất (mua khi momentum ổn định)

class ScoringEngine:
    """
    Chấm điểm 0-10 mỗi indicator, tổng hợp weighted + regime + bonus.

    v5.2: Thêm VSA sub-score với weight = 10 (trong BASE_WEIGHTS).
    v5.3: Thêm momentum_div sub-score + BB squeeze bonus trong vp_position.
    VSA được tính qua VSAEngine.to_score() → scale 0-10 consistent.
    v5.4: ichimoku sub-score + ScoreCoeffs từ cfg thay magic numbers.
    """
    def __init__(self, cfg: QuantConfig = None):
        self.cfg = cfg or QuantConfig()


    def score(self, raw: dict, mode: str, weights: dict, regime_mult: float,
              bonus: float = 0.0) -> tuple[dict, float]:
        def clamp(x): return max(0.0, min(10.0, float(x)))
        s = {}
        cfg = self.cfg

        # ── Volume ratio ──────────────────────────────────────────────────────
        # Piecewise linear: vol thấp (<0.8x) điểm thấp, vol cao (>2.2x) điểm cao
        # Không tuyến tính để tránh overweight phiên có vol đột biến bất thường
        vol = raw.get("volume_ratio", 1.0)
        if   vol <= 0.8: s["volume_ratio"] = clamp(vol * cfg.sc_vol_low_slope)
        elif vol <  1.3: s["volume_ratio"] = clamp(cfg.sc_vol_mid_base + (vol - 0.8) * cfg.sc_vol_mid_slope)
        elif vol <  2.2: s["volume_ratio"] = clamp(cfg.sc_vol_hi_base  + (vol - 1.3) * cfg.sc_vol_hi_slope)
        else:            s["volume_ratio"] = clamp(cfg.sc_vol_vhi_base + (vol - 2.2) * cfg.sc_vol_vhi_slope)

        # ── CMF ───────────────────────────────────────────────────────────────
        # CMF range thực tế -0.25 → +0.25, map về 0-10
        s["cmf"] = clamp((raw.get("cmf", 0) + cfg.sc_cmf_shift) / cfg.sc_cmf_scale)

        # ── OBV ───────────────────────────────────────────────────────────────
        # Base 7.5 nếu OBV đang tăng, 2.5 nếu giảm
        # Điều chỉnh thêm theo slope (tốc độ thay đổi)
        ob       = 7.5 if raw.get("obv_signal") == 1 else 2.5
        s["obv"] = clamp(ob + min(2.8, max(-2.8, raw.get("obv_slope", 0) / 9)))

        # ── Force Index ───────────────────────────────────────────────────────
        s["force"] = clamp((raw.get("force_index", 0) + cfg.sc_force_shift) * cfg.sc_force_scale)

        # ── Institutional Flow ────────────────────────────────────────────────
        s["inst_flow"] = float(raw.get("inst_flow", 5.0))

        # ── VSA (v5.2) ────────────────────────────────────────────────────────
        # VSAEngine.to_score map signal → 1.5 (PHÂN PHỐI) đến 8.5 (NỔ VOL)
        s["vsa"] = VSAEngine.to_score(raw.get("vsa_signal", "—"))

        # ── Momentum Divergence (v5.3) ─────────────────────────────────────────
        # Phân kỳ dương: lực bán cạn, xác suất đảo chiều cao → điểm cao
        # Phân kỳ âm:    momentum suy yếu → điểm thấp
        # Base 5.0 (neutral), điều chỉnh theo signal
        # Kép (cả RSI lẫn MACD đồng thuận) → mạnh hơn đơn
        rbd = bool(raw.get("rsi_bull_div", False))
        mbd = bool(raw.get("macd_bull_div", False))
        rrd = bool(raw.get("rsi_bear_div", False))
        mrd = bool(raw.get("macd_bear_div", False))
        ha  = bool(raw.get("hidden_accumulation", False))
        sq  = bool(raw.get("is_bb_squeeze", False))

        if   rbd and mbd: div_score = 9.0   # Phân kỳ dương kép — cực mạnh
        elif rbd:         div_score = 8.0   # RSI phân kỳ dương
        elif mbd:         div_score = 7.5   # MACD phân kỳ dương
        elif ha:          div_score = 7.0   # Hidden accumulation
        elif sq:          div_score = 6.5   # BB Squeeze — chờ nổ
        elif rrd and mrd: div_score = 1.5   # Phân kỳ âm kép — rất xấu
        elif rrd:         div_score = 2.5   # RSI phân kỳ âm
        elif mrd:         div_score = 3.0   # MACD phân kỳ âm
        else:             div_score = 5.0   # Neutral
        s["momentum_div"] = div_score

        # ── Ichimoku (v5.4) ───────────────────────────────────────────────────
        # ichi_score đã tính đầy đủ trong IchimokuEngine (0-10)
        # Thêm bonus nếu 3TF đều bullish để phân biệt với 1TF
        ichi_raw = float(raw.get("ichi_score", 5.0))
        if bool(raw.get("ichi_bull_3tf", False)):
            ichi_raw = min(10.0, ichi_raw + 0.5)   # Bonus nhỏ confirm 3TF
        s["ichimoku"] = round(ichi_raw, 2)

        # ── DMI/Aroon (v5.6) ─────────────────────────────────────────────────
        # Base 5.0. DMI direction được GATE bởi ADX strength (direction không
        # đáng tin nếu ADX thấp). Aroon là lớp xác nhận cộng thêm độc lập.
        dmi_sc = 5.0
        gate = {"strong": 3.0, "weak": 1.5, "none": 0.5}[raw.get("dmi_strength", "none")]
        dmi_sc += gate if raw.get("dmi_bullish", False) else -gate
        dmi_sc += {"strong_bull": 2.0, "emerging_bull": 1.0, "neutral": 0.0,
                   "emerging_bear": -1.0, "strong_bear": -2.0}[raw.get("aroon_state", "neutral")]
        s["dmi_aroon"] = round(max(0.0, min(10.0, dmi_sc)), 2)

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = raw.get("rsi", 50.0)
        if mode == "hold":
            # Hold: muốn RSI ổn định (48-58) không quá nóng hay quá lạnh
            if   rsi <= 32: s["rsi"] = 6.5
            elif rsi <= 48: s["rsi"] = 8.5
            elif rsi <= 58: s["rsi"] = 9.0
            elif rsi <= 68: s["rsi"] = 5.5
            elif rsi <= 78: s["rsi"] = 2.0
            else:           s["rsi"] = 0.5
        else:
            # Swing: muốn RSI vừa từ oversold lên (28-45)
            if   rsi <= 28: s["rsi"] = 9.5
            elif rsi <= 45: s["rsi"] = 8.0 + (45 - rsi) / 8.5
            elif rsi <= 55: s["rsi"] = 6.5
            elif rsi <= 68: s["rsi"] = 4.0
            elif rsi <= 78: s["rsi"] = 1.5
            else:           s["rsi"] = 0.5

        # ── EMA %distance ─────────────────────────────────────────────────────
        # ema_pct = % giá cách EMA20: -6% → score 0, +6% → score ~11
        s["ema"] = clamp((raw.get("ema_pct", 0) + cfg.sc_ema_shift) / cfg.sc_ema_scale)

        # ── Stochastic ────────────────────────────────────────────────────────
        k_ = raw.get("stoch_k", 50.0)
        if   k_ < 22: s["stoch"] = 8.8
        elif k_ < 45: s["stoch"] = 7.0
        elif k_ < 75: s["stoch"] = 4.2
        else:         s["stoch"] = 1.2
        if raw.get("stoch_cross_up"): s["stoch"] = min(10.0, s["stoch"] + 1.8)

        # ── MACD ─────────────────────────────────────────────────────────────
        s["macd"] = clamp(5.0 + raw.get("macd_hist_pct", 0) * cfg.sc_macd_hist_mult)
        if raw.get("macd_cross_up"): s["macd"] = min(10.0, s["macd"] + 1.7)

        # ── VP Position ───────────────────────────────────────────────────────
        vp_map = {"breakout": 9.2, "value_area": 6.8, "at_poc": 5.5,
                  "below_value": 2.2, "unknown": 4.5}
        vp_score = vp_map.get(raw.get("vp_signal", "unknown"), 4.5)

        # v5.3: BB Squeeze + Breakout = bonus thêm +1.5 (tiếp năng lượng tích lũy)
        if sq and raw.get("vp_signal") == "breakout":
            vp_score = min(10.0, vp_score + 1.5)
        s["vp_position"] = vp_score

        # ── Trend ─────────────────────────────────────────────────────────────
        tr_map = {"uptrend": 9.5, "weak_up": 6.8, "sideway": 4.2, "downtrend": 0.8}
        s["trend"] = tr_map.get(raw.get("trend_long", "sideway"), 4.2)

        # ── RS Score ─────────────────────────────────────────────────────────
        s["rs_score"] = clamp(raw.get("rs", 1.0) * cfg.sc_rs_mult + cfg.sc_rs_base)

        # ── Tổng hợp ─────────────────────────────────────────────────────────
        tw       = sum(weights.values()) or 1
        weighted = sum(s.get(k, 4.5) * weights.get(k, 0) for k in weights)
        total    = round(weighted / tw * regime_mult + bonus, 2)

        return {k_: round(v, 2) for k_, v in s.items()}, total

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — POSITION SIZER                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# "Mua bao nhiêu?" quan trọng hơn "Mua cổ phiếu nào?"
# Mua đúng mã, sai size → vẫn mất tài khoản (one bad trade wipes gains)
#
# 3 phương pháp, lấy MIN (conservative):
#
# 1. Fixed Risk:
#    Shares = (capital × risk%) / (entry - stoploss)
#    → Kiểm soát mức lỗ tối đa bằng tiền tuyệt đối
#
# 2. Volatility Adjustment:
#    vol_adj = min(1.0, 2.0 / ATR%)
#    → Mã biến động cao (ATR 4%) → giảm size 50%
#    → Tránh tình trạng stop bị bắn bởi noise bình thường
#
# 3. Kelly Fraction (1/4 Kelly):
#    Kelly = (win% × RR - loss%) / RR
#    Shares = capital × kelly_fraction / entry
#    → Giới hạn size theo xác suất thắng ước tính
#    → Dùng 1/4 Kelly (conservative) vì win rate là estimate không chắc
#
# v5.2: Kelly calculation dùng win_rate từ WinRateEstimator thay vì estimate thô

class PositionSizer:
    """
    Tính size lệnh: Kelly-fractional + Fixed Risk + Volatility Adjustment.

    Lấy MIN của 3 phương pháp để đảm bảo bảo toàn vốn.
    Cap tối đa 20% NAV/mã tránh concentration risk.
    Làm tròn xuống 100 cổ (lot chuẩn HOSE).
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def calculate(self, entry: float, stoploss: float, score: float,
                  atr_ratio: float, win_rate: float = 50.0) -> dict:
        cfg      = self.cfg
        capital  = cfg.capital
        risk_amt = capital * cfg.risk_per_trade       # VNĐ cho phép lỗ
        sl_dist  = max(entry - stoploss, entry * 0.005)  # Khoảng cách SL, min 0.5%

        # Method 1: Fixed Risk
        shares_risk = risk_amt / sl_dist

        # Method 2: Volatility Adjustment
        # ATR 2% = baseline không điều chỉnh
        # ATR 4% = giảm 50% size (biến động gấp đôi bình thường)
        vol_adj      = min(1.0, 2.0 / max(atr_ratio, 0.5))
        shares_vol   = shares_risk * vol_adj

        # Method 3: Kelly Fraction
        # win_rate từ WinRateEstimator (đã tích hợp pattern + regime)
        win_est      = win_rate / 100.0
        rr_est       = 2.0    # Assume RR = 2 (conservative baseline)
        kelly        = max(0, (win_est * rr_est - (1 - win_est)) / rr_est)
        kelly_frac   = kelly * cfg.kelly_fraction   # 1/4 Kelly
        shares_kelly = (capital * kelly_frac) / entry

        # Lấy MIN (conservative) + cap max position
        shares_raw = min(shares_vol, shares_kelly)
        max_shares = (capital * cfg.max_position_pct) / entry
        shares     = min(shares_raw, max_shares)

        # Làm tròn xuống lot 100 cổ (chuẩn HOSE)
        shares_lot = int(shares // 100) * 100
        if shares_lot <= 0: shares_lot = 100

        cost        = shares_lot * entry
        risk_actual = shares_lot * sl_dist
        pct_nav     = cost / capital * 100

        return {
            "shares":       shares_lot,
            "cost_vnd":     round(cost, 0),
            "pct_nav":      round(pct_nav, 1),
            "risk_vnd":     round(risk_actual, 0),
            "risk_pct_nav": round(risk_actual / capital * 100, 2),
            "kelly":        round(kelly_frac * 100, 1),
            "vol_adj":      round(vol_adj, 2),
        }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — SIGNAL BUILDER                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Tính Smart Entry, Levels, Filters, Verdict.
#
# Smart Entry: không mua đuổi theo giá hiện tại, đặt lệnh chờ ở vùng tốt hơn
#   - Hold:    chờ về VAL (vùng giá trị thấp) hoặc dip 1.5×ATR
#   - Breakout: chờ retest VAH để tránh mua đỉnh
#   - At POC:  mua sát POC — điểm cân bằng volume cao nhất
#   - EMA20:   mua test đường MA động
#   - Default: dip nhỏ theo ATR
#
# Stoploss:
#   Swing: sl_base × 0.993 (buffer anti-hunt), max 2.0×ATR
#   Hold:  max(sl_base × 0.97, entry - 3×ATR, entry × 0.92) → không cut > 8%
#   Lý do Hold lỏng hơn: T+2.5 có rủi ro bị rũ bởi MM trong ngắn hạn
#
# Hard filters (apply_filters):
#   v5.2 thêm: VSA PHÂN PHỐI → không mua T+
#              VSA CẠN CẦU + breakout → không mua (bull trap)
#
# R:R Gate:
#   Swing: T2/risk ≥ 2.0, Hold: T2/risk ≥ 3.0

class SignalBuilder:
    """
    Tính Smart Entry, Levels, Filters, Verdict.
    v5.2: Thêm 2 hard filter VSA vào apply_filters.
    v5.3 updates:
      - smart_entry: ưu tiên điểm entry theo divergence và BB Squeeze
      - levels: dùng bb_upper cho T1 khi sideways; ép target khi bear_div
      - classify_horizon: cộng/trừ điểm theo divergence và vol_dry_up+squeeze
      - apply_filters: giữ nguyên VSA hard filters từ v5.2
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def smart_entry(self, df: pd.DataFrame, raw: dict, vp: dict, mode: str) -> dict:
        """
        Xác định vùng entry tốt nhất.

        v5.4 Priority Order (Ichimoku phong cách Trịnh Phát):
        1. Bull Divergence kép (RSI+MACD) → mua ngay, xác suất cao nhất
        2. BB Squeeze + Breakout → mua đuổi an toàn
        3. Ichimoku: Pullback về Kijun Daily (vùng vàng) khi trên Kumo
        4. Ichimoku: Giá test mép trên Kumo (SpanA/B) → hỗ trợ mây
        5. Hold mode + VAL → chờ về vùng giá trị
        6. Retest VAH (breakout không squeeze) → tránh mua đỉnh
        7. Bull Divergence đơn → entry gần giá
        8. BB Lower làm support ưu tiên
        9. Các trường hợp khác (POC, EMA20, ATR dip)
        """
        close   = float(df["close"].iloc[-1])
        atr     = raw.get("atr_abs", close * 0.01)
        poc     = vp.get("poc"); vah = vp.get("vah"); val = vp.get("val")
        sig     = raw.get("vp_signal", "unknown")
        ema20   = raw.get("ema20", close)
        bb_lower    = raw.get("bb_lower")
        bb_squeeze  = bool(raw.get("is_bb_squeeze", False))
        rsi_bull_div  = bool(raw.get("rsi_bull_div", False))
        macd_bull_div = bool(raw.get("macd_bull_div", False))
        pb      = self.cfg.entry_pullback_pct / 100
        amult   = self.cfg.entry_atr_mult

        # Ichimoku levels
        kijun_d    = raw.get("ichi_kijun_d")
        kijun_w    = raw.get("ichi_kijun_w")
        kumo_bot_d = raw.get("ichi_span_b_d")   # Mép dưới Kumo Daily
        above_d    = bool(raw.get("ichi_above_cloud_d", False))
        above_w    = bool(raw.get("ichi_above_cloud_w", False))

        # ── Priority 1: Bullish Divergence ───────────────────────────────────
        # Phân kỳ dương xác nhận đáy → không cần chờ, mua gần giá hiện tại
        # Entry = Close - 0.2×ATR (chỉ lùi nhẹ để tránh mua đúng đỉnh nến)
        # Kép (RSI + MACD) → tin cậy hơn → entry sát hơn (0.15×ATR)
        if rsi_bull_div and macd_bull_div:
            # Nếu đồng thời có Kijun Daily gần → entry tại Kijun (tốt hơn)
            if kijun_d and above_d and abs(close - kijun_d) / close < 0.025:
                entry = round(kijun_d * 1.003, 2)
                note  = "💚 Kijun Daily + Phân kỳ dương kép — Mua mạnh"
            else:
                entry = round(close - atr * 0.15, 2)
                note  = "💚 Mua ngay — Phân kỳ dương kép xác nhận đáy"
            return {"entry": entry, "entry_note": note}

        # ── Priority 2: BB Squeeze + Breakout ────────────────────────────────
        # Squeeze = tích lũy dồn nén, breakout = năng lượng thoát ra
        # Không cần chờ pullback — mua đuổi an toàn, momentum sẽ kéo tiếp
        if bb_squeeze and sig == "breakout":
            entry = round(close * 1.002, 2)  # +0.2% để chắc chắn breakout xác nhận
            note  = "🗜 Squeeze Breakout — Mua đuổi an toàn"
            return {"entry": entry, "entry_note": note}

        # ── Priority 3: Ichimoku — Pullback về Kijun Daily (Trịnh Phát style) ─
        # Kijun Daily = đường cân bằng ngắn hạn, hỗ trợ mạnh khi trên Kumo
        # Điều kiện: trên Kumo Daily + giá cách Kijun ≤ 1.5×ATR
        if (kijun_d and above_d and
                kijun_d < close and
                (close - kijun_d) <= atr * 1.5):
            entry = round(kijun_d * 1.003, 2)   # +0.3% buffer nhỏ
            note  = "☁️ Pullback Kijun Daily — Hỗ trợ mây mạnh"
            return {"entry": entry, "entry_note": note}

        # ── Priority 4: Ichimoku — Test mép trên Kumo (SpanA/B) ──────────────
        # Khi giá pullback từ trên xuống test mép mây → hỗ trợ động rất mạnh
        # (Đặc biệt khi mây dày ≥ 2% → "tường thành" theo Trịnh Phát)
        if (kumo_bot_d and not above_d and
                raw.get("ichi_in_cloud_d") and
                (close / kumo_bot_d - 1) < 0.025):
            entry = round(kumo_bot_d * 1.005, 2)
            note  = "☁️ Test mép Kumo — Mua tại ngưỡng mây"
            return {"entry": entry, "entry_note": note}

        # ── Priority 5: Hold mode → VAL hoặc Kijun Weekly ────────────────────
        if mode == "hold":
            # Kijun Weekly = hỗ trợ trung hạn quan trọng hơn VAL
            if kijun_w and kijun_w < close and (close - kijun_w) / close < 0.05:
                entry = round(kijun_w * 1.005, 2)
                note  = "📅 Kijun Weekly — Hỗ trợ trung hạn (Hold)"
            elif val and val < close * 0.98:
                entry = round(val * 1.005, 2)
                note  = "Chờ về vùng VAL"
            else:
                entry = round(close - atr * 1.5, 2)
                note  = "Chờ nhịp dip sâu"
            return {"entry": entry, "entry_note": note}

        # ── Priority 6: Breakout → Retest VAH ────────────────────────────────
        if sig == "breakout":
            if vah and vah < close:
                entry, note = round(vah * 1.003, 2), "Retest VAH (tránh mua đỉnh)"
            else:
                entry, note = round(close * (1 - pb), 2), f"Pullback {pb*100:.1f}%"
            return {"entry": entry, "entry_note": note}

        # ── Priority 7: Phân kỳ dương đơn ────────────────────────────────────
        if rsi_bull_div or macd_bull_div:
            entry = round(close - atr * 0.20, 2)
            src   = "RSI" if rsi_bull_div else "MACD"
            note  = f"🟢 Mua sớm — {src} Phân kỳ dương"
            return {"entry": entry, "entry_note": note}

        # ── Priority 8: At POC ────────────────────────────────────────────────
        if sig == "at_poc" and poc:
            entry, note = round(poc * 0.997, 2), "Mua sát POC"
            return {"entry": entry, "entry_note": note}

        # ── Priority 9: EMA20 support ─────────────────────────────────────────
        if abs(close - ema20) / close < 0.015:
            entry, note = round(ema20 * 1.002, 2), "Test hỗ trợ EMA20"
            return {"entry": entry, "entry_note": note}

        # ── Priority 10: BB Lower support ────────────────────────────────────
        if bb_lower and bb_lower > close * 0.90:
            entry = round(bb_lower * 1.003, 2)
            note  = "⬇ Chờ về BB Lower (support động)"
            return {"entry": entry, "entry_note": note}

        # ── Fallback: ATR dip ─────────────────────────────────────────────────
        entry, note = round(close - atr * amult, 2), f"Dip {amult}×ATR"
        return {"entry": entry, "entry_note": note}

    def find_swing_low(self, df: pd.DataFrame, lookback: int = 18) -> float:
        """
        Tìm đáy Swing Low gần nhất để đặt stoploss.

        Pivot Low: nến thấp hơn 2 nến trước và 2 nến sau.
        Chỉ xét pivot ở cuối lookback (>40% từ đầu window)
        để tránh dùng đáy quá cũ làm SL quá xa.
        """
        r  = df["low"].tail(lookback).reset_index(drop=True)
        sl = float(r.min())
        for i in range(2, len(r) - 2):
            if (r.iloc[i] < r.iloc[i-1] and r.iloc[i] < r.iloc[i+1] and
                r.iloc[i] <= r.iloc[i-2] and r.iloc[i] <= r.iloc[i+2]):
                if i > len(r) * 0.4:
                    sl = float(r.iloc[i])
                    break
        return round(sl, 2)

    def find_resistance(self, df: pd.DataFrame, mode: str = "swing") -> float:
        """
        Tìm kháng cự ngắn hạn để capping target.

        Swing: 15 phiên (kháng cự gần, tránh target bị chặn sớm)
        Hold:  30 phiên (kháng cự xa hơn, target cần bỏ qua kháng cự ngắn)
        """
        lookback = 15 if mode == "swing" else 30
        return float(df["high"].tail(lookback).max())

    def levels(self, df: pd.DataFrame, raw: dict, vp: dict,
               se: dict, mode: str) -> dict:
        """
        Tính SL và TP cho cả swing và hold.

        v5.4 additions — T+ Target ≥ 9%:
        - T1 = entry + risk × cfg.swing_t1_mult (2.0, tăng từ 1.5)
        - T2 = entry + risk × cfg.swing_t2_mult (3.5, tăng từ 2.5)
        - ATR% filter: mã ATR < cfg.min_atr_pct → không đủ biên động
        - min_tp_pct filter: nếu T1 < entry × (1 + min_tp_pct/100)
          → tự động mở rộng target lên mức tối thiểu
        - Ichimoku T2: nếu Kijun Weekly ở trên entry đủ xa → dùng làm T2
          (Kijun W = target tự nhiên T+ theo Trịnh Phát)
        - Ichimoku T1: nếu Kumo top W ở trên → dùng làm T1 nếu ≥ min_tp
        - bear_div kép: ép target ngắn lại như v5.3
        """
        atr     = raw.get("atr_abs", 0)
        sl_base = self.find_swing_low(df, 18)
        resist  = self.find_resistance(df, mode)
        entry   = se["entry"]
        close   = float(df["close"].iloc[-1])
        bb_upper = raw.get("bb_upper")
        rsi_bear_div  = bool(raw.get("rsi_bear_div", False))
        macd_bear_div = bool(raw.get("macd_bear_div", False))
        bear_div_any  = rsi_bear_div or macd_bear_div
        bear_div_both = rsi_bear_div and macd_bear_div

        # Ichimoku levels cho target
        kijun_w   = raw.get("ichi_kijun_w")
        kijun_m   = raw.get("ichi_kijun_m")
        resist_1  = raw.get("ichi_resist_1")   # Kháng cự Ichi gần nhất
        resist_2  = raw.get("ichi_resist_2")   # Kháng cự Ichi thứ 2
        above_d   = bool(raw.get("ichi_above_cloud_d", False))

        # Tham số từ cfg
        t1_mult   = self.cfg.swing_t1_mult   # 2.0
        t2_mult   = self.cfg.swing_t2_mult   # 3.5
        min_tp    = self.cfg.min_tp_pct      # 9.0%
        min_atr   = self.cfg.min_atr_pct     # 1.5%

        # ── ATR filter flag (không block, chỉ ghi nhận) ──────────────────────
        atr_too_low = bool(raw.get("atr_ratio", 0) < min_atr)

        # ── Swing SL ─────────────────────────────────────────────────────────
        sl_sw = round(sl_base * 0.993, 2)
        if (entry - sl_sw) > atr * 2.2:
            sl_sw = round(entry - atr * 2.0, 2)

        # ── Hold SL ───────────────────────────────────────────────────────────
        sl_hd = max(
            round(sl_base * 0.97, 2),
            round(entry - atr * 3.0, 2),
            round(entry * 0.92, 2),
        )

        risk_sw = max(entry - sl_sw, entry * 0.005)
        risk_hd = max(entry - sl_hd, entry * 0.005)

        # ── Swing targets ─────────────────────────────────────────────────────
        bear_div_note = ""
        if bear_div_both:
            t1_sw = round(entry + atr * 1.2, 2)
            t2_sw = round(entry + atr * 1.8, 2)
            bear_div_note = "⚠️ PHÂN KỲ ÂM KÉP — Chốt siêu ngắn!"
        elif bear_div_any:
            t1_sw = round(entry + risk_sw * 1.2, 2)
            t2_sw = round(entry + risk_sw * 1.5, 2)
            src   = "RSI" if rsi_bear_div else "MACD"
            bear_div_note = f"⚠️ {src} Phân kỳ âm — Ưu tiên chốt ngắn!"
        else:
            # ── v5.4: T1 tính với multiplier 2.0 (tăng từ 1.5) ──────────────
            t1_sw = round(entry + risk_sw * t1_mult, 2)

            # ── v5.4: Ichimoku kháng cự gần nhất làm T1 nếu hợp lý ──────────
            # Điều kiện: kháng cự Ichi > entry * (1 + min_tp/100) ≈ 9%+
            # Và < entry * 1.20 (không quá xa, còn khả thi trong T+)
            if resist_1 and (entry * (1 + min_tp / 100) <= resist_1 <= entry * 1.20):
                # Dùng Ichi resist làm T1 nếu cao hơn R:R-based T1
                t1_sw = round(max(t1_sw, resist_1 * 0.995), 2)
            elif bb_upper:
                vp_sig = raw.get("vp_signal", "unknown")
                if vp_sig in ("value_area", "at_poc") and entry < bb_upper < entry * 1.15:
                    t1_sw = round(min(t1_sw, bb_upper * 0.995), 2)

            # Cap tại kháng cự tĩnh
            if resist > entry * 1.01:
                t1_sw = round(min(t1_sw, resist * 0.99), 2)

            # ── v5.4: Đảm bảo T1 ≥ min_tp_pct% từ entry ─────────────────────
            # Nếu không đạt tối thiểu 9% → đẩy lên, vì dưới ngưỡng này
            # phí giao dịch + rủi ro không xứng với công sức T+
            min_t1 = round(entry * (1 + min_tp / 100), 2)
            if t1_sw < min_t1:
                t1_sw = min_t1

            # ── v5.4: T2 = Kijun Weekly nếu đủ xa và hợp lý ─────────────────
            # Kijun W = target T+ tự nhiên vì là đường cân bằng trung hạn
            # Điều kiện: Kijun W > entry × 1.12 (xa hơn T1) và < entry × 1.35
            t2_sw = round(entry + risk_sw * t2_mult, 2)
            if (kijun_w and above_d and
                    entry * 1.12 < kijun_w < entry * 1.35):
                t2_sw = round(max(t2_sw, kijun_w * 0.995), 2)
            elif resist_2 and entry * 1.12 < resist_2 < entry * 1.35:
                t2_sw = round(max(t2_sw, resist_2 * 0.995), 2)

        if t2_sw <= t1_sw:
            t2_sw = round(t1_sw + max(atr, entry * 0.03), 2)  # Tối thiểu +3%

        rr_sw = round((t2_sw - entry) / risk_sw, 2) if risk_sw > 0 else 0.0

        # ── Hold targets ──────────────────────────────────────────────────────
        t1_hd = round(entry + risk_hd * self.cfg.hold_t1_mult, 2)
        t2_hd = round(entry + risk_hd * self.cfg.hold_t2_mult, 2)
        # Hold T2: dùng Kijun Monthly nếu có (target dài hạn)
        if kijun_m and kijun_m > t1_hd * 1.05:
            t2_hd = round(max(t2_hd, kijun_m * 0.995), 2)
        if t2_hd <= t1_hd:
            t2_hd = round(t1_hd + atr * 2, 2)
        rr_hd = round((t2_hd - entry) / risk_hd, 2) if risk_hd > 0 else 0.0

        cutloss_sw_pct = round((entry - sl_sw) / entry * 100, 2)
        cutloss_hd_pct = round((entry - sl_hd) / entry * 100, 2)
        t1_sw_pct = round((t1_sw - entry) / entry * 100, 2)

        if mode == "swing":
            rr_ok = rr_sw >= self.cfg.min_rr_swing
            if atr_too_low:
                action = f"⚠️ ATR thấp ({raw.get('atr_ratio',0):.1f}%<{min_atr}%) — Biên hẹp"
            elif not rr_ok or cutloss_sw_pct > 8.0:
                action = f"⛔ R:R={rr_sw:.1f}<{self.cfg.min_rr_swing}"
            else:
                action = f"✅ BUY T+  TP1:{t1_sw_pct:.1f}%"
        else:
            rr_ok  = rr_hd >= self.cfg.min_rr_hold
            action = "✅ BUY HOLD" if rr_ok and cutloss_hd_pct <= 8.0 else f"⛔ R:R={rr_hd:.1f}<{self.cfg.min_rr_hold}"

        if bear_div_note:
            action = bear_div_note

        return {
            "entry": entry, "entry_note": se["entry_note"],
            "stoploss": sl_sw, "cutloss_sw_pct": cutloss_sw_pct,
            "swing_t1": t1_sw, "swing_t2": t2_sw, "swing_rr": rr_sw,
            "swing_t1_pct": t1_sw_pct,
            "swing_sl_pct": round((sl_sw - entry) / entry * 100, 2),
            "hold_sl": sl_hd, "cutloss_hd_pct": cutloss_hd_pct,
            "hold_t1": t1_hd, "hold_t2": t2_hd, "hold_rr": rr_hd,
            "hold_t1_pct": round((t1_hd - entry) / entry * 100, 2),
            "action": action, "rr_ok": rr_ok,
            "bear_div_note": bear_div_note,
            "atr_too_low": atr_too_low,
        }

    def classify_horizon(self, raw: dict) -> str:
        """
        Phân loại T+ / Hold / Tiềm năng dựa trên điều kiện kỹ thuật.

        Hold score: giá vs SMA50/200, RSI, CMF+OBV, RS
        T+ score:   volume ratio, RSI không quá mua, EMA, cross signals
        → Tổ hợp → 5 nhãn từ 'T+ & Hold' đến 'Chờ thêm'

        v5.4 additions:
        - Ichimoku 3TF bull: +3 cho cả hold và tp
        - Ichimoku above D+W: hold +2
        - ichi_tk_cross_up: tp +1
        - ichi_below_cloud_d: tp -2 (không nên đánh T+ khi dưới mây)

        v5.3 additions:
          - vol_dry_up + is_bb_squeeze: T+ +2 (setup nổ cạn cung)
          - macd_bull_div: Hold +2 (tín hiệu đảo chiều trung hạn)
          - rsi_bear_div: T+ -3 (momentum cảnh báo)
        """
        close  = raw.get("close", 0); sma50 = raw.get("sma50", close)
        hold = 0
        if sma50 and close > sma50:                                   hold += 1
        if raw.get("sma200") and close > raw.get("sma200", 0):       hold += 1
        if 38 <= raw.get("rsi", 50) <= 65:                           hold += 1
        if raw.get("cmf", 0) > 0.025 and raw.get("obv_signal") == 1: hold += 2
        if raw.get("rs", 1.0) >= 1.0:                                hold += 1
        # v5.3: MACD bull div = tín hiệu hold tốt
        if bool(raw.get("macd_bull_div", False)):                     hold += 2

        tp = 0
        vr = raw.get("volume_ratio", 1.0)
        if vr >= 1.35:                                                tp += 2
        if raw.get("rsi", 50) < 68:                                  tp += 1
        if raw.get("ema_pct", 0) > -3.5:                             tp += 1
        if raw.get("stoch_cross_up") or raw.get("macd_cross_up") or vr >= 1.8: tp += 2
        if raw.get("vsa_signal") in ("🚀 NỔ VOL", "🔨 RÚT CHÂN"):   tp += 2
        if raw.get("vsa_signal") == "🩸 CẠN VOL":                    hold += 1
        # v5.3: Vol dry up + BB Squeeze = setup nổ → cộng T+
        if bool(raw.get("vol_dry_up", False)) and bool(raw.get("is_bb_squeeze", False)):
            tp += 2
        # v5.3: RSI bull div = T+ tốt (bắt đáy)
        if bool(raw.get("rsi_bull_div", False)):                      tp += 1
        # v5.3: RSI bear div = cảnh báo, trừ T+
        if bool(raw.get("rsi_bear_div", False)):                      tp -= 3


        # ── v5.4: Ichimoku alignment ─────────────────────────────────────────
        if bool(raw.get("ichi_bull_3tf", False)):
            hold += 3
            tp   += 3
        elif (bool(raw.get("ichi_above_cloud_d", False)) and
              bool(raw.get("ichi_above_cloud_w", False))):
            hold += 2
            tp   += 1
        elif bool(raw.get("ichi_above_cloud_d", False)):
            hold += 1
        if bool(raw.get("ichi_tk_cross_up_d", False)):                tp += 1
        if bool(raw.get("ichi_below_cloud_d", False)):                tp -= 2
        if bool(raw.get("ichi_bear_3tf", False)):
            hold -= 2
            tp   -= 2

        # ── v5.6: DMI/Aroon alignment ───────────────────────────────────────
        if bool(raw.get("dmi_bullish")) and raw.get("dmi_strength") == "strong":
            hold += 2; tp += 1
        if raw.get("aroon_state") == "emerging_bull":
            tp += 2   # Aroon bắt sớm điểm khởi phát trend — hợp T+ swing
        if raw.get("aroon_state") == "strong_bear" or (
            not raw.get("dmi_bullish") and raw.get("dmi_strength") == "strong"):
            hold -= 2; tp -= 2

        if hold >= 4 and tp >= 4:  return "🔥 T+ & Hold"
        if hold >= 4:              return "📈 Hold"
        if tp >= 4:                return "⚡ T+ only"
        if hold >= 3 or tp >= 3:   return "🔄 Tiềm năng"
        return                            "⏳ Chờ thêm"

    def liquidity_ok(self, raw: dict, cfg: QuantConfig) -> tuple[bool, str]:
        avg_vol = raw.get("avg_vol_20d", 0)
        avg_val = raw.get("avg_val_20d", 0)
        if avg_vol < cfg.min_avg_vol_20d:
            return False, f"Thanh khoản thấp ({avg_vol} cp/ngày), Min AVG Vol = {cfg.min_avg_vol_20d}"
        if avg_val < cfg.min_avg_val_20d:
            return False, f"Giá trị thấp ({avg_val}/ngày), Min AVG VAL = {cfg.min_avg_val_20d}"
        return True, ""

    def apply_filters(self, raw: dict, mode: str, liq_ok: bool,
                      liq_reason: str, dryup: dict) -> tuple[bool, list]:
        """
        Hard filter — loại mã không đủ điều kiện.

        Filters v5.2 mới (VSA-based):
        1. PHÂN PHỐI trong swing mode → Upthrust, xả hàng núp bóng
           → Không mua T+ dù score cao vì tiền đang rút
        2. Breakout + CẠN CẦU → Bull trap
           → Vol không đỡ breakout, sẽ fail và quay lại range

        Wyckoff Spring exception:
        Nếu có Spring (vol đột tăng sau nền khô) → bỏ qua warning vol thấp
        vì đây là vol tốt, không phải vol yếu
        Thêm v5.3:
          - Nếu bear_div kép (RSI + MACD) trong mode swing → warning (không hard block
            vì bear_div chỉ ép target ngắn, không hủy hoàn toàn setup)
        """
        reasons = []
        if not liq_ok: reasons.append(liq_reason)
        if raw.get("volume_ratio", 0) < 0.5: reasons.append(f"Vol thấp ({raw.get('volume_ratio',0):.2f}x)")

        rsi = raw.get("rsi", 50)
        if rsi > 85: reasons.append(f"RSI quá mua ({rsi:.0f})")
        elif rsi > 75 and mode == "swing": reasons.append(f"RSI cao T+ ({rsi:.0f})")

        if raw.get("ema_pct", 0) < -9: reasons.append(f"Dưới EMA20 sâu ({raw.get('ema_pct',0):.1f}%)")
        if raw.get("cmf", 0) < -0.20:  reasons.append(f"CMF âm nặng ({raw.get('cmf',0):.3f})")
        if mode == "hold" and raw.get("trend_long") == "downtrend": reasons.append("Downtrend dài hạn")
        if mode == "hold" and raw.get("obv_signal") == -1: reasons.append("OBV suy yếu")
        if raw.get("stoch_k", 50) > 75: reasons.append(f"Stoch quá mua ({raw.get('stoch_k',50):.0f})")

        # ── VSA Hard Filters (v5.2) ──────────────────────────────────────────
        vsa = raw.get("vsa_signal", "—")

        # Filter 1: Upthrust / Phân phối
        # Vol lớn nhưng đóng thấp trong nến → Smart Money đang xả
        # Không mua T+ dù các indicator khác tốt
        if mode == "swing" and vsa == "⚠️ PHÂN PHỐI":
            reasons.append("⚠️ VSA: Upthrust / Xả hàng núp bóng")

        # Filter 2: Breakout không có Vol → bull trap
        # Breakout phải kèm Vol lớn, nếu CẠN CẦU thì sẽ fail
        if raw.get("vp_signal") == "breakout" and vsa == "🥀 CẠN CẦU":
            reasons.append("🥀 VSA: Breakout thiếu Vol (Bull Trap)")

        # Spring exception: bỏ qua warning vol thấp nếu có Wyckoff Spring
        if dryup.get("spring_flag") and any("Vol thấp" in r for r in reasons):
            reasons = [r for r in reasons if "Vol thấp" not in r]

        return len(reasons) == 0, reasons

    @staticmethod
    def new_buy_tag(verdict_str: str, raw: dict, ok: bool) -> str:
        """
        v5.6: Tag 3 mức cho quyết định MỞ VỊ THẾ MỚI — verdict() có sẵn
        (5 tier, mode-aware) làm nền, DMI/Aroon làm lớp xác nhận xu hướng.

        Nguyên tắc (tôn trọng risk-first của apply_filters có sẵn):
        - ok=False (hard filter fail / RR không đạt) → LUÔN BỎ QUA,
          DMI/Aroon KHÔNG được override quyết định filter cứng.
        - DMI/Aroon chỉ HẠ 1 bậc khi mâu thuẫn với verdict (bảo vệ thêm),
          hoặc NÂNG QUAN SÁT→MUA khi xác nhận mạnh cả DMI lẫn Aroon
          (KHÔNG bao giờ nâng từ BỎ QUA vì đó là do hard filter).
        """
        if not ok:
            return "❌ BỎ QUA"

        if "MẠNH" in verdict_str:                base = "MUA"
        elif "XEM XÉT" in verdict_str:            base = "QUAN SÁT"
        elif "THEO DÕI" in verdict_str:           base = "QUAN SÁT"
        else:                                     base = "BỎ QUA"

        adx      = raw.get("adx", 0)
        bullish_strong = bool(raw.get("dmi_bullish") and raw.get("dmi_strength") == "strong")
        bearish_strong = bool(not raw.get("dmi_bullish", True) and raw.get("dmi_strength") == "strong")
        aroon_strong_bull = raw.get("aroon_state") == "strong_bull"
        aroon_strong_bear = raw.get("aroon_state") == "strong_bear"
        no_trend = raw.get("dmi_strength") == "none"

        if base == "MUA":
            if bearish_strong or aroon_strong_bear:
                return "🟡 QUAN SÁT (DMI/Aroon nghịch hướng)"
            if no_trend:
                return "🟡 QUAN SÁT (ADX<20 — chưa rõ xu hướng)"
            return "🟢 MUA"

        if base == "QUAN SÁT":
            if bullish_strong and aroon_strong_bull:
                return "🟢 MUA (DMI+Aroon xác nhận mạnh)"
            return "🟡 QUAN SÁT"

        return "❌ BỎ QUA"

    @staticmethod
    def verdict(score: float, passed: bool, mode: str) -> str:
        if not passed: return "🚫 BỊ LỌC"
        if mode == "hold":
            if score >= 8.0: return "💎 HOLD MẠNH"
            if score >= 6.5: return "✅ XEM XÉT HOLD"
            if score >= 5.5: return "👀 THEO DÕI"
            return "❌ BỎ QUA"
        else:
            if score >= 7.5: return "🔥 MUA MẠNH T+"
            if score >= 6.5: return "✅ XEM XÉT T+"
            if score >= 5.0: return "👀 THEO DÕI"
            if score >= 3.5: return "⚠️  YẾU"
            return "❌ BỎ QUA"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10b — POSITION ADVISOR  (NEW v5.5)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Khác với SignalBuilder.verdict() (dành cho người CHƯA mua), class này
# trả lời cho người ĐANG GIỮ: Cắt lỗ / Chốt lời / Giảm tỷ trọng / Gia tăng / Giữ?
#
# Thứ tự ưu tiên (nghiêm trọng nhất trước — giống apply_filters logic):
#   1. 🔴 CẮT LỖ      : vi phạm SL, hoặc downtrend + mất MA ngắn, hoặc Ichi 3TF Bear
#   2. 🟠 CHỐT LỜI/BÁN: đạt T2, hoặc phân kỳ âm kép + quá mua, hoặc rơi khỏi Kumo
#   3. 🟡 GIẢM TỶ TRỌNG: mất MA ngắn hạn nhưng CHƯA vi phạm SL — cảnh báo sớm
#   4. 🟢 GIA TĂNG    : dao găm xác nhận hỗ trợ + trend dài hạn còn tốt
#   5. ⚪ TIẾP TỤC GIỮ: default, không có tín hiệu mạnh theo hướng nào

class PositionAdvisor:
    """
    Khuyến nghị quản lý vị thế cho người ĐANG HOLD cổ phiếu.

    Trả về tuple (action_label, reason_string) — reason luôn giải thích
    CỤ THỂ vì sao ra khuyến nghị đó để người dùng tự đối chiếu, không
    phải black-box.
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def advise(self, raw: dict, lvl: dict, dao_gam: dict, ma_sig: dict) -> tuple[str, str]:
        close = raw.get("close", 0)
        sl_sw = lvl.get("stoploss")
        t2_sw = lvl.get("swing_t2")
        trend = raw.get("trend_long")
        rsi   = raw.get("rsi", 50)
        vsa   = raw.get("vsa_signal", "—")
        rrd   = bool(raw.get("rsi_bear_div", False))
        mrd   = bool(raw.get("macd_bear_div", False))
        lost_short = bool(ma_sig.get("lost_short_trend", False))
        ichi_bear_3tf = bool(raw.get("ichi_bear_3tf", False))
        ichi_below_d  = bool(raw.get("ichi_below_cloud_d", False))

        # ── 1. CẮT LỖ ─────────────────────────────────────────────────────────
        reasons = []
        if sl_sw and close < sl_sw:
            reasons.append(f"Giá {close:.2f} đã xuyên Stoploss ({sl_sw:.2f})")
        if trend == "downtrend" and lost_short:
            reasons.append("Downtrend dài hạn + mất MA9/10 — xu hướng xấu cả 2 khung")
        if vsa == "⚠️ PHÂN PHỐI" and lost_short:
            reasons.append("VSA Phân phối (tiền lớn xả) + mất MA ngắn hạn")
        if ichi_bear_3tf:
            reasons.append("Ichimoku 3TF Bear — downtrend xác nhận cả D/W/M")
        if reasons:
            return "🔴 CẮT LỖ", " | ".join(reasons)

        # ── 2. CHỐT LỜI / BÁN ─────────────────────────────────────────────────
        reasons = []
        if t2_sw and close >= t2_sw:
            reasons.append(f"Đã đạt target T2 ({t2_sw:.2f}) — nên chốt lời theo kế hoạch")
        if rrd and mrd and rsi > 68:
            reasons.append("Phân kỳ âm kép (RSI+MACD) + RSI quá mua — momentum cạn")
        if ichi_below_d and trend not in ("uptrend", "weak_up"):
            reasons.append("Rơi khỏi Kumo Daily trong khi trend dài hạn không còn mạnh")
        if reasons:
            return "🟠 CHỐT LỜI/BÁN", " | ".join(reasons)

        # ── 3. GIẢM TỶ TRỌNG ──────────────────────────────────────────────────
        reasons = []
        if lost_short:
            reasons.append(
                f"Mất MA{self.cfg.ma_short_period_1}/{self.cfg.ma_short_period_2} — "
                "cảnh báo sớm xu hướng ngắn hạn suy yếu, cân nhắc bán 1/3-1/2 vị thế "
                "để bảo toàn lợi nhuận, giữ phần còn lại theo dõi SL"
            )
        if rrd or mrd:
            src = "RSI" if rrd else "MACD"
            reasons.append(f"Phân kỳ âm {src} đơn — momentum đang yếu đi, theo dõi sát")
        if reasons:
            return "🟡 GIẢM TỶ TRỌNG", " | ".join(reasons)

        # ── 4. GIA TĂNG ───────────────────────────────────────────────────────
        reasons = []
        if dao_gam.get("dao_gam_d") or dao_gam.get("dao_gam_w") or dao_gam.get("dao_gam_m"):
            reasons.append(f"{dao_gam.get('dao_gam_label')} — hỗ trợ Kijun được xác nhận")
        if ma_sig.get("touch_ma_support") and trend in ("uptrend", "weak_up"):
            lvl_ma = ma_sig.get("ma_support_level")
            reasons.append(f"Test thành công MA hỗ trợ ({lvl_ma}) trong khi trend dài hạn còn tốt")
        if reasons and trend != "downtrend":
            return "🟢 GIA TĂNG", " | ".join(reasons)

        # ── 5. TIẾP TỤC GIỮ (default) ────────────────────────────────────────
        return "⚪ TIẾP TỤC GIỮ", "Chưa có tín hiệu đảo chiều rõ ràng — giữ theo kế hoạch entry ban đầu"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10c — DIVERGENCE & DAO GĂM ADVISOR  (NEW v5.7)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Biến 2 nguồn tín hiệu thành KHUYẾN NGHỊ đọc được:
#   (1) Phân kỳ RSI-Composite  (CompositeDivergenceEngine, cell 5e)
#   (2) Dao găm Ichimoku Trịnh Phát D / W / M (DaoGamDetector, cell 5d)
#
# Class này KHÔNG tự tính chỉ báo — chỉ đọc các cờ đã có trong dict `raw`:
#   cd_bull_reg_fresh, cd_bull_hid_fresh, cd_bear_reg_fresh, cd_bear_hid_fresh,
#   cd_osc_above_fast, dao_gam_d/w/m, ichi_kijun_d/w/m, trend_long, close
#
# 4 việc:
#   dg_alert()        → cột "Cảnh báo DG": DG ngày/tuần/tháng gắn với khung thời gian
#                       giao dịch (Ngày=ngắn hạn, Tuần=trung hạn, Tháng=dài hạn),
#                       kèm giá Kijun và cảnh báo "sắp về Kijun".
#   adjust_buy_tag()  → chỉnh cột "Khuyến nghị Mua Mới" (chỉ chỉnh khi mã ĐÃ qua bộ lọc).
#   adjust_hold()     → chỉnh cột "Khuyến nghị Hold" cho người đang giữ mã.
#   watch_signal()    → cột "Theo dõi PK/DG": danh sách mã để theo dõi kể cả khi
#                       CHƯA đủ điều kiện mua.
#
# NGUYÊN TẮC AN TOÀN (giống new_buy_tag của DMI/Aroon):
#   - Phân kỳ/DG KHÔNG BAO GIỜ vượt qua hard filter: mã đã "❌ BỎ QUA" giữ nguyên.
#   - Phân kỳ ÂM thường chỉ HẠ 1 bậc (MUA → QUAN SÁT) — bảo vệ thêm.
#   - Chỉ NÂNG QUAN SÁT → MUA khi có HỘI TỤ: phân kỳ dương thường + DG (bất kỳ khung),
#     không ngược xu hướng dài hạn (trend_long ≠ downtrend) và không có phân kỳ âm.
#     Một tín hiệu đơn lẻ là chưa đủ vì tỷ lệ thắng của phân kỳ chưa được kiểm chứng.

class DivergenceAdvisor:
    # Khung Ichimoku → (tên khung, khung thời gian giao dịch tương ứng)
    TF = {"D": ("Ngày", "ngắn hạn"), "W": ("Tuần", "trung hạn"), "M": ("Tháng", "dài hạn")}

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    # ── Tiện ích ────────────────────────────────────────────────────────────
    @staticmethod
    def dg_hits(raw: dict) -> list:
        """Danh sách khung đang có Dao găm, vd ['D','W']."""
        return [tf for tf in ("D", "W", "M") if raw.get(f"dao_gam_{tf.lower()}")]

    # ── (1) Cột "Cảnh báo DG" ────────────────────────────────────────────────
    def dg_alert(self, raw: dict) -> str:
        """
        Ghép cảnh báo Dao găm cho cả 3 khung.
          🗡️ DG <khung> [<hạn>] (Kijun x)  : đã có dao găm (đáy nến chạm Kijun rồi đóng trên Kijun)
          👀 Gần Kijun <khung> +x% [<hạn>] : CHƯA chạm nhưng giá đang nằm trong cfg.dg_near_pct%
                                             phía trên Kijun → có thể sắp về test (chuẩn bị)
        Trả "—" nếu không có gì.
        """
        close = raw.get("close") or 0
        parts, near_pct = [], self.cfg.dg_near_pct
        for tf, (name, horizon) in self.TF.items():
            kijun = raw.get(f"ichi_kijun_{tf.lower()}")
            if raw.get(f"dao_gam_{tf.lower()}"):
                k_txt = f" (Kijun {kijun:.2f})" if kijun else ""
                parts.append(f"🗡️ DG {name} [{horizon}]{k_txt}")
            elif kijun and close > kijun:
                dist = (close / kijun - 1) * 100          # % giá đang cao hơn Kijun
                if dist <= near_pct:
                    parts.append(f"👀 Gần Kijun {name} +{dist:.1f}% [{horizon}]")
        return "  ·  ".join(parts) if parts else "—"

    # ── (2) Chỉnh "Khuyến nghị Mua Mới" ─────────────────────────────────────
    def adjust_buy_tag(self, tag: str, raw: dict) -> str:
        """
        tag: kết quả của SignalBuilder.new_buy_tag (đã gồm DMI/Aroon).
        Quy tắc, xét theo thứ tự:
          1. tag bắt đầu "❌" (hard filter / RR / score không đạt) → GIỮ NGUYÊN.
          2. Có PK âm thường + tag đang "🟢 MUA" → hạ xuống "🟡 QUAN SÁT".
          3. Có PK dương thường + DG + tag đang "🟡 QUAN SÁT" + không downtrend
             + không có PK âm → nâng lên "🟢 MUA (hội tụ)".
          4. Còn lại: giữ tag, gắn hậu tố thông tin (PK+/PK−/DG) để dễ lọc trong Excel.
        """
        if tag.startswith("❌"):
            return tag
        bull  = bool(raw.get("cd_bull_reg_fresh"))
        bear  = bool(raw.get("cd_bear_reg_fresh"))
        dg    = self.dg_hits(raw)
        dg_tx = "+".join(dg)

        if bear and tag.startswith("🟢 MUA"):
            return "🟡 QUAN SÁT (RSI phân kỳ âm — không mua đuổi)"
        if (bull and dg and not bear and tag.startswith("🟡 QUAN SÁT")
                and raw.get("trend_long") != "downtrend"):
            return f"🟢 MUA (hội tụ PK dương RSI + DG {dg_tx})"

        tags = []
        if bull: tags.append("PK+ RSI")
        if bear: tags.append("PK− RSI")
        if dg:   tags.append(f"DG {dg_tx}")
        return tag + (" | " + " · ".join(tags) if tags else "")

    # ── (3) Chỉnh "Khuyến nghị Hold" (người đang giữ) ────────────────────────
    def adjust_hold(self, action: str, reason: str, raw: dict) -> tuple:
        """
        action/reason: kết quả của PositionAdvisor.advise.
        - Không bao giờ ghi đè 🔴 CẮT LỖ hay 🟠 CHỐT LỜI (đó là quyết định rủi ro).
        - PK âm thường đã được PositionAdvisor xử lý qua cờ rsi_bear_div (giờ lấy từ engine mới).
        - Bổ sung: PK âm ẨN khi đang "TIẾP TỤC GIỮ" mà trend chưa uptrend → GIẢM TỶ TRỌNG.
        - Luôn nối thêm câu giải thích phân kỳ (cd_note) vào lý do để đối chiếu chart.
        """
        if (action == "⚪ TIẾP TỤC GIỮ" and raw.get("cd_bear_hid_fresh")
                and raw.get("trend_long") not in ("uptrend", "weak_up")):
            action = "🟡 GIẢM TỶ TRỌNG"
            reason = "PK âm ẩn (RSI-Composite) trong xu hướng chưa tăng — đà giảm có thể tiếp diễn"
        note = raw.get("cd_note")
        if note and note not in reason:
            reason = f"{reason} | {note}"
        return action, reason

    # ── (4) Cột "Theo dõi PK/DG" ────────────────────────────────────────────
    def watch_signal(self, raw: dict, ok: bool) -> str:
        """
        Gom mọi thứ đáng theo dõi thành 1 dòng. Dùng để lọc watchlist trong Excel,
        KỂ CẢ mã chưa qua bộ lọc mua (ok=False) — khi đó thêm ghi chú.
        """
        bull_r, bull_h = raw.get("cd_bull_reg_fresh"), raw.get("cd_bull_hid_fresh")
        bear_r, bear_h = raw.get("cd_bear_reg_fresh"), raw.get("cd_bear_hid_fresh")
        dg = self.dg_hits(raw)
        parts = []

        if bull_r and dg:
            parts.append(f"⭐ HỘI TỤ: PK dương RSI + DG {'+'.join(dg)} — ưu tiên theo dõi điểm vào")
        elif bull_r:
            momentum = ("đà đã quay lên (Composite > SMA13)" if raw.get("cd_osc_above_fast")
                        else "chờ Composite cắt lên SMA13 để xác nhận đà")
            warn = " — NGƯỢC xu hướng dài hạn, rủi ro cao" if raw.get("trend_long") == "downtrend" else ""
            parts.append(f"🟢 PK dương RSI: {momentum}{warn}")
        elif dg:
            parts.append(f"🗡️ Về DG {'+'.join(dg)} — theo dõi phản ứng giá tại Kijun")
        if bull_h:
            parts.append("🟩 PK dương ẩn — nhịp chỉnh trong uptrend có thể kết thúc")
        if bear_r:
            parts.append("🔴 CẢNH BÁO PK âm RSI — không mua đuổi, cân nhắc chốt")
        if bear_h:
            parts.append("🟠 PK âm ẩn — đà giảm có thể tiếp diễn")

        if parts and not ok and (bull_r or bull_h or dg):
            parts.append("(chưa đủ điều kiện lọc mua — chỉ để theo dõi)")
        return " | ".join(parts) if parts else "—"

_TREND = {
    "uptrend":   "🟢 Uptrend",
    "weak_up":   "🔵 Weak Up",
    "sideway":   "🟡 Sideway",
    "downtrend": "🔴 Downtrend",
}
_VP = {
    "breakout":    "⬆ Breakout",
    "value_area":  "◼ Value Area",
    "at_poc":      "⬛ POC",
    "below_value": "⬇ Dưới VAL",
    "unknown":     "—",
}

def _display(df: pd.DataFrame, mode: str):
    return

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 12 — QUANT ENGINE (CONTROLLER)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Orchestrate toàn pipeline theo thứ tự:
#
# Fetch → Clean → VP(30bins) → Trend → RS → MF → VSA → Indicators
# → DryUp → WinRate → Score(+VSA+bonus) → Filter(+VSA hard) → Entry
# → Levels(mode-aware) → PositionSize(win_rate) → Output
#
# Hai entry point chính:
#   engine.run(mode="swing")        → Scan toàn bộ VN100
#   engine.run(["VND","FPT"], mode) → Scan danh sách riêng
#   engine.detail("VND")            → Phân tích chi tiết 1 mã

class QuantEngine:
    """
    Controller chính v5.4 — orchestrate toàn bộ pipeline.

    Pipeline v5.4 vs v5.3:
    + IchimokuEngine.compute() sau MomentumEngine (Step 5b)
    + Ichimoku features merge vào raw dict
    + ScoringEngine nhận ichimoku sub-score (weight 11)
    + WinRateEstimator: 6 Ichimoku patterns mới
    + SignalBuilder: Kijun/Kumo làm entry zone + target T2
    + Output: cột Ichimoku, S1/S2/R1/R2 Ichi levels
    + T+ % hiển thị ngay trong Action để dễ đọc nhanh
    + ScoringEngine nhận cfg → không dùng magic numbers
    """

    def __init__(self, capital: float = 100_000_000, cfg: QuantConfig = None):
        self.cfg     = cfg or QuantConfig(capital=capital)
        self.cfg.capital = capital
        self.dl      = DataLayer(self.cfg)
        self.regime  = RegimeEngine(self.dl)
        self.ind_eng = IndicatorEngine()
        self.mom_eng  = MomentumEngine(self.cfg)   # ✏️ v5.3
        self.ichi_eng = IchimokuEngine(self.cfg)          # ✏️ v5.4
        self.score_e = ScoringEngine(self.cfg)    # ✏️ v5.4: truyền cfg
        self.sig_b   = SignalBuilder(self.cfg)
        self.dao_gam = DaoGamDetector(self.cfg)   # ✏️ v5.5
        self.pos_adv = PositionAdvisor(self.cfg)  # ✏️ v5.5
        self.pos_sz  = PositionSizer(self.cfg)
        self.dryup_d = VolumeDryupDetector(self.cfg)
        self.wr_est  = WinRateEstimator(self.cfg)
        self.cd_eng  = CompositeDivergenceEngine(self.cfg)   # ✏️ v5.7: phân kỳ RSI-Composite
        self.div_adv = DivergenceAdvisor(self.cfg)           # ✏️ v5.7: khuyến nghị PK + Dao găm
        # VSAEngine là stateless (static methods) — không cần instantiate

    def run_pipeline(self, symbol: str, df_vni: pd.DataFrame,
                     mode: str, weights: dict, regime_mult: float,
                     min_score: float, wr_mult: float = 1.0,
                     df_override: pd.DataFrame = None,
                     df_long_override: pd.DataFrame = None) -> Optional[dict]:
        """Pipeline xử lý 1 mã — return dict hoặc None nếu bỏ qua."""

        # Step 1: Fetch data
        # ✏️ v5.7 FIX: trước đây df_override bị bỏ qua và df_long_override bị fetch đè
        # → QuantBacktester chạy trên dữ liệu HIỆN TẠI thay vì dữ liệu lịch sử đã cắt lát.
        # Nay: có override (backtest) thì dùng đúng lát cắt đó, không gọi API.
        df = df_override if df_override is not None else self.dl.fetch(symbol, self.cfg.lookback_short)
        if df is None or len(df) < 20:
            return None

        # Step 2: VP + Trend + RS + MF (VP và df_long chỉ tính/fetch 1 lần)
        vp     = self.ind_eng.volume_profile(df, self.cfg.vp_bins)
        df_lng = df_long_override if df_long_override is not None else self.dl.fetch(symbol, self.cfg.lookback_long)
        trend  = self.ind_eng.long_trend(df_lng)
        rs     = self.ind_eng.relative_strength(df, df_vni)
        mf     = self.ind_eng.money_flow(df)

        # Step 3: VSA
        vsa_signal = VSAEngine.get_latest(df)

        # Step 4: Indicators (lưu _series vào raw để MomentumEngine dùng)
        raw = self.ind_eng.compute_all(df, vp, trend, rs, mf, vsa_signal)

        # Step 5: MomentumEngine (v5.3) — tính BB, Divergence, Hidden Acc
        mom = self.mom_eng.compute(df, raw)
        raw.update(mom)  # Merge momentum features vào raw

        # Step 5b: IchimokuEngine (v5.4) — Daily/Weekly/Monthly + S/R levels
        ichi = self.ichi_eng.compute(df, df_lng)
        raw.update(ichi)

        # Step 5b2: Composite Index RSI Divergence (v5.7) — dùng df_lng (RSI Wilder cần warm-up dài).
        # merge_into_raw GHI ĐÈ rsi_bull_div / rsi_bear_div bằng phân kỳ THƯỜNG của engine này
        # (bản cũ → *_legacy) nên Scoring/WinRate/Entry/Levels/Advisor tự dùng tín hiệu mới.
        cd = self.cd_eng.compute(df_lng)
        self.cd_eng.merge_into_raw(raw, cd)

        # Step 5c: Dao Găm Detector (v5.5) — cần kijun đã có trong raw
        dg = self.dao_gam.detect(
            df_lng, raw.get("ichi_kijun_d"), raw.get("ichi_kijun_w"),
            raw.get("ichi_kijun_m"), raw["close"]
        )

        raw.update(dg)   # ✏️ v5.7: dao_gam_d/w/m + label vào raw cho DivergenceAdvisor

        # Step 5d: MA Signals (v5.5)
        ma_sig = self.ind_eng.ma_signals(df, df_lng, self.cfg)

        # Step 5e: DMI/Aroon (v5.6)
        dmi = self.ind_eng.dmi_aroon(df_lng, self.cfg)
        raw.update(dmi)

        # Step 6: Volume Dry-up (Wyckoff)
        dryup = self.dryup_d.detect(df)

        # Step 7: Win Rate (bao gồm divergence patterns)
        wr = self.wr_est.estimate(raw, dryup, wr_mult)

        # Step 8: Liquidity filter
        liq_ok, liq_reason = self.sig_b.liquidity_ok(raw, self.cfg)

        # Step 9: Hard filter
        passed, reasons = self.sig_b.apply_filters(raw, mode, liq_ok, liq_reason, dryup)

        # Step 10: Score (bao gồm momentum_div sub-score)
        scores, score = self.score_e.score(raw, mode, weights, regime_mult,
                                           bonus=dryup["bonus_score"])

        # Step 11: Entry + Levels (dùng BB và divergence)
        se  = self.sig_b.smart_entry(df, raw, vp, mode)
        lvl = self.sig_b.levels(df, raw, vp, se, mode)

        # Step 11b: Hold Advisory (v5.5) — dành cho người ĐANG GIỮ mã
        hold_action, hold_reason = self.pos_adv.advise(raw, lvl, dg, ma_sig)

        # Step 12: Position Sizing
        sl_use = lvl["stoploss"] if mode == "swing" else lvl["hold_sl"]
        pos = self.pos_sz.calculate(lvl["entry"], sl_use, score,
                                    raw["atr_ratio"], wr["win_rate_est"])

        # Step 13: Signal + Horizon
        verdict = self.sig_b.verdict(score, passed, mode)
        horizon = self.sig_b.classify_horizon(raw)
        rr_check = lvl["swing_rr"] if mode == "swing" else lvl["hold_rr"]
        ok = passed and score >= min_score and rr_check >= self.cfg.min_rr
        new_buy = self.sig_b.new_buy_tag(verdict, raw, ok)

        # ✏️ v5.7: lớp Phân kỳ RSI-Composite + Dao găm D/W/M lên trên khuyến nghị có sẵn
        new_buy               = self.div_adv.adjust_buy_tag(new_buy, raw)
        hold_action, hold_reason = self.div_adv.adjust_hold(hold_action, hold_reason, raw)
        dg_alert              = self.div_adv.dg_alert(raw)
        watch                 = self.div_adv.watch_signal(raw, ok)

        # Ichimoku S/R display helpers
        def _fmt_lvl(v):
            return f"{v:.2f}" if v else "—"

        return {
            "Mã":          symbol,
            "Tín hiệu":    verdict,
            "Khuyến nghị": horizon,
            "Score":       round(score, 2),
            "Win%":        f"{wr['win_rate_est']:.0f}%",
            "Giá":         f"{raw['close']:.2f}",
            "RSI":         f"{raw['rsi']:.2f}",
            "CMF":         f"{round(raw['cmf'], 3):.3f}",
            "RS":          f"{raw['rs']:.1f}",
            "Vol":         f"{raw['volume_ratio']:.1f}x",
            "VSA":         vsa_signal,
            "Phân kỳ":     raw.get("div_label", "—"),      # ✏️ v5.3
            "Ichimoku":    raw.get("ichi_label", "—"),         # ✏️ v5.4
            "Thanh khoản": f"{raw['avg_vol_20d']//1000:.0f}K/ng",
            "Dòng tiền":   raw["mf_label"],
            "Cạn cung":    dryup["dry_up_label"],
            "VP":          _VP.get(raw["vp_signal"], "—"),
            "Trend":       _TREND.get(raw["trend_long"], "—"),
                        # Ichimoku S/R levels — ✏️ v5.4
            "S1":          _fmt_lvl(raw.get("ichi_support_1")),
            "S2":          _fmt_lvl(raw.get("ichi_support_2")),
            "R1":          _fmt_lvl(raw.get("ichi_resist_1")),
            "R2":          _fmt_lvl(raw.get("ichi_resist_2")),
            "Entry":       f"{lvl['entry']:.2f}",
            "Entry zone":  lvl["entry_note"],
            "SL T+":       f"{lvl['stoploss']:.2f}",
            "SL Hold":     f"{lvl['hold_sl']:.2f}",
            "T+ T1":       f"{lvl['swing_t1']:.2f}",
            "T+ T2":       f"{lvl['swing_t2']:.2f}",
            "T+ R:R":      f"1:{lvl['swing_rr']:.1f}",
            "T+ %":        f"{lvl['swing_t1_pct']:.2f}",
            "Hold T1":     f"{lvl['hold_t1']:.2f}",
            "Hold T2":     f"{lvl['hold_t2']:.2f}",
            "Hold R:R":    f"1:{lvl['hold_rr']:.1f}",
            "Hold %":      f"{lvl['hold_t1_pct']:.2f}",
            "Action":      lvl["action"],
            "Lý do":       reasons,
            "Giá trị":     f"{pos['cost_vnd']/1e6:.0f}M",
            "% NAV":       f"{pos['pct_nav']:.2f}%",
            "Rủi ro %":    f"{pos['risk_pct_nav']:.2f}%",
            # Internal
            "_score":        score,
            "_ok":           ok,
            "_win_rate":     wr["win_rate_est"],
            "_dry_strength": dryup["dry_up_strength"],
            "_vsa":          vsa_signal,
            "_div":          raw.get("div_label", "—"),
            "_ichi":         raw.get("ichi_label", "—"),
            "_ichi_3tf":     bool(raw.get("ichi_bull_3tf", False)),
            "_atr_low":      bool(lvl.get("atr_too_low", False)),
            "_t1_pct":       lvl["swing_t1_pct"],
            "_positives":    " | ".join(f"{k}={v:.1f}" for k, v in scores.items() if v >= 6.5),
            "_wr_patterns":  ", ".join(wr["wr_pattern_names"]),
            "_entry_raw":  lvl["entry"],
            "_sl_raw":     sl_use,
            "_tp1_raw":    lvl["swing_t1"] if mode == "swing" else lvl["hold_t1"],
            "_tp2_raw":    lvl["swing_t2"] if mode == "swing" else lvl["hold_t2"],
            # ══════════ CỘT MỚI v5.5 — bên phải để copy Excel ══════════
            "DMI/Aroon":            raw.get("dmi_aroon_label", "—"),
            "Khuyến nghị Mua Mới":  new_buy,
            "Dao Găm D":       "🗡️ Chạm" if dg["dao_gam_d"] else "—",
            "Dao Găm W":       "🗡️ Chạm" if dg["dao_gam_w"] else "—",
            "Dao Găm M":       "🗡️ Chạm" if dg["dao_gam_m"] else "—",
            "Tín hiệu MA":     ma_sig["ma_signal_label"],
            "Khuyến nghị Hold": hold_action,
            "Lý do Hold":      hold_reason,
            # ══════════ CỘT MỚI v5.7 — bên phải cùng để copy Excel ══════════
            "PK RSI (Composite)": raw.get("cd_label", "—"),   # phân kỳ RSI-Composite còn "mới"
            "Cảnh báo DG":        dg_alert,                   # Dao găm Ngày/Tuần/Tháng + gần Kijun
            "Theo dõi PK/DG":     watch,                      # gợi ý theo dõi (kể cả mã chưa qua lọc)
        }

    def run(self, symbols: list = None, mode: str = "swing", top_n: int = 300) -> pd.DataFrame:
        assert mode in ("swing", "hold")
        watchlist = symbols or VN100
        label     = "SWING T+1→T+2.5" if mode == "swing" else "HOLD DÀI HẠN"

        print(f"\n{'═'*72}")
        print(f"  VN QUANT ENGINE v5.7 | {label} | {len(watchlist)} mã")
        print(f"  Vốn: {self.cfg.capital/1e6:.0f}M  Risk/trade: {self.cfg.risk_per_trade*100:.1f}%")
        print(f"  VP:{self.cfg.vp_bins}bins  Lookback:{self.cfg.lookback_short}d/{self.cfg.lookback_long}d  "
              f"BB:{self.cfg.bb_period}  Div:{self.cfg.div_lookback}n")
        print(f"  Ichimoku D({self.cfg.ichi_tenkan}/{self.cfg.ichi_kijun}/{self.cfg.ichi_senkou_b}) "
              f"W(9/26/52) M(3/6/12)  MinTP:{self.cfg.min_tp_pct:.0f}%  MinATR:{self.cfg.min_atr_pct:.1f}%")
        print(f"  {datetime.today().strftime('%d/%m/%Y %H:%M')}")
        print(f"{'═'*72}\n")

        ctx       = self.regime.detect()
        weights   = self.regime.get_dynamic_weights()
        r_mult    = ctx["score_mult"]
        wr_mult   = ctx.get("wr_mult", 1.0)
        min_score = self.regime.get_min_score(mode, self.cfg)

        print(f"\n  Regime: {ctx['regime'].upper()}  Score×{r_mult:.2f}  WR×{wr_mult:.2f}  MinScore:{min_score:.1f}")
        print(f"  Weights: vol={weights.get('volume_ratio',0):.0f}% cmf={weights.get('cmf',0):.0f}% "
              f"vsa={weights.get('vsa',0):.0f}% div={weights.get('momentum_div',0):.0f}% "
              f"ichi={weights.get('ichimoku',0):.0f}% dmi={weights.get('dmi_aroon',0):.0f}% "
              f"trend={weights.get('trend',0):.0f}%\n")

        df_vni  = self.dl.fetch("VNINDEX", self.cfg.lookback_short)
        records = []

        for i, sym in enumerate(watchlist):
            print(f"  [{i+1:3d}/{len(watchlist)}] {sym:6s}...", end=" ", flush=True)
            try:
                rec = self.run_pipeline(sym, df_vni, mode, weights, r_mult, min_score, wr_mult)
                if rec is None:
                    print("❌ thiếu dữ liệu"); continue

                ichi_icon = ("☀️" if rec["_ichi_3tf"] else
                             "🌤" if "Trên Kumo" in rec["_ichi"] else
                             "☁️" if "Trong Mây" in rec["_ichi"] else
                             "🌧" if "Dưới Kumo" in rec["_ichi"] else "")
                div_icon  = ("💚" if "kép" in rec["Phân kỳ"] and "âm" not in rec["Phân kỳ"] else
                             "🟢" if "dương" in rec["Phân kỳ"] else
                             "🔴" if "âm" in rec["Phân kỳ"] else
                             "🗜" if "Squeeze" in rec["Phân kỳ"] else "")
                vsa_icon  = ("🔥" if "NỔ VOL" in rec["VSA"] else
                             "🔨" if "RÚT CHÂN" in rec["VSA"] else
                             "⚠"  if "PHÂN PHỐI" in rec["VSA"] else "")
                dry_icon  = "💎" if rec["_dry_strength"] >= 3 else "📦" if rec["_dry_strength"] == 2 else ""
                ok_icon   = ("🔥" if "🔥" in rec["Tín hiệu"] or "💎" in rec["Tín hiệu"]
                             else "✅" if rec["_ok"] else "·")
                atr_warn  = "⚡" if rec["_atr_low"] else ""
                t1_disp   = f"T+%:{rec['_t1_pct']:.1f}%" if rec["_t1_pct"] else ""

                print(f"{ok_icon}{ichi_icon}{vsa_icon}{div_icon}{dry_icon}{atr_warn}  "
                      f"{rec['Score']:.1f}  WR:{rec['Win%']}  {t1_disp}  {rec['_ichi'][:25]}")
                records.append(rec)
                time.sleep(self.cfg.delay_sec)
            except Exception as e:
                print(f"⚠ {e}")

            if i < len(watchlist) - 1:
                time.sleep(self.cfg.delay_sec)

        if not records:
            print("❌ Không có dữ liệu."); return pd.DataFrame()

        res = (pd.DataFrame(records)
               .sort_values(["_ok", "_score"], ascending=[False, False])
               .reset_index(drop=True))

        n_ok      = res["_ok"].sum()
        n_f       = res["Tín hiệu"].str.contains("🔥|💎").sum()
        n_g       = res["Tín hiệu"].str.contains("✅").sum()
        n_du      = (res["_dry_strength"] >= 2).sum()
        n_vsa     = res["_vsa"].str.contains("NỔ VOL|RÚT CHÂN").sum()
        n_bull_d  = res["_div"].str.contains("dương").sum()
        n_bear_d  = res["_div"].str.contains("âm").sum()
        n_3tf     = res["_ichi_3tf"].sum()
        n_above_d = res["_ichi"].str.contains("Trên Kumo|3TF").sum()
        # Đếm mã đạt T+ ≥ 9%
        n_tp9     = (res["_t1_pct"] >= self.cfg.min_tp_pct).sum()

        print(f"\n{'═'*72}")
        print(f"  {ctx['label']}")
        print(f"  Scan:{len(res)}  Đủ ĐK:{n_ok}  Mạnh:{n_f}  Xem:{n_g}")
        print(f"  Cạn cung:{n_du}  VSA+:{n_vsa}  Div dương:{n_bull_d}  Div âm:{n_bear_d}")
        print(f"  Ichimoku 3TF Bull:{n_3tf}  Trên Kumo:{n_above_d}  T+≥{self.cfg.min_tp_pct:.0f}%:{n_tp9}")
        print(f"{'═'*72}\n")

        _display(res.head(top_n) if not res.empty else res, mode)
        self._print_detail(res)
        return res

    def detail(self, symbol: str, mode: str = "swing"):
        """Phân tích chi tiết 1 mã — v5.4 với đầy đủ Ichimoku."""
        print(f"\n{'═'*65}")
        print(f"  PHÂN TÍCH CHI TIẾT: {symbol}  (VN Quant Engine v5.7)")
        print(f"{'═'*65}\n")
        ctx     = self.regime.detect()
        weights = self.regime.get_dynamic_weights()
        df_vni  = self.dl.fetch("VNINDEX", self.cfg.lookback_short)
        rec     = self.run_pipeline(symbol, df_vni, mode, weights, ctx["score_mult"],
                                    self.regime.get_min_score(mode, self.cfg),
                                    ctx.get("wr_mult", 1.0))
        if rec is None:
            print("Không lấy được dữ liệu."); return

        print(f"  Mã : {symbol}  |  Giá : {rec['Giá']}  |  Score : {rec['Score']:.1f}  |  Win% : {rec['Win%']}")
        print(f"  Tín hiệu : {rec['Tín hiệu']}  |  Khuyến nghị : {rec['Khuyến nghị']}")
        print(f"\n  ── KỸ THUẬT ────────────────────────────────────────────────")
        print(f"  VSA       : {rec['VSA']}")
        print(f"  Phân kỳ   : {rec['Phân kỳ']}")
        print(f"  Ichimoku  : {rec['Ichimoku']}")
        print(f"  PK RSI-CI : {rec['PK RSI (Composite)']}")
        print(f"  Dao găm   : {rec['Cảnh báo DG']}")
        print(f"  Theo dõi  : {rec['Theo dõi PK/DG']}")
        print(f"  Dòng tiền : {rec['Dòng tiền']}  |  Cạn cung : {rec['Cạn cung']}")
        print(f"  Trend     : {rec['Trend']}   |  VP       : {rec['VP']}")
        print(f"\n  ── ICHIMOKU LEVELS (S/R) ───────────────────────────────────")
        print(f"  Hỗ trợ : S1={rec['S1']}  S2={rec['S2']}")
        print(f"  Kháng cự: R1={rec['R1']}  R2={rec['R2']}")
        print(f"\n  ── GIAO DỊCH ───────────────────────────────────────────────")
        print(f"  Entry   : {rec['Entry']}  ({rec['Entry zone']})")
        print(f"  SL T+   : {rec['SL T+']}  |  SL Hold : {rec['SL Hold']}")
        print(f"  Action  : {rec['Action']}")
        print(f"\n  T+ (1-3 phiên) : T1={rec['T+ T1']} (+{rec['T+ %']})  T2={rec['T+ T2']}  R:R={rec['T+ R:R']}")
        print(f"  Hold (dài hạn) : T1={rec['Hold T1']} (+{rec['Hold %']})  T2={rec['Hold T2']}  R:R={rec['Hold R:R']}")
        if rec["_atr_low"]:
            print(f"\n  ⚡ CẢNH BÁO: ATR thấp — biên độ hẹp, cân nhắc skip T+")
        print(f"\n  ── POSITION SIZING ─────────────────────────────────────────")
        print(f"  Giá trị : {rec['Giá trị']}  |  % NAV : {rec['% NAV']}  |  Rủi ro : {rec['Rủi ro %']} NAV")
        print(f"\n  ── WIN RATE PATTERNS ───────────────────────────────────────")
        print(f"  {rec['_wr_patterns']}")
        if rec["_positives"]:
            print(f"\n  ── ĐIỂM MẠNH ───────────────────────────────────────────────")
            print(f"  {rec['_positives']}")
        if rec["Lý do"]:
            print(f"\n  ── LÝ DO BỊ LỌC ────────────────────────────────────────────")
            for r in rec["Lý do"]:
                print(f"  • {r}")
        print()

    def _print_detail(self, res: pd.DataFrame):
        """In chi tiết top mã đạt điều kiện."""
        top = res[res["Tín hiệu"].str.contains("🔥|💎|✅")].head(8)
        if top.empty: return
        print(f"\n  📋 Chi tiết top {len(top)} mã:")
        for _, r in top.iterrows():
            dry  = f"  {r['Cạn cung']}" if r["_dry_strength"] >= 2 else ""
            div  = f"  {r['Phân kỳ']}"  if r["Phân kỳ"] != "—" else ""
            ichi = f"  {r['_ichi']}"    if r["_ichi"] != "—"    else ""
            atr_w = "  ⚡ATR thấp" if r["_atr_low"] else ""
            print(f"\n  {r['Mã']:6s}  {r['Tín hiệu']}  {r['Khuyến nghị']}")
            print(f"         Score={r['Score']:.1f}  WR={r['Win%']}  VSA={r['VSA']}{dry}")
            print(f"         Ichimoku: {r['Ichimoku']}{ichi}")
            if div: print(f"         Div: {r['Phân kỳ']}")
            print(f"         {r['Dòng tiền']}")
            print(f"         S1={r['S1']}  S2={r['S2']}  R1={r['R1']}  R2={r['R2']}")
            print(f"         Entry : {r['Entry']} ({r['Entry zone']})  SL: {r['SL T+']}{atr_w}")
            print(f"         T+   → T1:{r['T+ T1']} (+{r['T+ %']})  T2:{r['T+ T2']}  R:R {r['T+ R:R']}")
            print(f"         Hold → T1:{r['Hold T1']} (+{r['Hold %']})  T2:{r['Hold T2']}  R:R {r['Hold R:R']}")
            print(f"         {r['Action']}  {r['Giá trị']}  {r['% NAV']} NAV  Rủi ro {r['Rủi ro %']}")

# ── App extensions: demo data, API key, progress scan ──────────────────────

def register_api_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return "empty"
    try:
        from vnstock import register_user
        register_user(key)
        return "ok"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def make_synthetic_ohlcv(symbol: str, n: int = 420) -> pd.DataFrame:
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    # Different "personalities" per ticker
    drift = 0.0004 + (seed % 7) * 0.00015
    vol = 0.018 + (seed % 5) * 0.004
    if symbol.upper() in ("VNINDEX", "VN30"):
        drift = 0.00055
        vol = 0.012
    rets = rng.normal(drift, vol, n)
    # occasional squeeze / breakout
    for i in range(40, n, 70):
        rets[i:i+8] *= 0.35
        if i + 12 < n:
            rets[i+8:i+12] += rng.choice([-1, 1]) * 0.025
    px = 20 + (seed % 80)
    close = px * np.exp(np.cumsum(rets))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.002, 0.028, n))
    low = close * (1 - rng.uniform(0.002, 0.028, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    base_vol = 80_000 + (seed % 20) * 25_000
    volume = rng.lognormal(np.log(base_vol), 0.55, n).astype(int)
    df = pd.DataFrame({
        "time": dates,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
    })
    return df


class AppDataLayer(DataLayer):
    """DataLayer with cache + demo fallback."""

    def __init__(self, cfg: QuantConfig, demo: bool = False, cache: dict | None = None):
        super().__init__(cfg)
        self.demo = demo
        self.cache = cache if cache is not None else {}

    def fetch(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        days = int(days)
        if self.demo:
            full_key = ("full", symbol)
            if full_key not in self.cache:
                self.cache[full_key] = make_synthetic_ohlcv(symbol, n=420)
            df = self.cache[full_key].tail(min(days, len(self.cache[full_key]))).reset_index(drop=True)
            return df.copy()
        for (sym, d), cached in list(self.cache.items()):
            if sym == symbol and isinstance(d, int) and d >= days and cached is not None:
                return cached.tail(days).reset_index(drop=True).copy()
        df = super().fetch(symbol, days)
        if df is None:
            return None
        self.cache[(symbol, days)] = df
        return df.copy()


def classify_strategies(rec: dict) -> dict:
    """Map one pipeline record onto 4 philosophies + consensus."""
    vsa = str(rec.get("VSA", "") or "")
    ichi = str(rec.get("Ichimoku", "") or "")
    score = float(rec.get("_score", rec.get("Score", 0)) or 0)
    ok = bool(rec.get("_ok", False))
    t1 = float(rec.get("_t1_pct", 0) or 0)
    div = str(rec.get("PK RSI (Composite)", rec.get("Phân kỳ", "")) or "")
    hold = str(rec.get("Khuyến nghị Hold", "") or "")
    buy_tag = str(rec.get("Khuyến nghị Mua Mới", "") or "")
    dg = str(rec.get("Cảnh báo DG", "") or "")
    trend = str(rec.get("Trend", "") or "")
    below_kumo = "Dưới Kumo" in ichi or "below" in ichi.lower()
    ichi_3tf = bool(rec.get("_ichi_3tf", False))
    above = "Trên Kumo" in ichi or "3TF" in ichi or ichi_3tf

    def bucket(buy, watch, exit_):
        if exit_:
            return "EXIT"
        if buy:
            return "BUY"
        if watch:
            return "WATCH"
        return "SKIP"

    # 1. Ichimoku / Claude
    ichi_exit = below_kumo or "CẮT" in hold or "BÁN HẾT" in hold
    ichi_buy = (not ichi_exit) and ok and (ichi_3tf or above) and t1 >= 7
    ichi_watch = (not ichi_exit) and (above or "Kijun" in str(rec.get("Entry zone", ""))) and score >= 4.5
    s_ichi = bucket(ichi_buy, ichi_watch, ichi_exit)

    # 2. VSA / Grok
    vsa_exit = "PHÂN PHỐI" in vsa or "CẠN CẦU" in vsa
    vsa_buy = (not vsa_exit) and (("NỔ VOL" in vsa) or ("RÚT CHÂN" in vsa) or ("CẠN VOL" in vsa)) and score >= 5
    vsa_watch = (not vsa_exit) and score >= 5.5
    s_vsa = bucket(vsa_buy, vsa_watch, vsa_exit)

    # 3. RSI Divergence + Dao Găm
    bull_div = any(x in div.lower() for x in ["dương", "bull", "hidden+"]) or "Mua" in buy_tag
    bear_div = any(x in div.lower() for x in ["âm", "bear", "hidden-"])
    dg_touch = "Chạm" in str(rec.get("Dao Găm D", "")) or "Chạm" in str(rec.get("Dao Găm W", "")) or "chạm" in dg.lower()
    s_dg = bucket(
        (bull_div or dg_touch) and not bear_div and score >= 4.5,
        (bull_div or dg_touch or "Theo dõi" in str(rec.get("Theo dõi PK/DG", ""))),
        bear_div and ("âm" in div.lower() or "CẮT" in hold),
    )

    # 4. Daily scanner BUY / WATCH / EXIT
    daily_exit = any(k in hold for k in ["CẮT LỖ", "BÁN HẾT", "GIẢM"]) or "Downtrend" in trend
    daily_buy = ok and score >= 6 and not daily_exit
    daily_watch = (score >= 5 or "✅" in str(rec.get("Tín hiệu", ""))) and not daily_exit
    s_daily = bucket(daily_buy, daily_watch, daily_exit)

    votes = {"BUY": 0, "WATCH": 0, "EXIT": 0, "SKIP": 0}
    for s in (s_ichi, s_vsa, s_dg, s_daily, rec.get("S_SMC", "SKIP")):
        votes[s if s in votes else "SKIP"] += 1

    # Consensus: EXIT if >=2 EXIT; BUY if >=2 BUY and EXIT < 2; else WATCH if any interest
    if votes["EXIT"] >= 2:
        consensus = "EXIT"
    elif votes["BUY"] >= 2 and votes["EXIT"] == 0:
        consensus = "BUY"
    elif votes["BUY"] >= 1 and votes["WATCH"] >= 1 and votes["EXIT"] == 0:
        consensus = "WATCH"
    elif votes["WATCH"] >= 2:
        consensus = "WATCH"
    elif votes["BUY"] == 1 and votes["EXIT"] == 0:
        consensus = "WATCH"
    else:
        consensus = "SKIP"

    return {
        "S_Ichimoku": s_ichi,
        "S_VSA": s_vsa,
        "S_RSI_DG": s_dg,
        "S_Daily": s_daily,
        "S_SMC": rec.get("S_SMC", "SKIP"),
        "votes_buy": votes["BUY"],
        "votes_watch": votes["WATCH"],
        "votes_exit": votes["EXIT"],
        "consensus": consensus,
    }


def scan_universe(
    engine: QuantEngine,
    symbols: list,
    mode: str = "swing",
    delay: float = 0.0,
    progress_cb: Optional[Callable] = None,
) -> tuple[pd.DataFrame, dict]:
    ctx = engine.regime.detect()
    weights = engine.regime.get_dynamic_weights()
    r_mult = ctx["score_mult"]
    wr_mult = ctx.get("wr_mult", 1.0)
    min_score = engine.regime.get_min_score(mode, engine.cfg)
    df_vni = engine.dl.fetch("VNINDEX", engine.cfg.lookback_short)
    records = []
    errors = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(i, n, sym)
        try:
            rec = engine.run_pipeline(sym, df_vni, mode, weights, r_mult, min_score, wr_mult)
            if rec is None:
                errors.append({"Mã": sym, "error": "thiếu dữ liệu"})
                continue
            ohlc = engine.dl.fetch(sym, engine.cfg.lookback_long)
            if ohlc is not None:
                try:
                    from vn_quant.smc import analyze_smc, plan_to_row
                    plan = analyze_smc(ohlc)
                    if plan is not None:
                        rec.update(plan_to_row(plan))
                except Exception as smc_e:
                    rec["S_SMC"] = "SKIP"
                    rec["SMC thesis"] = f"SMC lỗi: {type(smc_e).__name__}"
            rec.update(classify_strategies(rec))
            records.append(rec)
        except Exception as e:
            errors.append({"Mã": sym, "error": f"{type(e).__name__}: {e}"})
        if delay and i < n - 1:
            time.sleep(delay)
    if progress_cb:
        progress_cb(n, n, "done")
    df = pd.DataFrame(records)
    if not df.empty:
        order = {"BUY": 0, "WATCH": 1, "EXIT": 2, "SKIP": 3}
        df["_ord"] = df["consensus"].map(order).fillna(9)
        df = df.sort_values(["_ord", "votes_buy", "_score"], ascending=[True, False, False]).reset_index(drop=True)
    meta = {
        "regime": ctx,
        "weights": weights,
        "min_score": min_score,
        "errors": errors,
        "n_ok": int(df["_ok"].sum()) if not df.empty else 0,
        "n_rows": len(df),
    }
    return df, meta
