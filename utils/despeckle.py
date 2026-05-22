"""
utils/despeckle.py
==================
SAR 图像斑点降噪函数，用于在线（Dataset.__getitem__ 内）去噪。

SAR 专用滤波器（乘性斑点噪声模型）：
  - "lee"           : Lee 滤波，经典边缘保持自适应滤波器
  - "enhanced_lee"  : Enhanced Lee 滤波，三分类（均匀/非均匀/点目标）
  - "frost"         : Frost 滤波，指数衰减核自适应滤波
  - "gamma_map"     : Gamma MAP 滤波，假设场景服从 Gamma 分布

通用备选（非 SAR 专用，保底方案）：
  - "tv"            : Total Variation（Chambolle）
  - "bilateral"     : 双边滤波
  - "median"        : 中值滤波

所有函数输入/输出均为 [0, 1] 范围的 float32 numpy 数组。
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════════

def _local_stats(image: np.ndarray, window_size: int):
    """
    用盒式卷积计算局部均值和方差。

    Returns:
        local_mean : (H, W) float32
        local_var  : (H, W) float32, 已做 >=0 截断
    """
    from scipy.ndimage import uniform_filter

    local_mean = uniform_filter(image.astype(np.float64), window_size, mode="nearest")
    local_sq = uniform_filter((image.astype(np.float64)) ** 2, window_size, mode="nearest")
    local_var = np.maximum(local_sq - local_mean ** 2, 0.0)
    return local_mean.astype(np.float32), local_var.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# SAR 专用斑点滤波器
# ══════════════════════════════════════════════════════════════════════

def despeckle_lee(image: np.ndarray, window_size: int = 7, sigma: float = 0.06) -> np.ndarray:
    """
    Lee 滤波 — 乘性斑点噪声的边缘保持自适应滤波器。

    原理：在局部窗口内用最小均方误差（MMSE）估计真实后向散射系数。
    平滑强度自适应于局部异质性：均匀区域强平滑，边缘附近保持细节。

    Args:
        image:       (H, W) float32，值域 [0, 1]
        window_size: 滤波窗口大小（奇数），越大越平滑但越慢
        sigma:       噪声标准差（[0,1] 域下的估值）。
                     对于 Sentinel-1 EW GRD（ENL≈4）：
                       sigma_dB ≈ 4.343 / sqrt(4) ≈ 2.17 dB
                       dB_range  = 35（-30 到 +5 dB）
                       sigma ≈ 2.17 / 35 ≈ 0.062
                     调大 → 更强平滑；调小 → 更保边。

    Returns:
        (H, W) float32，值域 [0, 1]
    """
    local_mean, local_var = _local_stats(image, window_size)
    var_noise = float(sigma) ** 2

    # Lee 权重：k ∈ [0, 1]，0=全平滑(均值)，1=保留原值
    # k = 1 - var_noise / var_local
    k = 1.0 - var_noise / np.maximum(local_var, 1e-8)
    k = np.clip(k, 0.0, 1.0)

    result = local_mean + k * (image.astype(np.float32) - local_mean)
    return np.clip(result, 0.0, 1.0)


def despeckle_enhanced_lee(
    image: np.ndarray,
    window_size: int = 7,
    sigma: float = 0.06,
) -> np.ndarray:
    """
    Enhanced Lee 滤波 — Lee 的改进版，按局部变异系数三分类处理。

    - 均匀区（CV <= sigma）         → 用局部均值替代
    - 非均匀区（sigma < CV <= 2*sigma）→ Lee 加权平均
    - 点目标/边缘（CV > 2*sigma）    → 保留原值

    Args:
        image:       (H, W) float32，值域 [0, 1]
        window_size: 滤波窗口大小
        sigma:       噪声标准差，含义同 Lee

    Returns:
        (H, W) float32，值域 [0, 1]
    """
    local_mean, local_var = _local_stats(image, window_size)
    sigma_val = float(sigma)
    var_noise = sigma_val ** 2

    # 变异系数 CV = std / mean
    ci = np.sqrt(np.maximum(local_var, 1e-8)) / np.maximum(local_mean, 1e-6)

    result = image.copy().astype(np.float32)

    # 均匀区：全平滑
    homogeneous = ci <= sigma_val
    result[homogeneous] = local_mean[homogeneous]

    # 非均匀区：Lee 加权
    heterogeneous = (ci > sigma_val) & (ci <= 2.0 * sigma_val)
    k = 1.0 - var_noise / np.maximum(local_var, 1e-8)
    k = np.clip(k, 0.0, 1.0)
    result[heterogeneous] = (
        local_mean[heterogeneous]
        + k[heterogeneous] * (image.astype(np.float32)[heterogeneous] - local_mean[heterogeneous])
    )

    # 点目标/边缘：保留原值（result 已拷贝）

    return np.clip(result, 0.0, 1.0)


def despeckle_frost(
    image: np.ndarray,
    window_size: int = 7,
    damping: float = 2.0,
    sigma: float = 0.06,
) -> np.ndarray:
    """
    Frost 滤波 — 指数衰减核的自适应滤波。

    与 Lee 的硬权重不同，Frost 用指数衰减核做加权平均，
    核的衰减速率由局部变异系数控制，过渡更平滑。

    实现上，将 CI² 离散化为 20 个 bin，每 bin 预计算一次卷积，
    避免逐像素重建核。

    Args:
        image:       (H, W) float32，值域 [0, 1]
        window_size: 滤波窗口大小
        damping:     阻尼系数（>0），越大平滑越强
        sigma:       噪声标准差，含义同 Lee

    Returns:
        (H, W) float32，值域 [0, 1]
    """
    from scipy.ndimage import correlate

    local_mean, local_var = _local_stats(image, window_size)

    # CI² = var / mean²（局部变异系数的平方）
    ci_sq = local_var / np.maximum(local_mean ** 2, 1e-8)
    ci_sq = np.clip(ci_sq, 0.0, 50.0)

    half = window_size // 2
    y, x = np.ogrid[-half : half + 1, -half : half + 1]
    dist = np.sqrt(x ** 2 + y ** 2)

    # 离散化 CI² → 20 个 bin
    n_bins = 20
    ci_max = max(float(ci_sq.max()), 0.01)
    ci_bins = np.linspace(0, ci_max, n_bins + 1)
    ci_idx = np.digitize(ci_sq.flat, ci_bins[:-1]).reshape(ci_sq.shape)

    result = np.zeros_like(image, dtype=np.float64)

    for b in range(1, n_bins + 1):
        mask = ci_idx == b
        if not np.any(mask):
            continue
        ci_b = (ci_bins[b - 1] + ci_bins[min(b, n_bins)]) / 2.0
        alpha = float(damping) * ci_b
        kernel = np.exp(-alpha * dist)
        kernel = kernel / kernel.sum()  # 归一化

        filtered = correlate(image.astype(np.float64), kernel, mode="reflect")
        result[mask] = filtered[mask]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def despeckle_gamma_map(
    image: np.ndarray,
    window_size: int = 7,
    n_looks: float = 4.0,
) -> np.ndarray:
    """
    Gamma MAP 滤波 — 假设后向散射和斑点均服从 Gamma 分布的最大后验估计。

    比 Lee 更符合 SAR 强度的物理统计模型（Gamma 分布），
    对 Sentinel-1 GRD 强度数据理论最优。

    Args:
        image:       (H, W) float32，值域 [0, 1]
        window_size: 滤波窗口大小
        n_looks:     等效视数（ENL），S1 EW GRD 典型值 ≈ 4–5。
                     越大表示先验越强，平滑越多。

    Returns:
        (H, W) float32，值域 [0, 1]
    """
    local_mean, local_var = _local_stats(image, window_size)

    L = float(n_looks)
    local_mean_f = local_mean.astype(np.float64)
    local_var_f = np.maximum(local_var.astype(np.float64), 1e-10)

    # 变异系数平方
    ci_sq = local_var_f / (local_mean_f ** 2 + 1e-10)

    # Gamma MAP 解（MMSE 估计量）
    # 推导自：scene ~ Gamma(α,θ)，speckle ~ Gamma(L,L)，观测 = scene × speckle
    # 在均匀/非均匀区域，MAP 估计量为：
    #   R = [(α - L - 1)*E[z] + sqrt((α-L-1)²*E[z]² + 4*α*L*E[z]*z)] / (2*α)
    # 其中 α = (L+1) / (CI² - 1/L) （仅当 CI² > 1/L 时有意义）

    # 区域判定
    ci_min_sq = 1.0 / L  # 纯斑点噪声对应的最小 CI²

    # 均匀区：CI² <= 1/L → 直接用局部均值
    homogeneous = ci_sq <= ci_min_sq

    # α 参数（仅非均匀区有效）
    alpha = np.zeros_like(ci_sq)
    valid = ~homogeneous
    alpha[valid] = (L + 1.0) / (ci_sq[valid] - ci_min_sq)
    alpha = np.maximum(alpha, 0.0)  # 防止负值

    result = image.astype(np.float64).copy()
    img_f = image.astype(np.float64)

    denom = 2.0 * alpha
    safe = denom > 1e-10
    discriminant = (alpha - L - 1.0) * local_mean_f
    sqrt_term = np.sqrt(
        np.maximum(discriminant ** 2 + 4.0 * alpha * L * local_mean_f * img_f, 0.0)
    )

    result[safe] = ((alpha[safe] - L - 1.0) * local_mean_f[safe] + sqrt_term[safe]) / denom[safe]
    result[homogeneous] = local_mean_f[homogeneous]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 通用滤波器（保底方案）
# ══════════════════════════════════════════════════════════════════════

def _rtv_texture_weights(
    fin: np.ndarray, sigma: float, sharpness: float
) -> tuple:
    """Compute RTV texture weights. Matches computeTextureWeights.m (Xu et al. 2012)."""
    from scipy.ndimage import gaussian_filter

    eps_s = sharpness
    eps = 0.001

    # Horizontal (right) and vertical (down) finite differences
    fx = np.diff(fin, axis=1)
    fx = np.pad(fx, ((0, 0), (0, 1)), mode="constant")
    fy = np.diff(fin, axis=0)
    fy = np.pad(fy, ((0, 1), (0, 0)), mode="constant")

    # Structure edge weight: inverse of gradient magnitude
    wto = np.maximum(np.sqrt(fx**2 + fy**2), eps_s) ** (-1)

    # Low-pass filtered image for texture detection
    # Kernel size matches MATLAB: bitor(round(5*sigma), 1), then truncate accordingly
    import math
    ksize = max(int(math.floor(5 * sigma + 0.5)) | 1, 1)
    truncate = ((ksize - 1) / 2) / sigma
    fbin = gaussian_filter(fin.astype(np.float64), sigma=sigma, truncate=truncate)

    gfx = np.diff(fbin, axis=1)
    gfx = np.pad(gfx, ((0, 0), (0, 1)), mode="constant")
    gfy = np.diff(fbin, axis=0)
    gfy = np.pad(gfy, ((0, 1), (0, 0)), mode="constant")

    wtbx = np.maximum(np.abs(gfx), eps) ** (-1)
    wtby = np.maximum(np.abs(gfy), eps) ** (-1)

    wx = wtbx * wto
    wy = wtby * wto

    # Zero boundaries to prevent wrap-around connections
    wx[:, -1] = 0.0
    wy[-1, :] = 0.0

    return wx, wy


def _rtv_solve(
    IN: np.ndarray, wx: np.ndarray, wy: np.ndarray, lam: float
) -> np.ndarray:
    """
    Solve the RTV linear system. Matches solveLinearEquation.m.

    Constructs the weighted graph Laplacian system (I - λ·Lap_w)·S = I
    and solves with conjugate gradient (tol=0.1, matching MATLAB's pcg).

    In row-major order (C layout):
      pixel k = i*c + j
      east  neighbor: k+1  (weight wx[i,j], zero at last column)
      south neighbor: k+c  (weight wy[i,j], zero at last row)
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg

    r, c = IN.shape
    k = r * c

    dx = (-lam * wx).ravel()   # east edge weights  (negative = off-diagonal entries)
    dy = (-lam * wy).ravel()   # south edge weights

    # Diagonal: 1 - (east_w + west_w + south_w + north_w)
    # east  contribution at k:   dx[k]
    # west  contribution at k:   dx[k-1]  (= 0 when k is first column, since wx[:,−1]=0)
    # south contribution at k:   dy[k]
    # north contribution at k:   dy[k-c]  (= 0 for first row)
    e = dx.copy()
    w_arr = np.zeros(k, dtype=np.float64)
    w_arr[1:] = dx[:-1]

    s_arr = dy.copy()
    n_arr = np.zeros(k, dtype=np.float64)
    n_arr[c:] = dy[:-c]

    D = 1.0 - (e + w_arr + s_arr + n_arr)

    # Symmetric sparse matrix with 5 diagonals
    # +1/-1: horizontal connections  |  +c/-c: vertical connections
    A = diags(
        [D, dx[:-1], dx[:-1], dy[:-c], dy[:-c]],
        [0, 1, -1, c, -c],
        shape=(k, k),
        format="csr",
        dtype=np.float64,
    )

    IN_flat = IN.ravel().astype(np.float64)
    try:
        OUT_flat, info = cg(A, IN_flat, rtol=0.1, maxiter=100)   # SciPy >= 1.12
    except TypeError:
        OUT_flat, info = cg(A, IN_flat, tol=0.1, maxiter=100)    # SciPy < 1.12
    if info != 0:
        OUT_flat = IN_flat   # fallback: return input if solver fails

    return OUT_flat.reshape(r, c)


def despeckle_tv(
    image: np.ndarray,
    weight: float = 0.01,
    sigma: float = 3.0,
    sharpness: float = 0.005,
    max_iter: int = 4,
) -> np.ndarray:
    """
    Relative Total Variation (RTV) structure extraction (Xu et al., 2012).

    CORRECTED: implements tsmooth.m, NOT Chambolle TV.
    Separates iceberg structure from speckle / sea-ice texture patterns.

    For maximum physical accuracy use rtv_smooth_db() in prepare_dataset.py
    (applies on linear amplitude before normalization).  This online wrapper
    accepts [0, 1] normalized input for convenience.

    Args:
        image:     (H, W) float32, values in [0, 1]
        weight:    lambda smoothness weight (0, 0.05], default 0.01
        sigma:     texture element max scale in pixels, (0, 6], default 3.0
        sharpness: epsilon_s edge sharpness, [1e-3, 0.03], default 0.005
        max_iter:  iterations (4 matches tsmooth.m default)

    Returns:
        (H, W) float32, values in [0, 1]
    """
    I = image.astype(np.float64)
    x = I.copy()
    sigma_iter = float(sigma)
    lam = weight / 2.0

    for _ in range(max_iter):
        wx, wy = _rtv_texture_weights(x, sigma_iter, float(sharpness))
        x = _rtv_solve(I, wx, wy, lam)
        sigma_iter = max(sigma_iter / 2.0, 0.5)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def rtv_smooth_db(
    img_db: np.ndarray,
    weight: float = 0.01,
    sigma: float = 3.0,
    sharpness: float = 0.005,
    max_iter: int = 4,
) -> np.ndarray:
    """
    Offline RTV smoothing on SAR data in dB domain.

    Exactly matches the MATLAB pipeline in writesliceres.m:
      dB  →  10^(x/20) [db2mag: linear amplitude]  →  RTV  →  20·log10 [dB]

    Call this from prepare_dataset.py BEFORE normalize_sar(), on the raw dB patch.
    NaN pixels (no-data regions) are excluded from smoothing and restored afterwards.

    Args:
        img_db:    (H, W) float32/float64, SAR σ0 in dB
        weight:    lambda, (0, 0.05], default 0.01
        sigma:     texture scale in pixels, (0, 6], default 3.0
        sharpness: epsilon_s, [1e-3, 0.03], default 0.005
        max_iter:  iterations, default 4

    Returns:
        (H, W) float32, smoothed σ0 in dB; NaN values preserved
    """
    nan_mask = np.isnan(img_db)

    # dB → linear amplitude  (db2mag: 10^(x/20), matching MATLAB)
    img_linear = np.where(
        nan_mask, 0.0, 10.0 ** (img_db.astype(np.float64) / 20.0)
    )

    # RTV on linear amplitude
    I = img_linear
    x = I.copy()
    sigma_iter = float(sigma)
    lam = weight / 2.0

    for _ in range(max_iter):
        wx, wy = _rtv_texture_weights(x, sigma_iter, float(sharpness))
        x = _rtv_solve(I, wx, wy, lam)
        sigma_iter = max(sigma_iter / 2.0, 0.5)

    x = np.maximum(x, 0.0)   # linear amplitude must be non-negative

    # Linear → dB
    smoothed_db = 20.0 * np.log10(np.maximum(x, 1e-10))
    smoothed_db[nan_mask] = np.nan

    return smoothed_db.astype(np.float32)


def despeckle_bilateral(
    image: np.ndarray,
    d: int = 5,
    sigma_color: float = 0.1,
    sigma_space: float = 5,
) -> np.ndarray:
    """双边滤波：保边平滑。"""
    import cv2

    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    filtered_u8 = cv2.bilateralFilter(img_u8, d, sigma_color * 255.0, sigma_space)
    return filtered_u8.astype(np.float32) / 255.0


def despeckle_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """中值滤波：最简单，速度最快。"""
    import cv2

    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    filtered_u8 = cv2.medianBlur(img_u8, kernel_size)
    return filtered_u8.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════

_ALL_METHODS: Dict[str, Callable] = {
    "lee":          despeckle_lee,
    "enhanced_lee": despeckle_enhanced_lee,
    "frost":        despeckle_frost,
    "gamma_map":    despeckle_gamma_map,
    "tv":           despeckle_tv,
    "bilateral":    despeckle_bilateral,
    "median":       despeckle_median,
}


def get_despeckle_fn(cfg) -> tuple:
    """
    根据配置返回 (降噪函数, 参数字典)。

    Returns:
        (fn, kwargs_dict) — 若 despeckle.enabled=false，返回 (None, {})
    """
    ds_cfg = cfg["dataset"] if isinstance(cfg, dict) else cfg.dataset
    dsp_cfg = ds_cfg.get("despeckle", {})
    if isinstance(dsp_cfg, dict):
        enabled = dsp_cfg.get("enabled", False)
    else:
        enabled = dsp_cfg.enabled

    if not enabled:
        return None, {}

    method = dsp_cfg.get("method", "lee") if isinstance(dsp_cfg, dict) else dsp_cfg.method

    params_cfg = dsp_cfg.get(method, {}) if isinstance(dsp_cfg, dict) else getattr(dsp_cfg, method, {})
    params = dict(params_cfg) if not isinstance(params_cfg, dict) else params_cfg

    fn = _ALL_METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"未知降噪方法: {method}，可选: {', '.join(sorted(_ALL_METHODS))}"
        )
    return fn, params
