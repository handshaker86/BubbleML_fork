import torch
import torch.nn.functional as F
import numpy as np
import numba as nb
import math
from dataclasses import dataclass


@dataclass
class Metrics:
    mae: float
    rmse: float
    relative_error: float
    max_error: float
    boundary_rmse: float
    interface_rmse: float
    fourier_low: float
    fourier_mid: float
    fourier_high: float
    rel_l2: float
    rel_vort: float
    div_error: float
    spectral_error: float

    def __str__(self):
        return f"""
            MAE: {self.mae:.6f}
            RMSE: {self.rmse:.6f}
            Relative Error (Global): {self.relative_error:.6f}
            Max Error: {self.max_error:.6f}
            Boundary RMSE: {self.boundary_rmse:.6f}
            Interface RMSE: {self.interface_rmse:.6f}
            Fourier
                - Low: {self.fourier_low:.6f}
                - Mid: {self.fourier_mid:.6f}
                - High: {self.fourier_high:.6f}
            Physics Metrics
                - Rel L2 (Global): {self.rel_l2:.6f}
                - Rel Vorticity (Global): {self.rel_vort:.6f}
                - Div Consistency RMSE: {self.div_error:.6f}
                - Spectral Error (LogDist): {self.spectral_error:.6f}
        """


def compute_metrics(pred, label, dfun, Lx=1.0, Ly=1.0):
    low, mid, high = fourier_error(pred, label, Lx, Ly)
    rel_l2_val = calc_rel_l2(pred, label)
    spec_err_val = calc_spectral_error(pred, label)
    rel_vort_val, div_err_val = calc_physics_metrics(pred, label, Lx, Ly)

    return Metrics(
        mae=mae(pred, label),
        rmse=rmse(pred, label),
        relative_error=relative_error(pred, label),
        max_error=max_error(pred, label),
        boundary_rmse=boundary_rmse(pred, label),
        interface_rmse=interface_rmse(pred, label, dfun),
        fourier_low=low,
        fourier_mid=mid,
        fourier_high=high,
        rel_l2=rel_l2_val,
        rel_vort=rel_vort_val,
        div_error=div_err_val,
        spectral_error=spec_err_val,
    )


def write_metrics(pred, label, iter, stage, writer):
    writer.add_scalar(f"{stage}/MAE", mae(pred, label), iter)
    writer.add_scalar(f"{stage}/RMSE", rmse(pred, label), iter)
    writer.add_scalar(f"{stage}/Rel_L2_Global", relative_error(pred, label), iter)
    writer.add_scalar(f"{stage}/MaxError", max_error(pred, label), iter)


def mae(pred, label):
    """Mean absolute error (L1)."""
    return F.l1_loss(pred, label).item()


def rmse(pred, label):
    """Global RMSE over all elements."""
    assert pred.size() == label.size()
    mse = torch.mean((pred - label) ** 2)
    return torch.sqrt(mse).item()


def relative_error(pred, label):
    """Global relative L2: ||pred - label|| / ||label||."""
    return calc_rel_l2(pred, label)


def max_error(pred, label):
    """L-inf error (max absolute difference)."""
    return torch.max(torch.abs(pred - label)).item()


def _extract_boundary(tensor):
    """Extract boundaries of tensor [..., h, w]."""
    left = tensor[..., :, 0]
    right = tensor[..., :, -1]
    top = tensor[..., 0, :]
    bottom = tensor[..., -1, :]
    return torch.cat(
        [left.flatten(), right.flatten(), top.flatten(), bottom.flatten()], dim=-1
    )


def boundary_rmse(pred, label):
    """RMSE at domain boundaries."""
    assert pred.size() == label.size()
    bpred = _extract_boundary(pred)
    blabel = _extract_boundary(label)
    return torch.sqrt(torch.mean((bpred - blabel) ** 2)).item()


def interface_rmse(pred, label, dfun):
    """RMSE at interface pixels (dfun zero-crossing)."""
    assert pred.size() == label.size()

    if dfun.dim() > pred.dim():
        dfun = dfun.squeeze()

    total_squared_error = 0.0
    total_pixels = 0

    has_channel = pred.dim() == 4
    dfun_np = dfun.detach().cpu().numpy()

    for i in range(pred.size(0)):
        curr_dfun = dfun_np[i]
        if curr_dfun.ndim == 3:
            curr_dfun = curr_dfun[0]

        mask = get_interface_mask(curr_dfun)
        mask_t = torch.from_numpy(mask).to(pred.device)

        diff_sq = (pred[i] - label[i]) ** 2

        if has_channel:
            valid_pixels = diff_sq[:, mask_t]
        else:
            valid_pixels = diff_sq[mask_t]

        total_squared_error += torch.sum(valid_pixels).item()
        total_pixels += valid_pixels.numel()

    if total_pixels == 0:
        return 0.0

    return math.sqrt(total_squared_error / total_pixels)


@nb.njit
def get_interface_mask(dgrid):
    """Interface pixels (dfun sign change across neighbors)."""
    interface = np.zeros(dgrid.shape).astype(np.bool_)
    [rows, cols] = dgrid.shape
    for i in range(rows):
        for j in range(cols):
            adj = (
                (i < rows - 1 and dgrid[i][j] * dgrid[i + 1, j] <= 0)
                or (i > 0 and dgrid[i][j] * dgrid[i - 1, j] <= 0)
                or (j < cols - 1 and dgrid[i][j] * dgrid[i, j + 1] <= 0)
                or (j > 0 and dgrid[i][j] * dgrid[i, j - 1] <= 0)
            )
            interface[i][j] = adj
    return interface


def fourier_error(pred, target, Lx, Ly):
    """PDEBench-style Fourier band errors (radial binning)."""
    ILOW = 4
    IHIGH = 12

    assert pred.size() == target.size()

    if pred.dim() == 3:
        spatial_dims = [1, 2]
        nx, ny = pred.size(1), pred.size(2)
    elif pred.dim() == 4:
        spatial_dims = [2, 3]
        nx, ny = pred.size(2), pred.size(3)
    else:
        raise ValueError("Input must be 3D or 4D")

    pred_F = torch.fft.fftn(pred, dim=spatial_dims)
    target_F = torch.fft.fftn(target, dim=spatial_dims)
    _err_F = torch.abs(pred_F - target_F) ** 2

    if pred.dim() == 4:
        _err_F = torch.mean(_err_F, dim=1)
    _err_F = torch.mean(_err_F, dim=0)

    err_F = torch.zeros(min(nx // 2, ny // 2), device=pred.device)
    count_F = torch.zeros(min(nx // 2, ny // 2), device=pred.device)

    for i in range(nx // 2):
        for j in range(ny // 2):
            it = math.floor(math.sqrt(i**2 + j**2))
            if it < min(nx // 2, ny // 2):
                err_F[it] += _err_F[i, j]
                count_F[it] += 1

    valid_mask = count_F > 0
    err_F[valid_mask] = err_F[valid_mask] / count_F[valid_mask]

    err_F = torch.sqrt(err_F) / (nx * ny) * Lx * Ly

    low_err = torch.mean(err_F[:ILOW]).item()
    mid_err = torch.mean(err_F[ILOW:IHIGH]).item()
    high_err = torch.mean(err_F[IHIGH:]).item()
    return low_err, mid_err, high_err


def calc_rel_l2(pred, label):
    """Global relative L2: ||pred - label||_F / ||label||_F."""
    epsilon = 1e-8
    diff_norm = torch.norm(pred - label)
    true_norm = torch.norm(label)
    return (diff_norm / (true_norm + epsilon)).item()


def calc_spectral_error(pred, label):
    """Log spectral distance (RMSE of log PSD difference)."""
    epsilon = 1e-8
    spatial_dims = (-2, -1)

    fft_pred = torch.fft.fft2(pred, dim=spatial_dims)
    fft_true = torch.fft.fft2(label, dim=spatial_dims)

    psd_pred = torch.abs(fft_pred)
    psd_true = torch.abs(fft_true)
    log_diff = torch.log(psd_true + epsilon) - torch.log(psd_pred + epsilon)
    mse_log = torch.mean(log_diff**2)

    return torch.sqrt(mse_log).item()


def calc_physics_metrics(pred, label, Lx, Ly):
    """Relative vorticity error and divergence consistency RMSE."""
    epsilon = 1e-8

    if pred.dim() == 4 and pred.size(1) >= 2:
        _, _, h, w = pred.shape
        dx = Lx / h
        dy = Ly / w

        u_pred, v_pred = pred[:, 0], pred[:, 1]
        u_true, v_true = label[:, 0], label[:, 1]

        du_dx_p = torch.gradient(u_pred, spacing=dx, dim=2)[0]
        du_dy_p = torch.gradient(u_pred, spacing=dy, dim=1)[0]
        dv_dx_p = torch.gradient(v_pred, spacing=dx, dim=2)[0]
        dv_dy_p = torch.gradient(v_pred, spacing=dy, dim=1)[0]

        du_dx_t = torch.gradient(u_true, spacing=dx, dim=2)[0]
        du_dy_t = torch.gradient(u_true, spacing=dy, dim=1)[0]
        dv_dx_t = torch.gradient(v_true, spacing=dx, dim=2)[0]
        dv_dy_t = torch.gradient(v_true, spacing=dy, dim=1)[0]

        div_pred = du_dx_p + dv_dy_p
        div_true = du_dx_t + dv_dy_t
        div_error = torch.sqrt(torch.mean((div_pred - div_true) ** 2)).item()

        vor_pred = dv_dx_p - du_dy_p
        vor_true = dv_dx_t - du_dy_t

        vor_diff_norm = torch.norm(vor_true - vor_pred)
        vor_true_norm = torch.norm(vor_true)
        rel_vor_error = (vor_diff_norm / (vor_true_norm + epsilon)).item()

        return rel_vor_error, div_error

    return 0.0, 0.0
