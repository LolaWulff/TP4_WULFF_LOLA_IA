from typing import Tuple
import numpy as np

def hough_space(edge_bin: np.ndarray,
                theta_res_deg: float = 1.0,
                rho_res: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Acumulador de Hough para rectas en forma polar:
      rho = x cos(theta) + y sin(theta)
    edge_bin: imagen binaria de bordes (0/255)
    Retorna: (acc, rhos, thetas)
    """
    H, W = edge_bin.shape
    rho_max = int(np.ceil(np.hypot(H, W)))
    rhos = np.arange(-rho_max, rho_max + 1, rho_res, dtype=np.float32)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_res_deg, dtype=np.float32))

    acc = np.zeros((len(rhos), len(thetas)), dtype=np.uint32)

    ys, xs = np.nonzero(edge_bin)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for (y, x) in zip(ys, xs):
        r = x * cos_t + y * sin_t
        idx = np.round((r - rhos[0]) / (rhos[1] - rhos[0])).astype(int)
        valid = (idx >= 0) & (idx < len(rhos))
        acc[idx[valid], np.where(valid)[0]] += 1

    return acc, rhos, thetas

def find_peaks(acc: np.ndarray, k: int = 5, min_votes: int = 50):
    """
    Extrae hasta k picos por máximo global simple.
    Devuelve lista de tuplas (rho_idx, theta_idx, votes)
    """
    peaks = []
    acc_copy = acc.copy()
    for _ in range(k):
        idx = np.unravel_index(np.argmax(acc_copy), acc_copy.shape)
        votes = int(acc_copy[idx])
        if votes < min_votes:
            break
        peaks.append((idx[0], idx[1], votes))
        acc_copy[idx] = 0
    return peaks
