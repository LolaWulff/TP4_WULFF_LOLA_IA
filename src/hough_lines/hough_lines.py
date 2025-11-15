from typing import Tuple, Optional
import numpy as np
import math

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

def line_endpoints_from_rho_theta(rho: float, theta_rad: float, width: int, height: int
                                  ) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    A partir de (rho, theta) devuelve dos puntos (x1,y1)-(x2,y2) dentro de la imagen
    para poder dibujar la recta. Intersecta la recta con los bordes del cuadro [0,W) x [0,H).
    Retorna None si no encuentra dos intersecciones válidas.
    """

    def intersect_with_border():
        pts = []

        # Bordes: x=0, x=W-1, y=0, y=H-1
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

        # Evitar divisiones por cero con tolerancia
        eps = 1e-9

        # Intersección con x = 0 => rho = y*sin + 0*cos
        if abs(sin_t) > eps:
            y = rho / sin_t
            if 0 <= y < height:
                pts.append((0, int(round(y))))

        # Intersección con x = W-1
        if abs(sin_t) > eps:
            y = (rho - (width - 1) * cos_t) / sin_t
            if 0 <= y < height:
                pts.append((width - 1, int(round(y))))

        # Intersección con y = 0 => rho = 0*sin + x*cos
        if abs(cos_t) > eps:
            x = rho / cos_t
            if 0 <= x < width:
                pts.append((int(round(x)), 0))

        # Intersección con y = H-1
        if abs(cos_t) > eps:
            x = (rho - (height - 1) * sin_t) / cos_t
            if 0 <= x < width:
                pts.append((int(round(x)), height - 1))

        # Quitar duplicados y quedarnos con dos
        uniq = []
        for p in pts:
            if p not in uniq:
                uniq.append(p)
        return uniq[:2] if len(uniq) >= 2 else None

    return intersect_with_border()