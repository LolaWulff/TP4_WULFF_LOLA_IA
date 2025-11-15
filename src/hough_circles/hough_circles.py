from typing import Tuple
import numpy as np

def hough_circles(edge_bin: np.ndarray, radius: int) -> np.ndarray:
    """
    Calcula el acumulador para la Transformada de Hough de circunferencias
    con radio fijo.

    edge_bin: imagen binaria de bordes (0/255)
    radius: radio del círculo buscado (en píxeles)
    """
    H, W = edge_bin.shape
    acc = np.zeros((H, W), dtype=np.uint32)

    ys, xs = np.nonzero(edge_bin)

    # Para cada píxel de borde, se votan los posibles centros (a,b)
    for (y, x) in zip(ys, xs):
        for theta in range(0, 360, 1):  # 1° de resolución angular
            a = int(round(x - radius * np.cos(np.deg2rad(theta))))
            b = int(round(y - radius * np.sin(np.deg2rad(theta))))
            if 0 <= a < W and 0 <= b < H:
                acc[b, a] += 1

    return acc

def find_peaks(acc: np.ndarray, k: int = 3, min_votes: int = 50):
    peaks = []
    acc_copy = acc.copy()
    for _ in range(k):
        idx = np.unravel_index(np.argmax(acc_copy), acc_copy.shape)
        votes = int(acc_copy[idx])
        if votes < min_votes:
            break
        peaks.append((idx[1], idx[0], votes))  # (a, b, votos)
        acc_copy[idx] = 0
    return peaks
