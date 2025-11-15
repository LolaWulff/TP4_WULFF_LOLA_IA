import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from hough_lines.edges import load_grayscale, simple_sobel_edges, normalize_uint8
from hough_lines.hough_lines import hough_space, find_peaks, line_endpoints_from_rho_theta

def main():
    parser = argparse.ArgumentParser(description="TP4 IA - Hough Rectas (prototipo)")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada")
    parser.add_argument("--edge-thresh", type=float, default=50.0, help="Umbral Sobel")
    parser.add_argument("--theta-res", type=float, default=1.0, help="Resolución angular (grados)")
    parser.add_argument("--rho-res", type=float, default=1.0, help="Resolución ρ (pixeles)")
    parser.add_argument("--k", type=int, default=5, help="Máx. picos a extraer")
    parser.add_argument("--min-votes", type=int, default=50, help="Mín. votos por pico")
    args = parser.parse_args()

    in_path = Path(args.image)
    assert in_path.exists(), f"No existe la imagen: {in_path}"

    # 1) Cargar y bordes
    gray = load_grayscale(str(in_path))
    edges_bin, mag = simple_sobel_edges(gray, thresh=args.edge_thresh)

    # 2) Hough
    acc, rhos, thetas = hough_space(edges_bin, theta_res_deg=args.theta_res, rho_res=args.rho_res)
    peaks = find_peaks(acc, k=args.k, min_votes=args.min_votes)

    out_dir = Path("data/output"); out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Guardar acumulador
    acc_norm = normalize_uint8(acc.astype(np.float32))
    plt.figure()
    plt.imshow(acc_norm, aspect="auto", origin="lower",
               extent=(np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[0], rhos[-1]))
    plt.xlabel("θ (grados)")
    plt.ylabel("ρ (pixeles)")
    plt.title("Acumulador de Hough (rectas)")
    plt.colorbar()
    acc_png = out_dir / f"{in_path.stem}_acc.png"
    plt.savefig(acc_png, dpi=150, bbox_inches="tight")
    plt.close()

    # 4) Guardar picos
    peaks_txt = out_dir / f"{in_path.stem}_peaks.txt"
    with open(peaks_txt, "w", encoding="utf-8") as f:
        for (ri, ti, votes) in peaks:
            f.write(f"rho={rhos[ri]:.1f}, theta_deg={np.rad2deg(thetas[ti]):.1f}, votes={votes}\n")

    # 5) Dibujar las rectas detectadas sobre la imagen original
    base = Image.open(str(in_path)).convert("RGB")
    draw = ImageDraw.Draw(base)
    H, W = gray.shape

    for (ri, ti, votes) in peaks:
        rho = float(rhos[ri])
        theta = float(thetas[ti])
        seg = line_endpoints_from_rho_theta(rho, theta, width=W, height=H)
        if seg is not None:
            (x1, y1), (x2, y2) = seg
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)

    detected_png = out_dir / f"{in_path.stem}_lines_detected.png"
    base.save(detected_png)

    print(f"[OK] Hough ejecutado.")
    print(f"     Acumulador: {acc_png}")
    print(f"     Picos:      {peaks_txt}")
    print(f"     Imagen:     {detected_png}")

if __name__ == "__main__":
    main()
