import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from hough_circles.edges import load_grayscale, simple_sobel_edges, normalize_uint8
from hough_circles.hough_circles import hough_circles, find_peaks

def main():
    parser = argparse.ArgumentParser(description="TP4 IA - Hough Circunferencias (prototipo)")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada")
    parser.add_argument("--radius", type=int, required=True, help="Radio del círculo (px)")
    parser.add_argument("--edge-thresh", type=float, default=50.0, help="Umbral de bordes Sobel")
    parser.add_argument("--min-votes", type=int, default=50, help="Mínimo de votos para considerar un centro")
    parser.add_argument("--k", type=int, default=3, help="Cantidad máxima de circunferencias")
    args = parser.parse_args()

    in_path = Path(args.image)
    assert in_path.exists(), f"No existe la imagen: {in_path}"

    gray = load_grayscale(str(in_path))
    edges_bin, mag = simple_sobel_edges(gray, thresh=args.edge_thresh)

    print("[INFO] Calculando Hough de circunferencias...")
    acc = hough_circles(edges_bin, radius=args.radius)
    peaks = find_peaks(acc, k=args.k, min_votes=args.min_votes)

    out_dir = Path("data/output"); out_dir.mkdir(parents=True, exist_ok=True)
    acc_png = out_dir / f"{in_path.stem}_circles_acc.png"
    acc_norm = normalize_uint8(acc.astype(np.float32))

    plt.imshow(acc_norm, cmap="hot", origin="lower")
    plt.title("Acumulador de Hough (circunferencias)")
    plt.colorbar()
    plt.savefig(acc_png, dpi=150, bbox_inches="tight")
    plt.close()

    peaks_txt = out_dir / f"{in_path.stem}_circles_peaks.txt"
    with open(peaks_txt, "w", encoding="utf-8") as f:
        for (a, b, votes) in peaks:
            f.write(f"Centro: ({a}, {b}) - votos: {votes}\n")

    # Dibujar circunferencias detectadas
    img = Image.open(str(in_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (a, b, votes) in peaks:
        draw.ellipse(
            (a - args.radius, b - args.radius, a + args.radius, b + args.radius),
            outline="red", width=2
        )
    out_img = out_dir / f"{in_path.stem}_circles_detected.png"
    img.save(out_img)

    print(f"[OK] Circunferencias detectadas.")
    print(f"     Acumulador: {acc_png}")
    print(f"     Picos:      {peaks_txt}")
    print(f"     Imagen:     {out_img}")

if __name__ == "__main__":
    main()
