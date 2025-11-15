import argparse
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Ruta de salida PNG")
    parser.add_argument("--w", type=int, default=400)
    parser.add_argument("--h", type=int, default=300)
    args = parser.parse_args()

    W, H = args.w, args.h
    img = Image.new("L", (W, H), 255)  # fondo blanco
    d = ImageDraw.Draw(img)

    # algunas líneas negras
    d.line((20, 20, W-20, 40), fill=0, width=2)            # casi horizontal
    d.line((30, H-30, W-30, 60), fill=0, width=2)          # diagonal
    d.line((W//2, 0, W//2, H), fill=0, width=2)            # vertical
    d.line((0, H//2, W, H//2), fill=0, width=2)            # horizontal

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"[OK] Imagen de prueba guardada en: {out}")

if __name__ == "__main__":
    main()
