import argparse
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser(description="Genera una imagen sintética con círculos")
    parser.add_argument("--out", required=True, help="Ruta de salida (PNG)")
    parser.add_argument("--w", type=int, default=500, help="Ancho (px)")
    parser.add_argument("--h", type=int, default=400, help="Alto (px)")
    parser.add_argument("--bg", type=int, default=255, help="Fondo (0-255)")
    parser.add_argument("--thickness", type=int, default=3, help="Grosor del trazo del círculo")
    # Tres círculos por defecto; podés cambiarlos con flags
    parser.add_argument("--c1", type=str, default="150,120,40", help="cx,cy,r para círculo 1")
    parser.add_argument("--c2", type=str, default="330,200,60", help="cx,cy,r para círculo 2")
    parser.add_argument("--c3", type=str, default="200,300,25", help="cx,cy,r para círculo 3")
    args = parser.parse_args()

    W, H = args.w, args.h
    img = Image.new("L", (W, H), args.bg)  # fondo gris/blanco
    d = ImageDraw.Draw(img)

    def parse_c(cstr):
        cx, cy, r = map(int, cstr.split(","))
        return cx, cy, r

    circles = [parse_c(args.c1), parse_c(args.c2), parse_c(args.c3)]

    for (cx, cy, r) in circles:
        bbox = (cx - r, cy - r, cx + r, cy + r)
        d.ellipse(bbox, outline=0, width=args.thickness)  # negro

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"[OK] Imagen con círculos guardada en: {out}")

if __name__ == "__main__":
    main()
