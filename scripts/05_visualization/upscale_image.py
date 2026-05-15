"""Upscale an image using PIL LANCZOS resampling.

For line-art / botanical illustrations on print, LANCZOS provides excellent
quality without artifacts. Not a true AI super-resolution (won't add detail
that wasn't there), but produces clean smooth enlargements suitable for poster
printing at 300 DPI.

Usage:
    python upscale_image.py <input.png> [scale=4] [output_path]

Example:
    python upscale_image.py ~/Downloads/wheat_strip.png 4
    # → ~/Downloads/wheat_strip_4x.png
"""
import sys
from pathlib import Path
from PIL import Image

def upscale(src: str, scale: int = 4, dst: str | None = None) -> Path:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    img = Image.open(src_path)
    print(f'Input:  {src_path.name}  {img.size}  mode={img.mode}')
    new_size = (img.width * scale, img.height * scale)
    upscaled = img.resize(new_size, Image.LANCZOS)
    if dst is None:
        dst_path = src_path.with_stem(f'{src_path.stem}_{scale}x')
    else:
        dst_path = Path(dst)
    upscaled.save(dst_path, optimize=True)
    print(f'Output: {dst_path.name}  {upscaled.size}  ({dst_path.stat().st_size/1024:.0f} KB)')
    return dst_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python upscale_image.py <input> [scale=4] [output]')
        sys.exit(1)
    src   = sys.argv[1]
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dst   = sys.argv[3] if len(sys.argv) > 3 else None
    upscale(src, scale, dst)
