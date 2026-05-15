"""Vectorize raster image (PNG/JPG) to SVG using VTracer.

VTracer is a Rust-based color vectorizer that excels at:
  - Botanical / scientific illustrations
  - Logos and line art
  - Posters with limited color palettes

Output is a clean editable SVG ready for Adobe Illustrator, Inkscape,
PowerPoint vector import, and infinite scaling without quality loss.

Usage:
    python vectorize_image.py <input.png> [output.svg]
    python vectorize_image.py wheat_lifecycle_AI4x.png

Tuning options (edit below):
    color_precision   — # bits per color (lower = simpler, fewer colors)
    layer_difference  — color similarity threshold for merging
    filter_speckle    — remove tiny isolated regions (noise reduction)
    corner_threshold  — sharper corners vs smoother curves
    splice_threshold  — line continuity (lower = more separate paths)
"""
import sys
import time
from pathlib import Path
import vtracer


def vectorize(src: str, dst: str | None = None, mode: str = 'illustration') -> Path:
    """
    mode = 'illustration' (default — best for botanical/poster art)
         | 'photograph'   (for photos)
         | 'lineart'      (for B&W line art only)
    """
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError(src_p)
    dst_p = Path(dst) if dst else src_p.with_suffix('.svg')

    # Tuned for botanical illustrations with limited color palette
    presets = {
        'illustration': {
            'colormode':         'color',
            'hierarchical':      'stacked',
            'mode':              'spline',
            'filter_speckle':    8,        # remove specks smaller than 8 px²
            'color_precision':   6,        # 6 bits per channel = 64 colors
            'layer_difference':  16,       # merge similar colors
            'corner_threshold':  60,       # higher = smoother curves
            'length_threshold':  4.0,
            'max_iterations':    10,
            'splice_threshold':  45,
            'path_precision':    3,        # 3 decimal places
        },
        'photograph': {
            'colormode':         'color',
            'mode':              'spline',
            'filter_speckle':    4,
            'color_precision':   8,
            'layer_difference':  10,
        },
        'lineart': {
            'colormode':         'binary',
            'mode':              'spline',
            'filter_speckle':    4,
            'corner_threshold':  60,
        },
    }
    opts = presets.get(mode, presets['illustration'])

    print(f'Input:    {src_p.name}')
    print(f'Mode:     {mode}')
    print(f'Settings: color_precision={opts.get("color_precision","")}, '
          f'filter_speckle={opts.get("filter_speckle","")}')

    t0 = time.time()
    vtracer.convert_image_to_svg_py(str(src_p), str(dst_p), **opts)
    elapsed = time.time() - t0

    size_kb = dst_p.stat().st_size / 1024
    print(f'Output:   {dst_p.name}  ({size_kb:.0f} KB, {elapsed:.1f}s)')
    return dst_p


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python vectorize_image.py <input> [output.svg] [mode]')
        sys.exit(1)
    src  = sys.argv[1]
    dst  = sys.argv[2] if len(sys.argv) > 2 else None
    mode = sys.argv[3] if len(sys.argv) > 3 else 'illustration'
    vectorize(src, dst, mode)
