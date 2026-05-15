"""AI super-resolution using OpenCV's dnn_superres + EDSR x4 model.

EDSR (Enhanced Deep Super-Resolution) is a strong CNN-based upscaler. It
*adds* detail (unlike Lanczos which only smooths). Best on illustrations,
botanical line art, photographs.

Usage:
    python upscale_ai.py <input.png> [output.png]

Requires:
    pip install opencv-contrib-python
    models/EDSR_x4.pb (downloaded automatically by setup)
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / 'models' / 'EDSR_x4.pb'


def upscale_ai(src: str, dst: str | None = None) -> Path:
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError(src_p)

    # Load image (handle alpha channel)
    img = cv2.imread(str(src_p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Could not read {src_p}')
    h, w = img.shape[:2]
    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False

    print(f'Input:  {src_p.name}  {w}×{h}  alpha={has_alpha}')

    # Initialize EDSR x4
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL_PATH))
    sr.setModel('edsr', 4)

    # EDSR works on RGB; if RGBA, separate alpha and upscale separately
    t0 = time.time()
    if has_alpha:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        rgb_up = sr.upsample(rgb)
        # Upscale alpha with Lanczos (preserves anti-aliasing)
        alpha_up = cv2.resize(alpha, (rgb_up.shape[1], rgb_up.shape[0]),
                              interpolation=cv2.INTER_LANCZOS4)
        result = np.dstack((rgb_up, alpha_up))
    else:
        result = sr.upsample(img)

    elapsed = time.time() - t0
    new_h, new_w = result.shape[:2]
    print(f'Output: {new_w}×{new_h}  (AI upscaled in {elapsed:.1f}s)')

    # Save
    if dst is None:
        dst_p = src_p.with_stem(f'{src_p.stem}_AI4x')
    else:
        dst_p = Path(dst)
    cv2.imwrite(str(dst_p), result)
    print(f'Saved:  {dst_p}  ({dst_p.stat().st_size/1024:.0f} KB)')
    return dst_p


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python upscale_ai.py <input> [output]')
        sys.exit(1)
    upscale_ai(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
