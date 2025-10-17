import cv2
from pathlib import Path

SN_TO_FOLDER = {
    "19274125": "cam0",
    "16471270": "cam1",
    "14135921": "cam2",
}

root = Path(".")
raw_dir = root / "raw"

for folder in SN_TO_FOLDER.values():
    (root / folder / "left").mkdir(parents=True, exist_ok=True)
    (root / folder / "right").mkdir(parents=True, exist_ok=True)

for img_path in raw_dir.glob("Explorer_*.png"):
    img_name = img_path.name
    sn_found = None

    for sn, folder in SN_TO_FOLDER.items():
        if sn in img_name:
            sn_found = sn
            break
    if not sn_found:
        print(f"Skip {img_name} (Wrong SN)")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read {img_path}")
        continue

    h, w = img.shape[:2]
    w_half = w // 2
    left = img[:, :w_half]
    right = img[:, w_half:]

    out_dir = root / SN_TO_FOLDER[sn_found]
    left_path = out_dir / "left" / f"{img_path.stem}_L.png"
    right_path = out_dir / "right" / f"{img_path.stem}_R.png"

    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)
    print(f"Saved {left_path.name}, {right_path.name} to {out_dir}")
