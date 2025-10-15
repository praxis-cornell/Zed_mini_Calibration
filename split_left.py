import cv2
from pathlib import Path

SN_TO_FOLDER = {
    "19274125": "cam0",
    "16471270": "cam1",
    "19372172": "cam2",
}

root = Path(".")
for folder in SN_TO_FOLDER.values():
    (root / folder).mkdir(exist_ok=True)

for img_path in root.glob("cam*/Explorer_*.png"):
    if "_left" in img_path.stem:
        continue

    img_name = img_path.name
    sn_found = None
    for sn, folder in SN_TO_FOLDER.items():
        if sn in img_name:
            sn_found = sn
            break
    if not sn_found:
        print(f"Skip {img_path.name} (cannot read this SN)")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"can read {img_path}")
        continue

    h, w = img.shape[:2]
    left = img[:, :w//2]
    out_folder = root / SN_TO_FOLDER[sn_found]
    out_path = out_folder / f"{img_path.stem}_left.png"

    cv2.imwrite(str(out_path), left)
    print(f"Saved {out_path}")

