# ZED Multi-Camera Calibration (AprilTag 36h11)

This repository provides a complete pipeline for calibrating **multiple ZED Mini cameras** using **AprilTag 36h11 markers**.

---

## 1. Environment Setup

### Create Conda Environment
```bash
conda env create -f environment.yml
conda activate calib
```

## 2. Hardware & Preparation

1. Install ZED SDK (includes ZED Explorer)

2. Print AprilTag 36h11, Tag size: 80 mm. Stick flat on table; ensure all cameras can see it.

3. Capture Images. Open ZED Explorer. Each camera takes 6–10 images. Move all captured images into:
```bash
./raw/
```

## 3. Calibration
```bash
# Step 1. Split ZED Explorer stereo images
python split_zed_images.py

# Step 2. Run AprilTag-based calibration (generate calibrate.pkl & metadata.json)
python apriltag_calib_c2w.py
```

## 4. Verification
```bash
# Step 3. Validate calibration numerically
chmod +x calibrate_check.sh
./calibrate_check.sh
```

## 5. Visualization
```bash
# Step 4. 3D visualize camera poses & AprilTag board
python varify_calibration.py
```

## 6. Output
```pgsql
calibrate.pkl      # camera-to-world 4x4 extrinsic matrix
metadata.json      # intrinsics & config
```
