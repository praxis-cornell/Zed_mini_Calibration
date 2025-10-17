import os
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from pupil_apriltags import Detector
import pyzed.sl as sl

# -------------------------------
# ① cam info
# -------------------------------
SERIALS = ["19274125", "16471270", "14135921"]   # SN
CAM_DIRS = ["cam0/left", "cam1/left", "cam2/left"]  # image folder
WH = [1920, 1080]
FPS = 30
TAG_SIZE_M = 0.08  # apriltag size 80 mm = 0.08 m

# -------------------------------
# ② read intrensic from ZED SDK
# -------------------------------
def get_intrinsics_from_sn(serial):
    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(int(serial))
    init.camera_disable_self_calib = False
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NONE
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Cannot open camera {serial}")

    calib = cam.get_camera_information().camera_configuration.calibration_parameters
    cam.close()

    K = [[calib.left_cam.fx, 0, calib.left_cam.cx],
         [0, calib.left_cam.fy, calib.left_cam.cy],
         [0, 0, 1]]
    print(f"SN {serial} intrensic: fx={calib.left_cam.fx:.1f}, fy={calib.left_cam.fy:.1f}, cx={calib.left_cam.cx:.1f}, cy={calib.left_cam.cy:.1f}")
    return K


# -------------------------------
# ③ get c2w（camera→world）from image
# -------------------------------
def solve_c2w_from_tag(img_bgr, K, detector, tag_size_m):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dets = detector.detect(gray,
                           estimate_tag_pose=True,
                           camera_params=[K[0][0], K[1][1], K[0][2], K[1][2]],
                           tag_size=tag_size_m)
    if len(dets) == 0:
        return None
    d = dets[0]
    R_tag2cam = d.pose_R
    t_tag2cam = d.pose_t.reshape(3, 1)
    R_cam2tag = R_tag2cam.T
    t_cam2tag = -R_tag2cam.T @ t_tag2cam

    T = np.eye(4)
    T[:3, :3] = R_cam2tag
    T[:3, 3] = t_cam2tag.squeeze()
    return T


# -------------------------------
# ④ main
# -------------------------------
def main():
    detector = Detector(families="tag36h11")
    Ks = []
    c2ws = []

    for serial, cam_dir in zip(SERIALS, CAM_DIRS):
        # get intrinsics
        K = get_intrinsics_from_sn(serial)
        Ks.append(K)
        # get extrensics
        imgs = sorted(list(Path(cam_dir).glob("*.png")))
        if not imgs:
            raise RuntimeError(f"{cam_dir} No Images!")

        poses = []
        for path in imgs[:10]:
            img = cv2.imread(str(path))
            T = solve_c2w_from_tag(img, K, detector, TAG_SIZE_M)
            if T is not None:
                poses.append(T)
        if not poses:
            raise RuntimeError(f"Camera {serial} not seen AprilTag")
        T_avg = np.mean(np.stack(poses), axis=0)
        c2ws.append(T_avg.astype(np.float32))
        print(f"Camera {serial}: Mean c2w Transform = {T_avg[:3,3]}")

    # Write calibrate.pkl
    with open("calibrate.pkl", "wb") as f:
        pickle.dump(c2ws, f)
    print("Saved calibrate.pkl")

    # Write metadata.json
    meta = {
        "intrinsics": Ks,
        "serial_numbers": SERIALS,
        "fps": FPS,
        "WH": WH,
        "frame_num": 0,
        "start_step": 0,
        "end_step": 0
    }
    with open("metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata.json")

    print("All Done!")


if __name__ == "__main__":
    main()
