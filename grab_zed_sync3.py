import pyzed.sl as sl
import cv2, time
from pathlib import Path

# === init cam ===
CAMERAS = [
    (19274125, "cam0"),
    (16471270, "cam1"),
    (19372172, "cam2"),
]

# === open cam ===
def open_camera(serial):
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080  # or HD720/HD2K
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.set_from_serial_number(serial)

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED SN={serial}")
    
    # auto
    cam.set_camera_settings(sl.VIDEO_SETTINGS.AUTO_EXPOSURE_GAIN, 1)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
    print(f"Opened ZED SN={serial}")
    return cam

def main():
    cams = [open_camera(sn) for sn, _ in CAMERAS]
    mats = [sl.Mat() for _ in CAMERAS]
    outs = [Path(folder) for _, folder in CAMERAS]
    for o in outs: o.mkdir(exist_ok=True)

    runtime = sl.RuntimeParameters()
    runtime.enable_depth = False
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    print("Cameras ready. Waiting 2s for exposure to stabilize...")
    time.sleep(2.0)

    print("Press <Enter> to capture all cameras, 'q' + <Enter> to quit.")
    idx = 0
    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == "q":
            break

        timestamp = int(time.time() * 1000)
        for i, (cam, mat, out) in enumerate(zip(cams, mats, outs)):
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.SIDE_BY_SIDE)
                frame = mat.get_data()
                fname = out / f"pair_{idx:03d}_{timestamp}.png"
                cv2.imwrite(str(fname), frame)
                print(f"📸 Saved {fname}")
        idx += 1

    for c in cams:
        c.close()
    print("✅ All cameras closed.")

if __name__ == "__main__":
    main()
