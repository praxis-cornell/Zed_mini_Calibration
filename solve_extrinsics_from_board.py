# solve_extrinsics_from_board.py
import json, argparse, numpy as np, cv2
from pathlib import Path
from pupil_apriltags import Detector
from collections import defaultdict

def hat(v):  # skew
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3:4]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3:4] = -R.T @ t
    return Ti

def avg_SE3(Ts):
    # rotation: quaternion average; translation: median
    def R_to_q(R):
        return cv2.Rodrigues(R)[0]  # actually rvec, simpler: use SVD-based quat? We'll use cv2.RQDecomp for robustness is overkill
    # use scipy? avoid. approximate: log-map mean
    # small set => do chordal mean on rotation matrices
    M = sum(T[:3,:3] for T in Ts) / len(Ts)
    U,_,Vt = np.linalg.svd(M)
    R = U@Vt
    t = np.median(np.stack([T[:3,3] for T in Ts],0), axis=0)
    Tm = np.eye(4); Tm[:3,:3]=R; Tm[:3,3]=t
    return Tm

def build_board_points(conf):
    tag = conf["tag_size_m"]; sp = conf["grid_spacing_m"]
    rows, cols = conf["grid_rows"], conf["grid_cols"]
    id_grid = conf["id_grid"]
    # Board坐标：以左上角tag左上角为(0,0,0)，X向右，Y向下，Z=0
    world_pts_by_id = {}
    for r in range(rows):
        for c in range(cols):
            tid = id_grid[r][c]
            x0 = c * sp
            y0 = r * sp
            # 每个tag的4个角(按检测器顺序 [lt, rt, rb, lb])
            corners = np.array([
                [x0,          y0,           0.0],
                [x0+tag,      y0,           0.0],
                [x0+tag,      y0+tag,       0.0],
                [x0,          y0+tag,       0.0],
            ], dtype=np.float32)
            world_pts_by_id[tid] = corners
    return world_pts_by_id

def pnp_from_multi_tags(img_gray, det, world_by_id, K, dist):
    tags = det.detect(img_gray, estimate_tag_pose=False)
    if len(tags)==0: return None, None, 1e9, 0
    obj_pts=[]; img_pts=[]
    used=0
    for t in tags:
        tid = int(t.tag_id)
        if tid in world_by_id:
            obj_pts.append(world_by_id[tid])
            img_pts.append(t.corners)   # 4x2
            used+=1
    if used<2:  # at least see 2
        return None, None, 1e9, used
    obj = np.concatenate(obj_pts, axis=0).astype(np.float32)
    img = np.concatenate(img_pts, axis=0).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None, None, 1e9, used
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.squeeze()-img, axis=1).mean()
    R = rodrigues_to_R(rvec)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.squeeze()
    return T, err, used, len(tags)

def load_intrinsics(conf, cam_key):
    c = conf["zed_intrinsics"][cam_key]
    K = np.array([[c["fx"],0,c["cx"]],[0,c["fy"],c["cy"]],[0,0,1]],dtype=np.float64)
    dist = np.zeros(5)  # 图像若已去畸变，可以用0；否则填真实畸变
    return K, dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--config", type=str, default="config_board.json")
    ap.add_argument("--cams", type=str, default="cam0,cam1,cam2")
    args = ap.parse_args()

    root = Path(args.root)
    conf = json.loads(Path(args.config).read_text())
    world_by_id = build_board_points(conf)
    det = Detector(families=conf["family"], nthreads=4, quad_decimate=1.0, refine_edges=True)

    cam_keys = args.cams.split(",")
    img_lists = {ck: sorted((root/ck).glob("img_*.png")) for ck in cam_keys}
    n = min(len(v) for v in img_lists.values())
    print(f"Found {n} synchronized groups")

    Ks = {}; dists={}
    for ck in cam_keys:
        Ks[ck], dists[ck] = load_intrinsics(conf, ck)

    poses = defaultdict(list)  # poses[ck][k] = T_cam_board
    stats = []

    for k in range(n):
        for ck in cam_keys:
            img = cv2.imread(str(img_lists[ck][k]), cv2.IMREAD_GRAYSCALE)
            T, err, used, total = pnp_from_multi_tags(img, det, world_by_id, Ks[ck], dists[ck])
            if T is None:
                print(f"[WARN] {ck} frame {k:03d}: not enough tags ({used}/{total}), skipped")
                break
            poses[ck].append((k,T,err,used,total))
        else:
            # only if all cams succeeded
            stat_line = f"k={k:03d}  " + "  ".join([f"{ck}: err={poses[ck][-1][2]:.3f}px used={poses[ck][-1][3]}" for ck in cam_keys])
            print(stat_line)
            stats.append(stat_line)
            continue
        # any fail -> drop this k for all
        for ck in cam_keys:
            if len(poses[ck]) and poses[ck][-1][0]==k:
                poses[ck].pop()

    # 计算 pairwise 外参（相对于 cam0）
    ref = cam_keys[0]
    Ts_10=[]; Ts_20=[]
    valid = min(len(poses[ref]), len(poses[cam_keys[1]]), len(poses[cam_keys[2]]))
    for i in range(valid):
        _, T0, e0, *_ = poses[ref][i]
        _, T1, e1, *_ = poses[cam_keys[1]][i]
        _, T2, e2, *_ = poses[cam_keys[2]][i]
        # T_camX_cam0 = T_camX_board * inv(T_cam0_board)
        T10 = T1 @ inv_T(T0)
        T20 = T2 @ inv_T(T0)
        Ts_10.append(T10); Ts_20.append(T20)

    T10_avg = avg_SE3(Ts_10)
    T20_avg = avg_SE3(Ts_20)

    np.save("extrinsics_cam1_to_cam0.npy", T10_avg)
    np.save("extrinsics_cam2_to_cam0.npy", T20_avg)

    def to_rt(T):
        R = T[:3,:3]; t=T[:3,3]
        rvec,_=cv2.Rodrigues(R)
        return rvec.squeeze(), t

    r10,t10 = to_rt(T10_avg)
    r20,t20 = to_rt(T20_avg)
    print("\n=== RESULT (Cam1 -> Cam0) ===")
    print("Rvec (rad):", r10)
    print("t (m):", t10)
    print("\n=== RESULT (Cam2 -> Cam0) ===")
    print("Rvec (rad):", r20)
    print("t (m):", t20)
    print(f"\nSaved: extrinsics_cam1_to_cam0.npy, extrinsics_cam2_to_cam0.npy")

if __name__ == "__main__":
    main()
