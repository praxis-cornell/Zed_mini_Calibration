import pickle, numpy as np, open3d as o3d

with open("calibrate.pkl", "rb") as f:
    c2ws = pickle.load(f)

meshes = []
colors = [(1,0,0),(0,1,0),(0,0,1)]
for i, T in enumerate(c2ws):
    cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    cam.transform(T)
    cam.paint_uniform_color(colors[i])
    meshes.append(cam)

tag = o3d.geometry.TriangleMesh.create_box(width=0.08, height=0.08, depth=0.001)
tag.paint_uniform_color((0.8,0.8,0.8))
tag.translate([-0.04, -0.04, 0])
meshes.append(tag)

o3d.visualization.draw_geometries(meshes)
