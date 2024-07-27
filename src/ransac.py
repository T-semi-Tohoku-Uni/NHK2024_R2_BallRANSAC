import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import pyransac3d as pyrsc

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

pc = rs.pointcloud()

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    # 点群の取得
    pc.map_to(depth_frame)
    points = pc.calculate(depth_frame)

    # 点群をNumpy配列に変換し、2次元配列に形状変更
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    
    # RANSAC sphere fitting
    sphere1 = pyrsc.Sphere()
    print(sphere1.fit(vtx, 0.05, 250))




# Stop streaming
pipeline.stop()

