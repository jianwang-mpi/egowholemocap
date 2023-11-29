#  Copyright Jian Wang @ MPI-INF (c) 2023.
from mmpose.utils.fisheye_camera.fisheye_to_sphere import Fisheye2Sphere

fisheye_camera_path = 'Z:/EgoMocap/work/EgocentricFullBody/mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'
fisheye2sphere = Fisheye2Sphere(fisheye_camera_path)

image_path = r'X:\ScanNet\work\egocentric_view\25082022\diogo1\imgs\img_002129.jpg'
fisheye2sphere.fisheye_to_sphere(image_path)