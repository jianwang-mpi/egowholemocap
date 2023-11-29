#  Copyright Jian Wang @ MPI-INF (c) 2023.
import cv2

import os
import numpy as np
from natsort import natsorted
from tqdm import tqdm


def img_path_list_to_video(img_path_list, video_path, fps=25):
    img = cv2.imread(img_path_list[0])
    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, size)
    print('Writing video to: ', video_path)
    for img_path in tqdm(img_path_list):
        img = cv2.imread(img_path)
        out.write(img)
    out.release()

def imgs_to_video(img_list, video_path, fps=25):
    img = img_list[0]
    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, size)
    print('Writing video to: ', video_path)
    for img in tqdm(img_list):
        out.write(img)
    out.release()


if __name__ == '__main__':
    id_name = 'jian2'
    image_dir = fr'Z:\EgoMocap\work\EgocentricFullBody\vis_results\hands_dataset\{id_name}\joint_2d'
    image_list = natsorted(os.listdir(image_dir))
    image_path_list = [os.path.join(image_dir, image_name) for image_name in image_list]
    video_path = fr'Z:\EgoMocap\work\EgocentricFullBody\vis_results\hands_dataset\{id_name}\{id_name}_joint_2d.mp4'
    img_path_list_to_video(image_path_list, video_path)