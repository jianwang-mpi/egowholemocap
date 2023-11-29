import pickle
from tqdm import tqdm
import os
import cv2
from mmpose.utils.visualization.draw import draw_keypoints


# result_path = r'Z:\EgoMocap\work\EgocentricFullBody\work_dirs\hrnet_256x256_3d_train_head_1\results.pkl'
result_path = r'/CT/EgoMocap/work/EgocentricFullBody/work_dirs/hrnet_256x256_full_body_2d_test/results.pkl'

out_dir = '/CT/EgoMocap/work/EgocentricFullBody/vis_results/hrnet_256x256_full_body_2d_test'
os.makedirs(out_dir, exist_ok=True)

with open(result_path, 'rb') as f:
    result_data = pickle.load(f)

image_path_list = result_data['image_file']
joint_2d_list = result_data['joints_2d_pred']

for i in tqdm(range(0, len(image_path_list), 100)):
    image_path = image_path_list[i]
    image_name = os.path.split(image_path)[1]
    img = cv2.imread(image_path)


    joint_i = joint_2d_list[i] * 1024 / 64
    joint_i[:, 0] += 128

    print(joint_i)

    img = draw_keypoints(joint_i, img)

    out_path = os.path.join(out_dir, image_name)

    cv2.imwrite(out_path, img)

