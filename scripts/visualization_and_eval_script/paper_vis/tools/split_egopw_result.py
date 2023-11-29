#  Copyright Jian Wang @ MPI-INF (c) 2023.
import json
import os
import pickle

def main():
    for id_name in ['jian1', 'jian2', 'diogo1', 'diogo2']:
        # id_name = 'jian1'
        print(id_name)
        if os.name == 'nt':
            base_path = fr'X:\ScanNet\work\egocentric_view\25082022\{id_name}'
        else:
            base_path = fr'/CT/EgoMocap/work/egocentric_view/25082022/{id_name}'

        with open(os.path.join(base_path, 'out', 'egopw_results.pkl'), 'rb') as f:
            egopw_pose_data = pickle.load(f)
        egopw_joint_dict = egopw_pose_data['estimated_local_skeleton']
        with open(os.path.join(base_path, 'out', 'egopw_results_estimated_pose.pkl'), 'wb') as f:
            pickle.dump(egopw_joint_dict, f)

if __name__ == '__main__':

    main()