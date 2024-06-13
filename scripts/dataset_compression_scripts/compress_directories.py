import os
import subprocess

base_path = '/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo'
directory_list = [
    'render_people_claudia',
    'render_people_eric',
    'render_people_janna',
    'render_people_joko',
    'render_people_joyce',
    'render_people_kyle',
    'render_people_maya',
    'render_people_rin',
    'render_people_scott',
    'render_people_serena',
    'render_people_shawn'
]

for directory in directory_list:
    output_zip_file = f'{base_path}/{directory}.zip'
    if os.path.exists(output_zip_file):
        # skip if the zip file already exists
        print('Skip:', output_zip_file)
        continue
    print('Compress:', output_zip_file)
    # compress the directories into zip file
    subprocess.run(['zip', '-0', '-r', f'{base_path}/{directory}.zip', f'{base_path}/{directory}'])