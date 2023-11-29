#  Copyright Jian Wang @ MPI-INF (c) 2023.

import os
from natsort import natsorted

def main():
    image_dir = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results\47\imgs'
    video_save_path = r'Z:\EgoMocap\work\EgocentricFullBody\vis_results\diffusion_results\47\video.mp4'
    # rename the files in image_dir
    for i, file in enumerate(natsorted(os.listdir(image_dir))):
        os.rename(os.path.join(image_dir, file), os.path.join(image_dir, '%06d.jpg' % i))

    os.system(f'ffmpeg -r 20 -f image2 -i {image_dir}/%06d.jpg -vcodec libx264 {video_save_path}')

if __name__ == '__main__':
    main()
