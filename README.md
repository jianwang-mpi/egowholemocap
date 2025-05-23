# Egocentric Whole-Body Motion Capture with FisheyeViT and Diffusion-Based Motion Refinement

[![](https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green)](https://arxiv.org/pdf/2311.16495.pdf)
[![](https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=githubpages&logoColor=blue)](https://people.mpi-inf.mpg.de/~jianwang/projects/egowholemocap/index.html)


This is the official implementation of the paper:

Wang, Jian et al. "Egocentric Whole-Body Motion Capture with FisheyeViT and Diffusion-Based Motion Refinement." CVPR. (2024).

[[Project Page]](https://people.mpi-inf.mpg.de/~jianwang/projects/egowholemocap/index.html)  <--- The dataset link in the project page might have expired.

[[EgoWholeBody Training Dataset]](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.SJYBX3) 

[[EgoWholeBody Test Dataset]](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.FSBR5V)

[[SceneEgo Hand Annotations]](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.VCIHDO)


## Installation

We base our code on the **0.x version** of [MMPose](https://github.com/open-mmlab/mmpose/tree/0.x).

1. First create the conda environment and activate it:

```shell
conda create -n egowholemocap python=3.9 -y
conda activate egowholemocap
```

2. Then install the pytorch version (tested on python 1.3.x) that matches your CUDA version. For example, if you have CUDA 11.7, you can install pytorch 1.13.1 with the following command:
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install this project:

```shell
pip install openmim
mim install mmcv-full==1.7.1
pip install -e .
```

4. Install the dependencies of this project:

```shell
pip3 install -r requirements.txt
```

5. If smplx installed open3d-python, you should uninstall it by running:

```shell
pip uninstall open3d-python
```

6. Change the torchgeometry code following [this issue](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).

7. Finally, download the [mano hand model](https://mano.is.tue.mpg.de/index.html), then put it under `./human_models/`.
The structure of `./human_models/` should be like this:

```shell
human_models
|-- mano
|-- |-- MANO_RIGHT.pkl
|-- |-- MANO_LEFT.pkl
```

## Run the demo

### 1. Download the pretrained models

1. Download the pretrained human body pose estimation model (FisheyeViT + pixel-aligned 3D heatmap) from [NextCloud](https://nextcloud.mpi-klsb.mpg.de/index.php/s/zmaFFAEBR33LFQt) and put it under `./checkpoints/`.
2. Download the pretrained hand detection model from [NextCloud](https://nextcloud.mpi-klsb.mpg.de/index.php/s/8zow6NEWKgPFnRF) and put it under `./checkpoints/`.
3. Download the pretrained hand pose estimation model from [NextCloud](https://nextcloud.mpi-klsb.mpg.de/index.php/s/343YTMdfgAneHcC) and put it under `./checkpoints/`.
4. Download the pretrained whole-body motion diffusion model from [NextCloud](https://nextcloud.mpi-klsb.mpg.de/index.php/s/ifgQeHBrfZMC5SN) and put it under `./checkpoints/`.

### 2. Prepare the data

The input data should be a image sequence in directory `./demo/resources/imgs/`.

For example, you can download the example sequence from [NextCloud](https://nextcloud.mpi-klsb.mpg.de/index.php/s/QNynZqQBCFppwcj), unzip the file and put it under `./demo/resources/`.

### 3. Run the single-frame whole-body pose estimation method

```shell
tools/python_test.sh configs/egofullbody/egowholebody_single_demo.py none
```
The result data will be saved in `./work_dirs/egowholebody_single_demo`.

#### (Optional) Visualize the result
Note: the headless server is not supported.

```shell
python scripts/visualization_script/vis_single_frame_whole_body_result.py \
        --pred_path work_dirs/egowholebody_single_demo/outputs.pkl \
        --image_id 0
```

### 4. Run the diffusion-based whole-body motion refinement method

```shell
python demo/demo_whole_body_diffusion.py \
 --pred_path work_dirs/egowholebody_single_demo/outputs.pkl
```
The result will be saved in `./work_dirs/egowholebody_diffusion_demo`.

#### (Optional) Visualize the result

Note: the headless server is not supported.

```shell
python scripts/visualization_script/vis_diffusion_whole_body_result.py \
        --pred_path work_dirs/egowholebody_diffusion_demo/outputs.pkl \
        --image_id 0
```


## Training

### Prepare the training data

1. Download the EgoWholeBody synthetic dataset from [NextCloud](https://nextcloud.mpi-klsb.mpg.de/index.php/s/oRHwkccnYcFiSMS).

2. Unzip all of the files. The file structure should be like this:

```shell
path_to_dataset_dir
 |-- renderpeople_adanna
 |-- renderpeople_amit
 |-- ......
 |-- renderpeople_mixamo_labels_old.pkl
 |-- ......
```

### Train the single-frame pose estimation model

1. Download the pre-trained ViT model from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/qJWCRc8EApcpP4r) and put it under `./pretrained_models/`.

2. Modify the config file `configs/egofullbody/fisheye_vit/undistort_vit_heatmap_3d.py`. Modify paths in line: 1, 19, 28, 29, 149, 150.

3. Modify the paths between line 22-35 in file `mmpose\datasets\datasets\egocentric\mocap_studio_dataset.py` to the paths of SceneEgo test dataset.

4. Train the model:

```shell
tools/python_train.sh configs/egofullbody/fisheye_vit/undistort_vit_heatmap_3d.py
```

### Finetune the single-frame pose estimation model on the SceneEgo training dataset

1. Modify the paths in config file `configs/egofullbody/fisheye_vit/undistort_vit_heatmap_3d_finetune_size_0.2_better_init.py`.
2. Modify the paths in file: `mmpose\datasets\datasets\egocentric\mocap_studio_finetune_dataset.py` to the SceneEgo training dataset.
3. Finetune the model:

```shell
tools/python_train.sh configs/egofullbody/fisheye_vit/undistort_vit_heatmap_3d_finetune_size_0.2_better_init.py
``` 

### Finetune the single-frame hand pose estimation model

1.  Download the hand4whole model from [here](https://drive.google.com/file/d/15wYR8psO2U3ZhFYQEH1-DWc81XkWvK2Y/view?usp=sharing) (in this [github repo](https://github.com/mks0601/Hand4Whole_RELEASE/tree/Pose2Pose)).
2.  To finetune the model on SceneEgo: 
    - Modify the paths in config file `configs/egofullbody/egohand/hands4whole_train.py`.
    - `tools/python_train.sh configs/egofullbody/egohand/hands4whole_train.py`
3. To finetune the model on EgoWholeBody:
    - Modify the paths in config file `configs/egofullbody/egohand/hands4whole_train_synthetic.py`.
    - `tools/python_train.sh configs/egofullbody/egohand/hands4whole_train_synthetic.py`

### Test the trained single-frame hand pose estimation model

- For testing on SceneEgo dataset, see `configs/egofullbody/egohand/hands4whole_test_finetuned.py`.
- For testing on synthetic EgoWholeBody dataset, see `configs/egofullbody/egofullbody_test_synthetic_fisheye.py`.


## How to modify

Since we are using mmpose as the training and evaluating framework,
please see [get_started.md](docs/en/get_started.md) for the basic usage of MMPose.
There are also tutorials:

- [learn about configs](docs/en/tutorials/0_config.md)
- [finetune model](docs/en/tutorials/1_finetune.md)
- [add new dataset](docs/en/tutorials/2_new_dataset.md)
- [customize data pipelines](docs/en/tutorials/3_data_pipeline.md)
- [add new modules](docs/en/tutorials/4_new_modules.md)
- [export a model to ONNX](docs/en/tutorials/5_export_model.md)
- [customize runtime settings](docs/en/tutorials/6_customize_runtime.md)

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{wang2024egocentric,
  title={Egocentric whole-body motion capture with fisheyevit and diffusion-based motion refinement},
  author={Wang, Jian and Cao, Zhe and Luvizon, Diogo and Liu, Lingjie and Sarkar, Kripasindhu and Tang, Danhang and Beeler, Thabo and Theobalt, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={777--787},
  year={2024}
}

```

## License

This project is released under the [Apache 2.0 license](LICENSE).
