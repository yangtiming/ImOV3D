<h2 align="center">
   <b>【NeurIPS 2024 🇨🇦】ImOV3D: Learning Open Vocabulary Point Clouds 3D Object Detection from Only 2D Images</b>
</h2>

- We are the first to accomplish Open-Vocabulary 3D Object Detection tasks without using any 3D ground truth data. 
- Thank you for 🌟 our ImOV3D.


[![ImOV3D Project on arXiv](https://img.shields.io/badge/ImOV3D_Project-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2410.24001v1)
[![ImOV3D Project Page](https://img.shields.io/badge/Project-ImOV3D_Page-blue?style=flat-square&logo=github)](https://yangtiming.github.io/ImOV3D_Page/)

> [Timing Yang*](https://yangtiming.github.io/), [Yuanliang Ju*](https://x.com/averyjuuu0213), [Li Yi](https://ericyi.github.io/) <br>
> Shanghai Qi Zhi Institute, IIIS Tsinghua University, Shanghai AI Lab<br>



## Overall Pipeline
 <p align="center"> <img src='img/pipe7.png' align="center" height="400px"> </p>

<!-- ## Main Results
 <p align="center"> <img src='img/mainresults.png' align="center" height="400px"> </p>

## More Ablation Study and Visualization

<p align="center"> <img src='img/abl_1.png' align="center" height="250px"> </p>
<p align="center"> <img src='img/abl_2_vis.png' align="center" height="400px"> </p> -->


## Environment Setup

To set up the project environment, follow this step:

**Create a virtual environment:**
```bash
conda env create -f environment.yml
```

After creating the virtual environment, activate it with:
```bash
conda activate ImOV3D
```

**PointNet++ Backbone Installation**
```bash
cd pointnet2
python setup.py install
cd ..
```



## Dataset Preparation



### Pretrain Stage

  For detailed guidance on setting up the dataset for the pretraining stage, see the [dataset instructions](./Data_Maker/).

### Adaptation
  See [Data Preparation](./Data_Maker/DATA_MAKER_DATASET_TRAINVAL_SUN_SCAN) for SUNRGBD or ScanNet.

### Format
    --[data_name]  # Root directory of the dataset
      ├── [data_name]_2d_bbox_train       # Training data with 2D bounding boxes
      ├── [data_name]_2d_bbox_val         # Validation data with 2D bounding boxes
      ├── [data_name]_pc_bbox_votes_train # Training data with point cloud bounding box votes
      ├── [data_name]_pc_bbox_votes_val   # Validation data with point cloud bounding box votes
      ├── [data_name]_trainval_train      # Training data (2D image + Calib)
      └── [data_name]_trainval_eval       # Evaluation data (2D image + Calib)

## Pretrain Weight

  | Module | Description | 
  |------------|-------------|
  | [PointCloudRender](./Data_Maker/PointCloudRender) | Finetuned ControlNet | 

| DataSet           | Description          | Logs                      |
|-------------------|----------------------|----------------------------|
| [LVIS](./Data_Maker)              | Pretrain Stage       | [SUNRGBD](./log_eval/log_eval_pretrain_sunrgbd.txt),[ScanNet](./log_eval/log_eval_pretrain_scannet.txt)        |
| [SUNRGBD](./Data_Maker/DATA_MAKER_DATASET_TRAINVAL_SUN_SCAN/)         | Adaptation Stage     | [SUNRGBD](./log_eval/log_eval_adapation_sunrgbd.txt)       |
| [ScanNet](./Data_Maker/DATA_MAKER_DATASET_TRAINVAL_SUN_SCAN)           | Adaptation Stage     | [ScanNet](./log_eval/log_eval_adapation_scannet.txt)       |



All weights and data are available on [🤗 Hugging Face](https://huggingface.co/TimingYang/ImOV3D) and [Baidu Cloud](https://pan.baidu.com/s/18v5VzVe3CtcUKwtiwqjEXg?pwd=0000).


## Training and Evaluation

**1️⃣ Pretrain**

Pretrain ImOV3D on the LVIS dataset:
```bash
bash ./scripts/train_lvis.sh
```

**2️⃣ Adapation**

For the SUNRGBD dataset:
```bash
bash ./scripts/train_sunrgbd.sh
```

For the ScanNet dataset:

```bash
bash ./scripts/train_scannet.sh
```

**3️⃣ Evaluation**

To measure the effectiveness of model, proceed to the evaluation phase.

```bash
bash ./scripts/eval.sh
```


## Contect
If you have any questions, please feel free to contact us:

Timing Yang: timingya@usc.edu
Yuanliang Ju: yuanliang.ju@mail.utoronto.ca


## Acknowledgement
Our code is based on [ImVoteNet](https://github.com/facebookresearch/imvotenet), [OV-3DET](https://github.com/lyhdet/OV-3DET), [Detic](https://github.com/facebookresearch/Detic), [ControlNet](https://github.com/lllyasviel/ControlNet), [ZoeDepth](https://github.com/isl-org/ZoeDepth), [surface_normal_uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty).


## Citation
```bibtex
@article{yang2024imov3d,
  title={ImOV3D: Learning Open Vocabulary Point Clouds 3D Object Detection from Only 2D Images},
  author={Yang, Timing and Ju, Yuanliang and Yi, Li},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={141261--141291},
  year={2024}
}
```

