

# Learning with Knowledge Distillation for Fine Grained Image Classification 
This repository is forked from https://github.com/MUKhattak/DeiT_ConvNeXt_KnowledgeDistillation.git repo and changes has been made to implement knowledge distillation with swin transformer base model as teacher model and convnext as student model.

This repository contains the PyTorch based training and evaluation codes for reproducing main results of our project. 

 ## Datasets
The instructions to download and install FGVC datasets is given below:-

<b> CUB Dataset </b>

Caltech-UCSD Birds (CUB) dataset is provided by Caltech Vision Lab. The dataset can be downloaded from this [link](https://www.vision.caltech.edu/datasets/cub_200_2011/).
The CUB-200 dataset consists of 11,788 images of 200 bird species, with each species having between 30 to 60 images.
The dataset folder should have the following structure:

```
CUB_dataset_root_folder/
    └─ images
    └─ image_class_labels.txt
    └─ train_test_split.txt
    └─ ....
```
<b> FGVC-Aircraft Dataset </b>

The FGVC-Aircraft-2013 dataset contains 10,000 images of airplanes, with each category having between 80 to 100 images. The Dataset is provided by Oxford Univeristy under the [link](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/). 

The dataset folder should have the following structure:

```
aircraft_dataset_root_folder/
    └─ data
        └─ images
            ├─ 0034309.jpg
            ├─ 0034958.jpg
            ├─ ...
        ├─ families.txt
        ├─ images_box.txt
        ├─ ...
    ├─ evaluation.m
    ├─ example_evaluation.m
    ├─ ...

```

<b> Stanford Dogs Dataset </b>

Stanford Dogs Dataset is provided by Stanford Univeristy under the [link](http://vision.stanford.edu/aditya86/ImageNetDogs/). 
The dataset consists of 20,580 images of 120 different dog breeds.
The dataset folder should have the following structure:

```
dog_dataset_root_folder/
    └─ Images
        ├─ n02092339-Weimaraner
            ├─ n02092339_107.jpg
            ├─ ....
        ├─ n02101388-Brittany_spaniel
            ├─ ....
        ├─ ....
    └─ splits
        ├─ file_list.mat
        ├─ test_list.mat
        ├─ train_list.mat

```

<b> FoodX Dataset </b>

FoodX-251 contains 251 food categories with a total of 310,000 images, making it one of the largest fgvc datasets present out there. the dataset can be downloaded from [here](https://github.com/karansikka1/iFood_2019). 
The dataset folder should have the following structure:

```
FoodX_dataset_root_folder/
    └─ annot
        ├─ class_list.txt
        ├─ train_info.csv
        ├─ val_info.csv
    └─ train_set
        ├─ train_039992.jpg
        ├─ ....
    └─ val_set
        ├─ val_005206.jpg
        ├─ ....
```



## Requirements and Installation

We have tested this code on Windows 10 LTS with Python 3.9. Follow the instructions below to setup the environment and install the dependencies.

py-3.9 -m pip install -r requirements.txt
 
## Training and Evaluation 

To finetune ConvNext model on CUB dataset, run the following command 

  ```bash
 $ python main.py --model convnext_base --drop-path 0.8 --input-size 384 --batch-size 16 --lr 5e-5 --warmup-epochs 0 --epochs 60 --weight-decay 1e-8 --cutmix 0 --mixup 0 --data-set CUB --data-path /path/to/dataset/root/folder --output_dir ./output/path --finetune /path/to/imagenet1k/pretrained/deit/weights.pth/
```

To further finetune ConvNext model (already finetuned on CUB dataset) using Knowledge Distillation from ConvNext teacher model, run the following commad:

  ```bash
 $ python main.py --model convnext_base_distilled --distillation-type hard --teacher-model convnext_base --drop-path 0.8 --input-size 384 --batch-size 16 --lr 5e-5 --warmup-epochs 0 --epochs 60 --weight-decay 1e-8 --cutmix 0 --mixup 0 --data-set CUB --data-path /path/to/dataset/root/folder --output_dir /path/to/save/output/files --finetune /path/of/deit/CUB_finetuned/weights 
```

Acknowledgement:
This code repo is forked and modified from the repository https://github.com/MUKhattak/DeiT_ConvNeXt_KnowledgeDistillation.git.

