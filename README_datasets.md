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

