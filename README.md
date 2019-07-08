
Code by **Nguyen Tuan Nam**

## 1. Introduction
This is source code reimplement a paper Cutting the Error by Half: Investigation of Very Deep CNN and Advanced Training
Strategies for Document Image Classification (https://arxiv.org/abs/1704.03557). This system is written by Python 3


## 2. Installation

This software depends on NumPy, Keras, Tensorflow, matplotlib, opencv-python. You must have them installed before using.
The simple way to install them is using pip: 
```sh
	# sudo pip3 install -r requirements.txt
```
We also provide **Dockerfile** to deploy environtment to run source code

## 3. Usage

### 3.1. Data
Downloading RVL_CDIP dataset (https://www.cs.cmu.edu/~aharley/rvl-cdip/) and Tobacco dataset(https://www.kaggle.com/patrickaudriaz/tobacco3482jpg). And extract all downloaded files(rvl-cdip.tar.gz, labels_only.tar.gz, tobacco3482jpg.zip) in root folder of this repo. 

After that, we run create_dataset.py by a following command: 
```sh
	# python3 create_dataset.py
```
This command will move all image with same label to same folder on ``datasets`` folder  and remove all image of rvl-cdip training dataset which is contained in tobaco3482 datasets.

### 3.2.Training, validation and testing

### 3.2.1. Training,validation and testing RVC_CDIP dataset
Firstly, we implement training and validation process on ``Train_rvl_cdip_first.ipynb``. Arcording to orignal paper, There are four network architectures(alexnet, googlenet, vgg16, resnet) and two training mode (training from scratch and fine-tuning from Imagenet pretrained model)  . Basically, my implementation is similar to training scheme on original paper but with few differents:
* We use Caffenetmodel (a single GPU version of AlexNet) instead of Alexnet since we do not have two GPU. We also use InceptionV3(an update version of Inception(Googlenet) with some improvement) instead of Googlenet. Because Keras does not provide Imagenet pretrained weight on Caffenet, so we can not train this network on fine-tuning mode. 
* Overall, each epoch took about an hour for training with each network architecture (Using GTX 1080) so I decide to train only 10 epochs instead of 40-80 epochs in original paper. After finishing training 10 epochs , I choose model's weights with highest validation accuracy to evaluate on testing dataset. I implement evaluate process on ``Evaluate_rvl_cdip_first.ipynb``  .I report my final result in the following table.

<img width="587" alt="Ảnh chụp Màn hình 2019-07-08 lúc 20 58 07" src="https://user-images.githubusercontent.com/48004872/60816101-28665300-a1c3-11e9-9650-9369932943b4.png">


### 3.2.2. Training and testing Tobacco dataset3482
In this part, my implementation is also similar to original paper. But instead of using of number of training example per class in range 10 to 100, my number of training example per class in range [20,40,60,80,100] can reduce a half of training time without change a goal of this training scheme in original paper. After finishing training process, we run ``plot_tobaco.py`` to draw a line graph. This graph illustrates the relationship between mean_accuracy on test dataset and number of training example per class as a result of the combination between 3 training modes (Document_pretrained, Imagenet_pretrained, no_pretrained) and 4 network architectures. 

![image](https://user-images.githubusercontent.com/48004872/60815716-67e06f80-a1c2-11e9-959a-17f8a573fbd6.png)

My final result on this dataset is shown on the following table. 

<img width="604" alt="Ảnh chụp Màn hình 2019-07-08 lúc 20 58 10" src="https://user-images.githubusercontent.com/48004872/60816117-31efbb00-a1c3-11e9-9cbd-2c0a8a357a49.png">


### 3.3. Inference testing
I created 2 source code files ``inference_RVL_CDIP.py`` and ``inference_tobaco.py`` to support inference testing model. We put some testing images on 2 folder ``input_RVL`` and ``input_tobaco``, run ``inference_RVL_CDIP.py`` and ``inference_tobaco.py`` to get a ``your_result.txt`` in ``output_RVL`` and ``output_tobaco`` respectively

