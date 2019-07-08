
Code by **Nguyen Tuan Nam**

## 1. Introduction
This report briefly explains the process to reproduce the results of the paper "Cutting the Error by Half: Investigation of Very Deep CNN and Advanced Training Strategies for Document Image Classification (https://arxiv.org/abs/1704.03557)". The repository was written in Python 3.

## 2. Installation

The produced software depends on NumPy, Keras, Tensorflow, matplotlib, and opencv-python. Therefore, ones who want to try it should have these open sources installed beforehand.

A simple way to install them is using pip: 
```sh
	# sudo pip3 install -r requirements.txt
```
I also provided **Dockerfile** to deploy environtment for running the source code.

## 3. Usage

### 3.1. Data
Firstly, the two datasets were downloaded: the RVL_CDIP dataset was available at https://www.cs.cmu.edu/~aharley/rvl-cdip/; and the Tobacco dataset was available at https://www.kaggle.com/patrickaudriaz/tobacco3482jpg. Next, all downloaded files (rvl-cdip.tar.gz, labels_only.tar.gz, tobacco3482jpg.zip) were extracted in a root folder of this repo. 

After that, I ran "create_dataset.py" by the following command: 
```sh
	# python3 create_dataset.py
```
This command would move all images with the same label into the same auto-created folder on ``datasets`` folder, and remove all images in rvl-cdip training dataset contained in tobaco3482 dataset.

### 3.2.Training, validation, and testing

### 3.2.1. Training, validating, and testing the RVL_CDIP dataset
Firstly, I implemented the training and validating process on ``Train_rvl_cdip_first.ipynb``. Arcording to the orignal paper, there were four network architectures (alexnet, googlenet, vgg16, and resnet) and two training modes (training from scratch and fine-tuning from Imagenet pretrained model) to be used. Basically, the implementation here was similar to the training scenarios in the original paper, but with a few differences as followed:
* The Caffenet model (a single GPU version of AlexNet) was utilized instead of Alexnet since I did not have two GPU available. I also used InceptionV3 (an update version of Inception (Googlenet) with some improvement) instead of Googlenet. Because Keras did not provide Imagenet pretrained weight on Caffenet, I could not train this network on fine-tuning mode. 
* Overall, it took about an hour to train each epoch for each network architecture (Using GTX 1080), so I decided to train only 10 epochs instead of 40-80 epochs as in the original paper. Upon finishing the training, I chose the model's weights with the highest validation accuracy to evaluate the model performance on the testing dataset. The evaluation process was implemented on ``Evaluate_rvl_cdip_first.ipynb``. Thr final result could be found in the following table:

<img width="587" alt="Ảnh chụp Màn hình 2019-07-08 lúc 20 58 07" src="https://user-images.githubusercontent.com/48004872/60816101-28665300-a1c3-11e9-9650-9369932943b4.png">


### 3.2.2. Training and testing the Tobacco dataset3482
In this part, I also followed the method in the original paper. However, instead of using training examples per class in range 10 to 100, I used the range [20,40,60,80,100] in order to reduce a half of the time without altering the proposed training scheme. After this training, I ran ``python3 plot_tobaco.py`` to draw a line graph. This graph illustrated the relationship between mean_accuracy on the test dataset and the number of training example per class, as a result of the combination between 3 training modes (Document_pretrained, Imagenet_pretrained, no_pretrained) and 4 network architectures. 

![image](https://user-images.githubusercontent.com/48004872/60815716-67e06f80-a1c2-11e9-959a-17f8a573fbd6.png)

The final result on this dataset was shown in the following table. 

<img width="604" alt="Ảnh chụp Màn hình 2019-07-08 lúc 20 58 10" src="https://user-images.githubusercontent.com/48004872/60816117-31efbb00-a1c3-11e9-9cbd-2c0a8a357a49.png">

Finally, all best model checkpoints were stored at https://drive.google.com/drive/u/1/folders/1_3ajKhF5T6nhR7CojpxzU4O89sXGkSn9

### 3.3. Inference testing
I created 2 source code files ``inference_RVL_CDIP.py`` and ``inference_tobaco.py`` to support the inference testing model. I put some testing images in 2 folders ``input_RVL`` and ``input_tobaco``, run ``inference_RVL_CDIP.py`` and ``inference_tobaco.py`` to get ``your_result.txt`` in both ``output_RVL`` and ``output_tobaco``.

## 4. References

Muhammad Zeshan Afzal, Andreas Kölsch, Sheraz Ahmed, Marcus Liwicki (2017), "Cutting the Error by Half: Investigation of Very Deep CNN and Advanced Training Strategies for Document Image Classification", (https://arxiv.org/abs/1704.03557)

