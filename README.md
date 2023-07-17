# consistency-ranking-loss
The code for ICLR2023 paper: [Confidence Estimation Using Unlabeled Data](https://openreview.net/forum?id=sOXU-PEJSgQ)  

### 1. Environment setup ###
Using conda:

conda env create -f consistency.yml

### 2. Training data for CIFAR10 ###
Both labeled and unlabeled data are available on [Google Drive](https://drive.google.com/file/d/1WpWVMyn8qEcKT77DIZAyMqcogJNhjVha/view?usp=sharing)  

### 2. Train and evaluate on CIFAR10 ###
Change the data location at lines in train_and_eval.py: 103, 108, 112

Run command: 

python train_and_eval.py --save_folder [Results saving folder] --gpus [GPU ID] --label_size 5000 [labeled samples size (2500, 5000, 10000)]
