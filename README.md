# post_training_quantization-
A module that enables post training quantization for a pytorch deep learning classification model 
## Setup
Set up the environment for using post training quantization by installing pytorch from the following link
(https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) 

## Running the Program
- -m  or --model: selecting 1 of 12 pretrained models implemented in torchvision   
- -c or --checkpoint: path for the checkpoint of the model that will go through quantization   
- -d or --data: path for the data which the original model was built upon   
- -n or --target: number of classes   
 
example of performing the post training quantization of a model built by a certain dataset 
```bash
python3 main.py -m resnet18 -c /ssd3/jhahn/ptq/model/saved_model/resnet18-model1.pth -d /ssd3/jhahn/ptq/data/imagenet_1k/ -n 1000
```
## Caution
1. In order to use this module the dataset should be in the same format as the following dataset \n 
https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip
2. This module only works for 12 classification model implemented in torch vision. 
The 12 models are the following:  
(1)resnet18   
(2)alexnet   
(3)squeezenet   
(4)vgg16   
(5)densenet   
(6)inception   
(7)googlenet   
(8)shufflenet   
(9)mobilenet   
(10)resnext50_32x4d   
(11)wide_resnet50_2   
(12)mnasnet  
https://pytorch.org/docs/stable/torchvision/models.html
## Result
![result](result.PNG)

