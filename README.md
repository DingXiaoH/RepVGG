# RepVGG: Making VGG-style ConvNets Great Again (PyTorch)

This is a super simple ConvNet architecture that achieves over 80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU! This repo contains the pretrained models, code for building the model, training, and the conversion from training-time model to inference-time.

You can use repvgg.py to build models and train with your code. I am working on an extremely simple training script in the style of the pytorch official example. I am uploading the pretrained weights.

# Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80\% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.


![image](https://github.com/DingXiaoH/RepVGG/blob/main/arch.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/speed_acc.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/table.PNG)

# Use our pretrained models

The following are the models trained on ImageNet. For the ease of transfer learning on other tasks, they are all training-time models (with identity and 1x1 branches). You can test the accuracy by running
```
python test.py [imagenet-folder with train and val folders] train [path to weights file] -a [model name]
```
Here "train" indicates the training-time architecture. For example,
```
python test.py [imagenet-folder with train and val folders] train RepVGG-B2-train.pth -a RepVGG-B2
```

| Model        | Top-1 acc           | Google Drive  | Baidu Cloud |
| -------------|:-------------:| -----:| -----:|
| RepVGG-A0   | 72.41 	|  	uploading | uploading |
| RepVGG-A1   | 74.46 	|  	uploading | uploading |
| RepVGG-B0   | 75.14 	|  	uploading | uploading |
| RepVGG-A2   | 76.48 	|  	uploading | uploading |
| RepVGG-B1g4   | 77.58 |  	uploading | uploading |
| RepVGG-B1g2   | 77.78 |  	uploading | uploading |
| RepVGG-B1   | 78.37 |  	uploading | uploading |
| RepVGG-B2g4   | 78.50 |  	uploading | uploading |
| RepVGG-B2   | 78.78 |  	uploading | uploading |


# Convert the training-time models into inference-time

You can convert a trained model into the inference-time structure with
```
python convert.py [weights file of the training-time model to load] [path to save] -a [model name]
```
For example,
```
python convert.py RepVGG-B2-train.pth RepVGG-B2-deploy.pth -a RepVGG-B2
```
Then you can test the inference-time model by
```
python test.py [imagenet-folder with train and val folders] deploy RepVGG-B2-deploy.pth -a RepVGG-B2
```
Note that the "deploy" arg builds a inference-time model.


# ImageNet training settings

We trained for 120 epochs with cosine learning rate decay from 0.1 to 0. We used 8 GPUs, global batch size of 256, weight decay of 1e-4 (no weight decay on fc.bias, bn.weight and bn.bias), and the same simple data preprocssing as the PyTorch official example:
```
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])
```
            

# Use like this in your own code
```
train_model = create_RepVGG_A0(deploy=False)
train train_model ...
deploy_model = repvgg_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
```

# FAQ
