# RepVGG: Making VGG-style ConvNets Great Again (PyTorch)

This is a super simple ConvNet architecture that achieves over 80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU! This repo contains the pretrained models, code for building the model, training, and the conversion from training-time model to inference-time.

The MegEngine version: https://github.com/megvii-model/RepVGG.

# Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80\% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.

![image](https://github.com/DingXiaoH/RepVGG/blob/main/arch.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/speed_acc.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/table.PNG)

# Use our pretrained models

You may download _all_ of the ImageNet-pretrained models reported in the paper from Google Drive (https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq?usp=sharing) or Baidu Cloud (https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ, the access code is "rvgg"). For the ease of transfer learning on other tasks, they are all training-time models (with identity and 1x1 branches). You can test the accuracy by running
```
python test.py [imagenet-folder with train and val folders] train [path to weights file] -a [model name]
```
Here "train" indicates the training-time architecture. For example,
```
python test.py [imagenet-folder with train and val folders] train RepVGG-B2-train.pth -a RepVGG-B2
```


# Convert the training-time models into inference-time

You may convert a trained model into the inference-time structure with
```
python convert.py [weights file of the training-time model to load] [path to save] -a [model name]
```
For example,
```
python convert.py RepVGG-B2-train.pth RepVGG-B2-deploy.pth -a RepVGG-B2
```
Then you may test the inference-time model by
```
python test.py [imagenet-folder with train and val folders] deploy RepVGG-B2-deploy.pth -a RepVGG-B2
```
Note that the argument "deploy" builds an inference-time model.


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
The training script in this repo has not been tested. Will do it tomorrow.
            

# Use like this in your own code
```
train_model = create_RepVGG_A0(deploy=False)
train train_model ...
deploy_model = repvgg_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
```

# FAQs

Q: How to use the pretrained RepVGG models for other tasks?

A: It is better to finetune the training-time RepVGG models on your datasets. Then you should do the conversion after finetuning and before you deploy the models. For example, say you want to use PSPNet for semantic segmentation, you should build a PSPNet with a training-time RepVGG model as the backbone, load pre-trained weights into the backbone, and finetune the PSPNet on your segmentation dataset. Then you should convert the backbone following the code provided in this repo and keep the other task-specific structures (the PSPNet parts, in this case). Finetuning with a converted RepVGG also makes sense if you insert a BN after each conv (the converted conv.bias params can be discarded), but the performance may be slightly lower.

Q: How to quantize a RepVGG model?

A1: Post-training quantization. After training and conversion, you can quantize the converted model with any post-training quantization method. Then you can insert a BN after each conv and finetune to recover the accuracy just like you quantize and finetune the other models. This is the recommended solution.

A2: Quantization-aware training. During the quantization-aware training, instead of constraining the params in a single kernel (e.g., making every param in {-127, -126, .., 126, 127} for int8) for ordinary models, you should constrain the equivalent kernel (kernel3 / std3 * gamma3 + kernel1 / std1 * gamma1 + identity / std0 * gamma0). 
