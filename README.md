# RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021) (PyTorch)

# Updates

***June 22, 2021*** A pure-VGG model (without [SE](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)) seems to outperform some vision transformer models with a better training scheme. Training.

***June 11, 2021*** An example of using a simple toolbox, torch.quantization, to quantize RepVGG. Please check it below.

***June 10, 2021*** Training with the custom weight decay has been tested. Just add ```--custwd``` to the training command.

***June 8, 2021*** found out that high-performance quantization required a custom weight decay. Such a weight decay also improves the full-precision accuracy. Will release the quantized models after tuning the hyper-parameters and finishing the QAT.

***Apr 25, 2021*** A deeper RepVGG model achieves **83.55\% top-1 accuracy on ImageNet** with [SE](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) blocks and an input resolution of 320x320 (and a wider version achieves **83.67\% accuracy** _without SE_). Note that it is trained with 224x224 but tested with 320x320, so that it is still trainable with a global batch size of 256 on a single machine with 8 1080Ti GPUs. If you test it with 224x224, the top-1 accuracy will be 81.82%. It has 1, 8, 14, 24, 1 layers in the 5 stages respectively. The width multipliers are a=2.5 and b=5 (the same as RepVGG-B2). The model name is "RepVGG-D2se". The code for building the model (repvgg.py) and testing with 320x320 (the testing example below) has been updated and the weights have been released at Google Drive and Baidu Cloud. Please check the links below.

***Apr 4, 2021*** A better implementation. For a RepVGG model or a model with RepVGG as one of its components (e.g., the backbone), you can convert the whole model by simply calling **switch_to_deploy** of every RepVGG block. This is the recommended way. Examples are shown in **convert.py** and **example_pspnet.py**.
```
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
```
***Apr 4, 2021*** An example of using RepVGG as the backbone of PSPNet for semantic segmentation (**example_pspnet.py**). It shows how to 1) build a PSPNet with RepVGG backbone, 2) load the ImageNet-pretrained weights, 3) convert the whole model with **switch_to_deploy**, 4) save and use the converted model for inference.

***Jan 13 - Feb 5, 2021*** You can get the equivalent kernel and bias in a differentiable way at any time (get_equivalent_kernel_bias in repvgg.py). This may help training-based pruning or quantization. This training script (a super simple PyTorch-official-example-style script) has been tested with RepVGG-A0 and B1. The results are even slightly better than those reported in the paper.

# Introduction

This is a super simple ConvNet architecture that achieves over **80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU**! This repo contains the **pretrained models**, code for building the model, training, and the conversion from training-time model to inference-time, and **an example of using RepVGG for semantic segmentation**.

The MegEngine version: https://github.com/megvii-model/RepVGG.

TensorRT implemention with C++ API by @upczww https://github.com/upczww/TensorRT-RepVGG. Great work!

Another PyTorch implementation by @zjykzj https://github.com/ZJCV/ZCls. He also presented detailed benchmarks at https://zcls.readthedocs.io/en/latest/benchmark-repvgg/. Nice work!

Included in a famous model zoo (over 7k stars) https://github.com/rwightman/pytorch-image-models.

Objax implementation and models by @benjaminjellis. Great work! https://github.com/benjaminjellis/Objax-RepVGG.

Citation:

    @inproceedings{ding2021repvgg,
    title={Repvgg: Making vgg-style convnets great again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={13733--13742},
    year={2021}
    }

# Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80\% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.

![image](https://github.com/DingXiaoH/RepVGG/blob/main/arch.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/speed_acc.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/table.PNG)

# Use our pretrained models

You may download _all_ of the ImageNet-pretrained models reported in the paper from Google Drive (https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq?usp=sharing) or Baidu Cloud (https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ, the access code is "rvgg"). For the ease of transfer learning on other tasks, they are all training-time models (with identity and 1x1 branches). You may test the accuracy by running
```
python test.py [imagenet-folder with train and val folders] train [path to weights file] -a [model name]
```
The default input resolution is 224x224. Here "train" indicates the training-time architecture, and the valid model names include
```
RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4, RepVGG-B2, RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4
```
For example,
```
python test.py [imagenet-folder with train and val folders] train RepVGG-B2-train.pth -a RepVGG-B2
```
To test the latest model RepVGG-D2se with 320x320 inputs,
```
python test.py [imagenet-folder with train and val folders] train RepVGG-D2se-200epochs-train.pth -a RepVGG-D2se -r 320
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


# ImageNet training

We trained for 120 epochs with cosine learning rate decay from 0.1 to 0. We used 8 GPUs, global batch size of 256, weight decay of 1e-4 (no weight decay on fc.bias, bn.bias, rbr_dense.bn.weight and rbr_1x1.bn.weight) (weight decay on rbr_identity.weight makes little difference, and it is better to use it in most of the cases), and the same simple data preprocssing as the PyTorch official example:
```
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```
The multi-processing training script in this repo is based on the [official PyTorch example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) for the simplicity and better readability. The only modifications include the model-building part, cosine learning rate scheduler, and the SGD optimizer that uses no weight decay on some parameters. You may find these code segments useful for your training code. 
We tested this training script with RepVGG-A0 and RepVGG-B1. The accuracy was 72.44 and 78.38, respectively, which was almost the same as (and even better than) the results we reported in the paper (72.41 and 78.37). You may train and test like this:
```
python train.py -a RepVGG-A0 --dist-url 'tcp://127.0.0.1:23333' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 [imagenet-folder with train and val folders] --tag hello --custwd --wd 4e-5
python test.py [imagenet-folder with train and val folders] train RepVGG-A0_hello_best.pth.tar -a RepVGG-A0
```
I would really appreciate it if you share with me your re-implementation results with other models. 
            

# Use like this in your own code
```
from repvgg import repvgg_model_convert, create_RepVGG_A0
train_model = create_RepVGG_A0(deploy=False)
train_model.load_state_dict(torch.load('RepVGG-A0-train.pth'))          # or train from scratch
# do whatever you want with train_model
deploy_model = repvgg_model_convert(train_model, save_path='RepVGG-A0-deploy.pth')
# do whatever you want with deploy_model
```
or
```
deploy_model = create_RepVGG_A0(deploy=True)
deploy_model.load_state_dict(torch.load('RepVGG-A0-deploy.pth'))
# do whatever you want with deploy_model
```
If you use RepVGG as a component of another model, the conversion is as simple as calling **switch_to_deploy** of every RepVGG block. 

# Quantization

The best solution for quantization is to constrain the equivalent kernel (get_equivalent_kernel_bias() in repvgg.py) to be low-bit (e.g., make every param in {-127, -126, .., 126, 127} for int8), instead of constraining the params of every kernel separately for an ordinary model.

For the simplicity, we can also use the off-the-shelf quantization toolboxes to quantize RepVGG. We use the simple QAT (quantization-aware training) tool in torch.quantization as an example.

1. The base model is trained with the custom weight decay (```--custwd```) and converted into inference-time structure. We insert BN after the converted 3x3 conv layers because QAT with torch.quantization requires BN. Specifically, we run the model on ImageNet training set and record the mean/std statistics and use them to initialize the BN layers, and initialize BN.gamma/beta accordingly so that the saved model has the same outputs as the inference-time model. 

```
python train.py -a RepVGG-A0 --dist-url 'tcp://127.0.0.1:23333' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 [imagenet-folder] --tag hello --custwd
python convert.py RepVGG-A0_hello_best.pth.tar RepVGG-A0_base.pth -a RepVGG-A0 
python insert_bn.py [imagenet-folder] RepVGG-A0_base.pth RepVGG-A0_withBN.pth -a RepVGG-A0 -b 32 -n 40000
```

2. Build the model, prepare it for QAT (torch.quantization.prepare_qat), and conduct QAT. The hyper-parameters may not be optimal and I am tuning them.
```
python quantization/quant_qat_train.py [imagenet-folder] -j 32 --epochs 20 -b 256 --lr 1e-3 --weight-decay 4e-5 --base-weights RepVGG-A0_withBN.pth --tag quanttest
```


# FAQs

**Q**: Is the inference-time model's output the _same_ as the training-time model?

**A**: Yes. You can verify that by
```
import torch
train_model = create_RepVGG_A0(deploy=False)
train_model.eval()      # Don't forget to call this before inference.
deploy_model = repvgg_model_convert(train_model)
x = torch.randn(1, 3, 224, 224)
train_y = train_model(x)
deploy_y = deploy_model(x)
print(((train_y - deploy_y) ** 2).sum())    # Will be around 1e-10
```

**Q**: How to use the pretrained RepVGG models for other tasks?

**A**: It is better to finetune the training-time RepVGG models on your datasets. Then you should do the conversion after finetuning and before you deploy the models. For example, say you want to use PSPNet for semantic segmentation, you should build a PSPNet with a training-time RepVGG model as the backbone, load pre-trained weights into the backbone, and finetune the PSPNet on your segmentation dataset. Then you should convert the backbone following the code provided in this repo and keep the other task-specific structures (the PSPNet parts, in this case). The pseudo code will be like
```
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
```
There is an example in **example_pspnet.py**.

Finetuning with a converted RepVGG also makes sense if you insert a BN after each conv (please see step 1 of the quantization part), but the performance may be slightly lower.

**Q**: I tried to finetune your model with multiple GPUs but got an error. Why are the names of params like "stage1.0.rbr_dense.conv.weight" in the downloaded weight file but sometimes like "module.stage1.0.rbr_dense.conv.weight" (shown by nn.Module.named_parameters()) in my model?

**A**: DistributedDataParallel may prefix "module." to the name of params and cause a mismatch when loading weights by name. The simplest solution is to load the weights (model.load_state_dict(...)) before DistributedDataParallel(model). Otherwise, you may insert "module." before the names like this
```
checkpoint = torch.load(...)    # This is just a name-value dict
ckpt = {('module.' + k) : v for k, v in checkpoint.items()}
model.load_state_dict(ckpt)
```
Likewise, if the param names in the checkpoint file start with "module." but those in your model do not, you may strip the names like line 50 in test.py.
```
ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}   # strip the names
model.load_state_dict(ckpt)
```
**Q**: So a RepVGG model derives the equivalent 3x3 kernels before each forwarding to save computations?

**A**: No! More precisely, we do the conversion only once right after training. Then the training-time model can be discarded, and the resultant model only has 3x3 kernels. We only save and use the resultant model.


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. (preprint, 2021) **A powerful MLP-style CNN building block**\
[RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883)\
[code](https://github.com/DingXiaoH/RepMLP).

2. (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **83.55%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. (preprint, 2020) **State-of-the-art** channel pruning\
[Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
