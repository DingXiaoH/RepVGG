# RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021) (PyTorch)

This is a super simple ConvNet architecture that achieves over **80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU**! This repo contains the **pretrained models**, code for building the model, training, and the conversion from training-time model to inference-time.

The MegEngine version: https://github.com/megvii-model/RepVGG.

TensorRT implemention with C++ API by @upczww https://github.com/upczww/TensorRT-RepVGG. Great work!

Another nice PyTorch implementation by @zjykzj https://github.com/ZJCV/ZCls.

Included in a famous model zoo (over 7k stars) https://github.com/rwightman/pytorch-image-models.

Update (Jan 13, 2021): you can get the equivalent kernel and bias in a differentiable way at any time (get_equivalent_kernel_bias in repvgg.py). This may help training-based pruning or quantization.

Update (Jan 31, 2021): this training script (a super simple PyTorch-official-example-style script) has been tested with RepVGG-A0 and B1. The results are even slightly better than those reported in the paper.

Update (Feb 5, 2021): added a function (**whole_model_convert in repvgg.py**) for easily converting a customized model with RepVGG as one of its components (e.g., the backbone of a semantic segmentation model). It will convert the RepVGG blocks only and keep the other parts. If it does not work with your model, please raise an issue.

Citation:

    @article{ding2021repvgg,
    title={RepVGG: Making VGG-style ConvNets Great Again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    journal={arXiv preprint arXiv:2101.03697},
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
Here "train" indicates the training-time architecture, and the valid model names include
```
RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4, RepVGG-B2, RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4
```
For example,
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
python train.py -a RepVGG-A0 --dist-url 'tcp://127.0.0.1:23333' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 [imagenet-folder with train and val folders]
python test.py [imagenet-folder with train and val folders] train model_best.pth.tar -a RepVGG-A0
```
I would really appreciate it if you share with me your re-implementation results with other models. 
            

# Use like this in your own code
```
from repvgg import repvgg_model_convert, create_RepVGG_A0
train_model = create_RepVGG_A0(deploy=False)
train_model.load_state_dict(torch.load('RepVGG-A0-train.pth'))          # or train from scratch
# do whatever you want with train_model
deploy_model = repvgg_model_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
# do whatever you want with deploy_model
```
or
```
deploy_model = create_RepVGG_A0(deploy=True)
deploy_model.load_state_dict(torch.load('RepVGG-A0-deploy.pth'))
# do whatever you want with deploy_model
```
If you use RepVGG as a component of another model, it will be more convenient to use **whole_model_convert in repvgg.py** for the conversion. Please refer to FAQs for more details.

# FAQs

**Q**: Is the inference-time model's output the _same_ as the training-time model?

**A**: Yes. You can verify that by
```
import torch
train_model = create_RepVGG_A0(deploy=False)
train_model.eval()      # Don't forget to call this before inference.
deploy_model = repvgg_model_convert(train_model, create_RepVGG_A0)
x = torch.randn(1, 3, 224, 224)
train_y = train_model(x)
deploy_y = deploy_model(x)
print(((train_y - deploy_y) ** 2).sum())    # Will be around 1e-10
```

**Q**: How to use the pretrained RepVGG models for other tasks?

**A**: It is better to finetune the training-time RepVGG models on your datasets. Then you should do the conversion after finetuning and before you deploy the models. For example, say you want to use PSPNet for semantic segmentation, you should build a PSPNet with a training-time RepVGG model as the backbone, load pre-trained weights into the backbone, and finetune the PSPNet on your segmentation dataset. Then you should convert the backbone following the code provided in this repo and keep the other task-specific structures (the PSPNet parts, in this case). Now we provide a function (**whole_model_convert in repvgg.py**) to do this. The pseudo code will be like
```
train_backbone = create_RepVGG_B2(deploy=False)
train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
train_pspnet = build_pspnet(backbone=train_backbone)
segmentation_train(train_pspnet)
deploy_backbone = create_RepVGG_B2(deploy=True)
deploy_pspnet = build_pspnet(backbone=deploy_backbone)
whole_model_convert(train_pspnet, deploy_pspnet)
segmentation_test(deploy_pspnet)
torch.save(deploy_pspnet.state_dict(), 'deploy_pspnet.pth')
```
Finetuning with a converted RepVGG also makes sense if you insert a BN after each conv (the converted conv.bias params can be discarded), but the performance may be slightly lower.

**Q**: How to quantize a RepVGG model?

**A1**: Post-training quantization. After training and conversion, you may quantize the converted model with any post-training quantization method. Then you may insert a BN after each conv and finetune to recover the accuracy just like you quantize and finetune the other models. This is the recommended solution.

**A2**: Quantization-aware training. During the quantization-aware training, instead of constraining the params in a single kernel (e.g., making every param in {-127, -126, .., 126, 127} for int8) for ordinary models, you should constrain the equivalent kernel (get_equivalent_kernel_bias() in repvgg.py). 

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

**Simple and powerful VGG-style ConvNet architecture** (preprint, 2021): [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)      (https://github.com/DingXiaoH/RepVGG)

**State-of-the-art** channel pruning (preprint, 2020): [Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260) (https://github.com/DingXiaoH/ResRep)

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)
