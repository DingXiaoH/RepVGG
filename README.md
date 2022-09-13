# RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021) (PyTorch)

## Highlights (Sep. 1st, 2022)

RepVGG and the methodology of re-parameterization have been used in **YOLOv6** ([paper](https://arxiv.org/abs/2209.02976), [code](https://github.com/meituan/YOLOv6))  and **YOLOv7** ([paper](https://arxiv.org/abs/2207.02696), [code](https://github.com/WongKinYiu/yolov7)). 

I have re-organized this repository and released the RepVGGplus-L2pse model with 84.06% ImageNet accuracy. Will release more RepVGGplus models in this month.

## Introduction

This is a super simple ConvNet architecture that achieves over **84% top-1 accuracy on ImageNet** with a VGG-like architecture! This repo contains the **pretrained models**, code for building the model, training, and the conversion from training-time model to inference-time, and **an example of using RepVGG for semantic segmentation**.

[The MegEngine version](https://github.com/megvii-model/RepVGG)

[TensorRT implemention with C++ API by @upczww](https://github.com/upczww/TensorRT-RepVGG). Great work!

[Another PyTorch implementation by @zjykzj](https://github.com/ZJCV/ZCls). He also presented detailed benchmarks [here](https://zcls.readthedocs.io/en/latest/benchmark-repvgg/). Nice work!

Included in a famous PyTorch model zoo https://github.com/rwightman/pytorch-image-models.

[Objax implementation and models by @benjaminjellis](https://github.com/benjaminjellis/Objax-RepVGG). Great work!

Included in the [MegEngine Basecls model zoo](https://github.com/megvii-research/basecls/tree/main/zoo/public/repvgg).

Citation:

    @inproceedings{ding2021repvgg,
    title={Repvgg: Making vgg-style convnets great again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={13733--13742},
    year={2021}
    }


## From RepVGG to RepVGGplus

We have released an improved architecture named RepVGGplus on top of the original version presented in the CVPR-2021 paper.

1. RepVGGplus is deeper

2. RepVGGplus has auxiliary classifiers during training, which can also be removed for inference

3. (Optional) RepVGGplus uses Squeeze-and-Excitation blocks to further improve the performance.

RepVGGplus outperformed several recent visual transformers with a top-1 accuracy of **84.06%** and higher throughput. Our training script is based on [codebase of Swin Transformer](https://github.com/microsoft/Swin-Transformer/). The throughput is tested with the Swin codebase as well. We would like to thank the authors of [Swin](https://arxiv.org/abs/2103.14030) for their clean and well-structured code. 

| Model        | Train image size       | Test size  | ImageNet top-1 | Throughput (examples/second), 320, batchsize=128, 2080Ti) |
| ------------- |:-------------:| -----:| -----:| -----:|
| RepVGGplus-L2pse    | 256 	|  	320 |   **84.06%**   |**147** |
| Swin Transformer | 320    |   320 |   84.0%     |102 |

("pse" means Squeeze-and-Excitation blocks after ReLU.)

Download this model: [Google Drive](https://drive.google.com/file/d/1x8VNLpfuLzg0xXDVIZv9yIIgqnSMoK-W/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/19YwKCTSPVgJu5Ueg0Q78-w?pwd=rvgg).

To train or finetune it, slightly change your training code like this:
```
        #   Build model and data loader as usual
        for samples, targets in enumerate(train_data_loader):
            #   ......
            outputs = model(samples)                        #   Your original code
            if type(outputs) is dict:                       
                #   A training-time RepVGGplus outputs a dict. The items are:
                    #   'main':     the output of the final layer
                    #   '*aux*':    the output of auxiliary classifiers
                loss = 0
                for name, pred in outputs.items():
                    if 'aux' in name:
                        loss += 0.1 * criterion(pred, targets)          #  Assume "criterion" is cross-entropy for classification
                    else:
                        loss += criterion(pred, targets)
            else:
                loss = criterion(outputs, targets)          #   Your original code
            #   Backward as usual
            #   ......
```

To use it for downstream tasks like semantic segmentation, just discard the aux classifiers and the final FC layer.

Pleased note that the custom weight decay trick I described last year turned out to be insignificant in our recent experiments (84.16% ImageNet acc and negligible improvements on other tasks), so I decided to stop using it as a new feature of RepVGGplus. You may try it optionally on your task. Please refer to the last part of this page for details.


## Use our pretrained model

You may download _all_ of the ImageNet-pretrained models reported in the paper from Google Drive (https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq?usp=sharing) or Baidu Cloud (https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ, the access code is "rvgg"). For the ease of transfer learning on other tasks, they are all training-time models (with identity and 1x1 branches). You may test the accuracy by running
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 main.py --arch [model name] --data-path [/path/to/imagenet] --batch-size 32 --tag test --eval --resume [/path/to/weights/file] --opts DATA.DATASET imagenet DATA.IMG_SIZE [224 or 320]
```
The valid model names include
```
RepVGGplus-L2pse, RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4, RepVGG-B2, RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4
```

## Convert a training-time RepVGG into the inference-time structure

For a RepVGG model or a model with RepVGG as one of its components (e.g., the backbone), you can convert the whole model by simply calling **switch_to_deploy** of every RepVGG block. This is the recommended way. Examples are shown in ```tools/convert.py``` and ```example_pspnet.py```.
```
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
```
We have also released a script for the conversion. For example, 
```
python convert.py RepVGGplus-L2pse-train256-acc84.06.pth RepVGGplus-L2pse-deploy.pth -a RepVGGplus-L2pse
```
Then you may build the inference-time model with ```--deploy```, load the converted weights and test
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 main.py --arch RepVGGplus-L2pse --data-path [/path/to/imagenet] --batch-size 32 --tag test --eval --resume RepVGGplus-L2pse-deploy.pth --deploy --opts DATA.DATASET imagenet DATA.IMG_SIZE [224 or 320]
```

Except for the final conversion after training, you may want to get the equivalent kernel and bias **during training** in a **differentiable** way at any time (```get_equivalent_kernel_bias``` in ```repvgg.py```). This may help training-based pruning or quantization. 

## Train from scratch

### Reproduce RepVGGplus-L2pse (not presented in the paper)

To train the recently released RepVGGplus-L2pse from scratch, activate mixup and use ```--AUG.PRESET raug15``` for RandAug.
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main.py --arch RepVGGplus-L2pse --data-path [/path/to/imagenet] --batch-size 32 --tag train_from_scratch --output-dir /path/to/save/the/log/and/checkpoints --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 AUG.MIXUP 0.2 DATA.DATASET imagenet DATA.IMG_SIZE 256 DATA.TEST_SIZE 320
```

### Reproduce original RepVGG results reported in the paper

To reproduce the models reported in the CVPR-2021 paper, use no mixup nor RandAug.
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main.py --arch [model name] --data-path [/path/to/imagenet] --batch-size 32 --tag train_from_scratch --output-dir /path/to/save/the/log/and/checkpoints --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 1e-4 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET weak AUG.MIXUP 0.0 DATA.DATASET imagenet DATA.IMG_SIZE 224
```
The original RepVGG models were trained in 120 epochs with cosine learning rate decay from 0.1 to 0. We used 8 GPUs, global batch size of 256, weight decay of 1e-4 (no weight decay on fc.bias, bn.bias, rbr_dense.bn.weight and rbr_1x1.bn.weight) (weight decay on rbr_identity.weight makes little difference, and it is better to use it in most of the cases), and the same simple data preprocssing as the PyTorch official example:
```
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```







## Other released models not presented in the paper

***Apr 25, 2021*** A deeper RepVGG model achieves **83.55\% top-1 accuracy on ImageNet** with [SE](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) blocks and an input resolution of 320x320 (and a wider version achieves **83.67\% accuracy** _without SE_). Note that it is trained with 224x224 but tested with 320x320, so that it is still trainable with a global batch size of 256 on a single machine with 8 1080Ti GPUs. If you test it with 224x224, the top-1 accuracy will be 81.82%. It has 1, 8, 14, 24, 1 layers in the 5 stages respectively. The width multipliers are a=2.5 and b=5 (the same as RepVGG-B2). The model name is "RepVGG-D2se". The code for building the model (repvgg.py) and testing with 320x320 (the testing example below) has been updated and the weights have been released at Google Drive and Baidu Cloud. Please check the links below.


## Example 1: use Structural Re-parameterization like this in your own code
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


## Example 2: use RepVGG as the backbone for downstream tasks

I would suggest you use popular frameworks like MMDetection and MMSegmentation. The features from any stage or layer of RepVGG can be fed into the task-specific heads. If you are not familiar with such frameworks and just would like to see a simple example, please check ```example_pspnet.py```, which shows how to use RepVGG as the backbone of PSPNet for semantic segmentation: 1) build a PSPNet with RepVGG backbone, 2) load the ImageNet-pretrained weights, 3) convert the whole model with **switch_to_deploy**, 4) save and use the converted model for inference.



## Quantization

RepVGG works fine with FP16 but the accuracy may decrease when directly quantized to INT8. If IN8 quantization is essential to your application, we suggest three practical solutions.

### Solution A: RepOptimizer

I strongly recommend trying RepOptimizer if quantization is essential to your application. RepOptimizer directly trains a VGG-like model via Gradient Re-parameterization without any structural conversions. Quantizing a VGG-like model trained with RepOptimizer is as easy as quantizing a regular model. RepOptimizer has already been used in YOLOv6.

Paper: https://arxiv.org/abs/2205.15242

Code: https://github.com/DingXiaoH/RepOptimizers

Tutorial provided by the authors of YOLOv6: https://github.com/meituan/YOLOv6/blob/main/docs/tutorial_repopt.md. Great work! Many thanks!

### Solution B: custom quantization-aware training

Another choice is is to constrain the equivalent kernel (get_equivalent_kernel_bias() in repvgg.py) to be low-bit (e.g., make every param in {-127, -126, .., 126, 127} for int8), instead of constraining the params of every kernel separately for an ordinary model.

### Solution C: use the off-the-shelf toolboxes

(TODO: check and refactor the code of this example)

For the simplicity, we can also use the off-the-shelf quantization toolboxes to quantize RepVGG. We use the simple QAT (quantization-aware training) tool in torch.quantization as an example.

1. Given the base model converted into the inference-time structure. We insert BN after the converted 3x3 conv layers because QAT with torch.quantization requires BN. Specifically, we run the model on ImageNet training set and record the mean/std statistics and use them to initialize the BN layers, and initialize BN.gamma/beta accordingly so that the saved model has the same outputs as the inference-time model. 

```
python quantization/convert.py RepVGG-A0.pth RepVGG-A0_base.pth -a RepVGG-A0 
python quantization/insert_bn.py [imagenet-folder] RepVGG-A0_base.pth RepVGG-A0_withBN.pth -a RepVGG-A0 -b 32 -n 40000
```

2. Build the model, prepare it for QAT (torch.quantization.prepare_qat), and conduct QAT. This is only an example and the hyper-parameters may not be optimal.
```
python quantization/quant_qat_train.py [imagenet-folder] -j 32 --epochs 20 -b 256 --lr 1e-3 --weight-decay 4e-5 --base-weights RepVGG-A0_withBN.pth --tag quanttest
```


## FAQs

**Q**: Is the inference-time model's output the _same_ as the training-time model?

**A**: Yes. You can verify that by
```
python tools/verify.py
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

Finetuning with a converted RepVGG also makes sense if you insert a BN after each conv (please see the quantization example), but the performance may be slightly lower.

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


## An optional trick with a custom weight decay (deprecated)

This is deprecated. Please check ```repvggplus_custom_L2.py```. The intuition is to add regularization on the equivalent kernel. It may work in some cases.

The trained model can be downloaded at [Google Drive](https://drive.google.com/file/d/14I1jWU4rS4y0wdxm03SnEVP1Tx6GGfKu/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1qFGmgJ6Ir6W3wAcCBQb9-w?pwd=rvgg)

The training code should be changed like this:
```
        #   Build model and data loader as usual
        for samples, targets in enumerate(train_data_loader):
            #   ......
            outputs = model(samples)                        #   Your original code
            if type(outputs) is dict:                       
                #   A training-time RepVGGplus outputs a dict. The items are:
                    #   'main':     the output of the final layer
                    #   '*aux*':    the output of auxiliary classifiers
                    #   'L2':       the custom L2 regularization term
                loss = WEIGHT_DECAY * 0.5 * outputs['L2']
                for name, pred in outputs.items():
                    if name == 'L2':
                        pass
                    elif 'aux' in name:
                        loss += 0.1 * criterion(pred, targets)          #  Assume "criterion" is cross-entropy for classification
                    else:
                        loss += criterion(pred, targets)
            else:
                loss = criterion(outputs, targets)          #   Your original code
            #   Backward as usual
            #   ......
```



## Contact

**xiaohding@gmail.com** (The original Tsinghua mailbox dxh17@mails.tsinghua.edu.cn will expire in several months)

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

Homepage: https://dingxiaohan.xyz/

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. RepLKNet (CVPR 2022) **Powerful efficient architecture with very large kernels (31x31) and guidelines for using large kernels in model CNNs**\
[Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)\
[code](https://github.com/DingXiaoH/RepLKNet-pytorch).

2. **RepOptimizer** uses **Gradient Re-parameterization** to train powerful models efficiently. The training-time model is as simple as the inference-time. It also addresses the problem of quantization. **It has already been used in YOLOv6.** \
[Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/pdf/2205.15242.pdf)\
[code](https://github.com/DingXiaoH/RepOptimizers).

3. RepVGG (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **84.16%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

4. RepMLP (CVPR 2022) **MLP-style building block and Architecture**\
[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)\
[code](https://github.com/DingXiaoH/RepMLP).

5. ResRep (ICCV 2021) **State-of-the-art** channel pruning (Res50, 55\% FLOPs reduction, 76.15\% acc)\
[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)\
[code](https://github.com/DingXiaoH/ResRep).

6. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

7. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
