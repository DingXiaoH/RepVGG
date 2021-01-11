# RepVGG: Making VGG-style ConvNets Great Again

This is a super simple ConvNet architecture that achieves over 80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU! 

The training code and pretrained models will be updated upon the announcement of arxiv preprint (in two days).

Star me if you are interested.


# ImageNet training settings:

We trained for 120 epochs with cosine learning rate decay from 0.1 to 0. We used 8 GPUs, the global batch size was 256. Weight decay=1e-4, no weight decay on bias, bn.gamma and bn.beta. The data preprocssing is just the same as the PyTorch official example:
```
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])
```
            

# Use like this:
```
train_model = create_RepVGG_A0(deploy=False)
train train_model ...
deploy_model = repvgg_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
```
