## AutoEncoder with Pytorch

This project is an implementation of auto-encoder with MNIST dataset and `pytorch = 1.4.0`

Learn tensorboard by the way

#### Instruction to Prepare MNIST dataset

```bash
bash ./script/prepare.sh
```

#### Instruction to Train Model

```bash
bash ./script/train.sh ${model_name}
```

`${model_name}` can be one of `conv/linear/vae`...

#### Instruction to use tensorboard visulize

```bash
tensorboard --logdir=runs --bind_all
```
##### Trainning Loss
![](http://leiblog.wang/static/image/2020/5/RrBF3Y.png)
##### Decode Performance
![](http://leiblog.wang/static/image/2020/5/9fiqeI.png)
##### Model Architecture
![](http://leiblog.wang/static/image/2020/5/1svLUs.png)
#### have to be aware of these information:

The output shape of `ConvTranspose2d` can be computed by formula:

![](http://leiblog.wang/static/image/2020/5/iF2yma.png)

### Reference

Hung-yi Lee MachineLearning Spring2020 Course (https://www.bilibili.com/video/BV1JE411g7XF?p=59)

https://github.com/L1aoXingyu/pytorch-beginner (Model from here, But train code with this repositories have a bug in transform operation, detail in [this pr page](https://github.com/L1aoXingyu/pytorch-beginner/pull/36))
