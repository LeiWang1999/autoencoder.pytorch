## AutoEncoder with Pytorch

This project is an implementation of auto-encoder with MNIST dataset and `pytorch = 1.4.0`

And, Learn tensorboard by the way.

Please feel free to pr :)

**have to be aware of this**:

The output shape of `ConvTranspose2d` can be computed by formula:

![](http://leiblog.wang/static/image/2020/5/iF2yma.png)

### Reference

Hung-yi Lee MachineLearning Spring2020 Course (https://www.bilibili.com/video/BV1JE411g7XF?p=59)

https://github.com/L1aoXingyu/pytorch-beginner (Model from here, But train code with this repositories have a bug in transform operation, detail in [this pr page](https://github.com/L1aoXingyu/pytorch-beginner/pull/36))
