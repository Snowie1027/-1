# 对比监督学习和自监督学习在图像分类任务上的性能表现
## 自监督学习
实现自监督学习SimCLR框架并使用该算法在自选的数据集CIFAR-10上训练ResNet-18, 
得到预训练模型权重及log文件保存在链接: https://pan.baidu.com/s/105crKhWoEWGQhoN127Xj2w?pwd=mbs2 提取码: mbs2,
随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测.

## baseline（监督学习）
在resnet18上进行cifar100的分类任务，实际训练及测试过程见Cifar100_resnet18.ipynb
runs文件夹中分别保存了3种数据增强方法及baseline方法的训练结果