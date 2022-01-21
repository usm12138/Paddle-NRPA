# 基于Paddle实现《NRPA: Neural Recommendation with Personalized Attention》
## 模型简介
该模型包含三个部分：User-Net、Item-Net和评分预测模块，User-Net和Item-Net具有的组成，分别用来学习user和item的嵌入，评分预测模块根据user embedding和item embedding预测user对item 的评分。

## 环境
+ python==3.8
+ nltk==3.6.7  
+ paddlepaddle-gpu==2.2.1
+ bs4
+ pickle
+ contractions
+ numpy
## 数据准备
原始数据Amazon：Electronics 从官网下载，已下载并上传云盘，可以在[此处下载](https://pan.baidu.com/s/1f0cdwevw1JMssBEOyChXCA?pwd=7s2i)，放置在data文件夹下
然后运行以下命令进行数据预处理
```
python data_preprocess.py
```
处理过程有点慢，可能需要几十分钟。
可以直接下载使用已经预处理好的数据，已处理数据可点击[此处下载](https://pan.baidu.com/s/1WPwlA7okKKmzsnZzzWv-yQ?pwd=82rz)，解压后也放在data文件夹下
```
/data
|--Electronics_5.json # 原始未处理数据，可以不用，直接使用下面两个已处理好的数据
|--data.pkl # 已经处理好的数据, 可以直接使用 
|--vocab.pkl #生成的字典
```
## 训练及测试
训练时，运行main.py 
```
python main.py
```
部分训练日志如下所示：
```
step: 1440 train mse_loss: 0.94309, time: 12.4570 s
step: 1480 train mse_loss: 0.93621, time: 12.6679 s
step: 1520 train mse_loss: 0.94401, time: 15.5520 s
step: 1560 train mse_loss: 0.96166, time: 13.3231 s
step: 1600 train mse_loss: 0.95926, time: 12.7999 s
Validing and Testing...............
valid mse_loss: 1.02980,  test mse_loss: 1.03738, time: 115.1671 s
```
## 复现效果
论文结果： Amazon-Electronics: MSE 1.047
复现结果： 1.037

## 参考
* https://medium.com/geekculture/data-preprocessing-and-eda-for-natural-language-processing-56e45c1df36d
* https://github.com/TianHongTao/ID-DAML
