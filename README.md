复赛有三点形式上的变化：
(1)采用docker镜像的方法

(2)采用数据流的方法进行预测，即每次预测一个样本，在预测结束之后，后台发送第2个预测样本...

(3)采用端到端的方法（吃了大亏，前期没有刷成绩，过早提交了端到端，没有日志，整个过程都是懵的，
最后显示超时，也没有日志，没有成绩，两个月白辛苦了）

但是也学到了很多东西。因此总结一下复赛过程中学到的知识点。

1、代码结构：
project

    |--README.md            # 解决方案及算法介绍文件，必选
    |--Dockerfile           # docker执行文件
    |--tcdata               #原始数据文件
    |--user_data            # 临时数据文件，生成的数据也都放里面
    |--src                  #代码文件
        |--train            # 训练代码文件夹示例，可自行组织
        |--test             # 预测代码文件夹示例，可自行组织
    |--run.sh               #整个端到端执行文件
    
    
    
2、整个代码是以docker镜像的方式打包的

1）阿里云docker开通以及简单操作教程：https://tianchi.aliyun.com/competition/entrance/531863/tab/253?spm=5176.12586973.0.0.469522c6c1I0Yr

2）最终提交的docker镜像：registry.cn-hangzhou.aliyuncs.com/zhengjiawei/tc_end2end2:2.3

可以直接拉取镜像执行代码

本地(或者服务器)安装docker之后(没有权限的，前面加上sudo，后面的也一样)

nvidia-docker run --rm -it registry.cn-hangzhou.aliyuncs.com/zhengjiawei/tc_end2end2:2.3

想要进入docker内部查看执行日志

docker ps -a 查看容器id

docker exec -it 容器id bash

进入之后在run.sh执行 pretrain的时候，就会生成nohup文件，可以进入内部查看日志内容

3)想要拉取代码然后实现的，需要下载比赛数据到tc_data和下载nezha-cn-base到user_data

数据链接: https://pan.baidu.com/s/1pXq9VJAscubOC4Wxk2Effg  密码: 17bt

nezha-cn-base链接：https://github.com/lonePatient/NeZha_Chinese_PyTorch



3、代码部分
首先整个过程待用的是三个nezha(ngram mask策略下)预训练和finetune之后进行模型融合
（代码完全相同，只不过随机数种子不同）

相当于从三个角度对原始数据进行预训练和finetune,学习到不同角度的内容

没有尝试4个原因是，比赛要求限制推理时间15分钟，4个可能会超时

看到其他选手有尝试两个nezha，两个bert


4、关于推理加速
刚开始采用了turbo_transformers的方法对bert加速，但是没办法对nezha加速，并且加速效果没有onnx效果好，所有最后是用了onnx方法
onnx加速推理使用教程：
https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb

1）首先根据模型参数生成onnx文件

2）根据生成onnx文件进行推理
加载一个模型预测一条数据的平均时间为0.015s
三个模型推理5w条数据时间在12分钟左右


5、主要提点技术

本方案中主要有四个提分点：

1）在对抗训练中发现pgd比fgm时间要长，但是效果也更好（需要调参数）

微调部分

25分钟一个epoch FGM

50分钟一个epoch PGD

2）采用混合精度计算(加快预训练速度)和预测了原始数据和对偶数据
假设需要推理的数据(数据源发送的数据)数据A,B,则分别预测(A,B)和(B,A),进行加权平均

3）采用了模型融合的方法，将三种模型的结果按照不同比例进行了加权平均，从而得到最终结果

4)finetune阶段采用了MSD(Multi-Sample Dropout):改进后的结果是加快了训练收敛速度
和提高了泛化能力。
学习链接：https://zhuanlan.zhihu.com/p/362068428                                         









