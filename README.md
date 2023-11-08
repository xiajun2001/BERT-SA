# BERT-SA
本项目基于预训练模型**distilbert-base-multilingual-cased-sentiments-student**，通过中文数据集进行迁移学习训练来对中文舆情进行情感分析。框架：pytorch。是初学者的初次练习...
## 数据集
数据集源自微博平台与抖音平台的评论信息，基于两个热点事件来对评论等信息进行爬取收集形成数据集。位于data/aclImdb下。原数据一共3W5条，但消极评论与中立评论远远大于积极评论。因此作特殊处理后，积极数据2601条，消极数据2367条，中立数据2725条，共7693条数据。由于代码中存在数据预处理部分，会剔除只含特殊字符，空格等无效信息，实际用到的数据可能小于7693。
## 预训练模型
使用的预训练模型为**distilbert-base-multilingual-cased-sentiments-student**，发布在huggingface平台上。模型文件放在data/Bert_related下，由于文件过大就没上传。
## Pytorch实现
代码参考了fnangle/text_classfication-with-bert-pytorch  
仓库链接：https://github.com/fnangle/text_classfication-with-bert-pytorch  
文件说明：
 - process_mydata.py 用于数据预处理，从excel文件中抽取数据并作数据清洗，并划分训练集与测试集。
 - Bert_mydata.py 为主函数
 - test_bert_mydata.py 用于测试模型

借助了huggingface的开源仓库[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers).
## 参数设置
BATCHSIZE = 8  
epoch = 4  
TRAINSET_SIZE = 7000  
TESTSET_SIZE = 2000  
调整参数lr
## Result
针对不同学习率，模型的accuracy如下。
|lr|accuracy  |
|--|--|
| 1e-3 | 83.8% |
|3e-4|84.6%|
|3e-5|88.3%|

通过修改epoch等参数或许可以得到更高的accuracy。
