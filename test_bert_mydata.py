import torch
from torch import nn

import Bert_mydata
import os
from transformers import AutoModelForSequenceClassification, AdamW,BertTokenizer,BertModel
 # 测试
if __name__=="__main__":
    # params_dir='./model/bert_base_model_mydata_beta.pkl'
    # params_dir='./model/bert_base_model_chinese_beta.pkl'
    params_dir='./model/distilbert-base-multilingual-cased-sentiments-student.pkl'

    path='./data/Bert_related/distilbert-base-multilingual-cased-sentiments-student/'
    model=AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
    # print(model)
    model.load_state_dict(torch.load(params_dir))
    Bert_mydata.test_model(model)
