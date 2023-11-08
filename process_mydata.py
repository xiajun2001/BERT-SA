import random
import re

import pandas as pd
import torch

# str = "./data/aclImdb/十堰爆炸_抖音.xlsx"
str = "./data/aclImdb/恩施吊桥_抖音_run.xlsx"

def file_list(f_dir):
    labels = [];texts = []
    df = pd.read_excel(f_dir, sheet_name='Sheet1')
    raw_texts = list(df['text'])
    raw_labels = list(df['prediction'])
    for text, label in zip(raw_texts, raw_labels):
        cleaned_text = re.sub(r'[@#&$%^*!]', '', text)
        text_without_spaces = re.sub(r'\s', '', cleaned_text)
        if text_without_spaces:
            texts.append(text_without_spaces)
            if label == '积极':
                labels.append(0)
            elif label == '中立':
                labels.append(1)
            else:
                labels.append(2)
    return texts, labels

train_texts, train_labels = file_list(str)

random.seed(1)
idx = [i for i in range(len(train_texts))]
random.shuffle(idx)

x = []
y = []

for id in idx:
    x.append(train_texts[id])
    y.append(train_labels[id])

# TRAINSET_SIZE = 2048
# TESTSET_SIZE = 240
TRAINSET_SIZE = 7000
TESTSET_SIZE = 2000

test_samples = train_texts[-TESTSET_SIZE:]
test_labels = train_labels[-TESTSET_SIZE:]

train_samples = x[:TRAINSET_SIZE]
train_labels = y[:TRAINSET_SIZE]

