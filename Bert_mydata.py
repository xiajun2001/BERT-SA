import os

from torch.optim.lr_scheduler import ExponentialLR

import process_mydata
import torch
import torch.nn as nn
import logging
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

# 使用BERT使其向量化

MAXLEN = 512 - 2
BATCHSIZE = 8
num_classes = 3
epoch = 4
lr = 3e-5


from transformers import AutoModelForSequenceClassification, AdamW, BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

# path = './data/Bert_related/bert_base_uncased/'
# path = './data/Bert_related/bert_base_chinese/'
path = './data/Bert_related/distilbert-base-multilingual-cased-sentiments-student/'
config_dir = path
tokenizer = BertTokenizer.from_pretrained(path)


def convert_text_to_ids(tokenizer, sentence, limit_size=MAXLEN):
    t = tokenizer.tokenize(sentence)[:limit_size]
    encoded_ids = tokenizer.encode(t)
    if len(encoded_ids) < limit_size + 2:
        tmp = [0] * (limit_size + 2 - len(encoded_ids))
        encoded_ids.extend(tmp)
    return encoded_ids


'''构建数据集和迭代器'''


input_ids = [convert_text_to_ids(tokenizer, sen) for sen in process_mydata.train_samples]
# input_labels = process_imdb.get_onehot_labels(process_imdb.train_labels)
input_labels = torch.unsqueeze(torch.tensor(process_mydata.train_labels), dim=1)


def get_att_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


atten_token_train = get_att_masks(input_ids)

'''构建数据集和数据迭代器，设定 batch_size 大小为'''

train_set = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(atten_token_train),
                          torch.LongTensor(input_labels))
train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCHSIZE,
                          shuffle=True
                          )

for i, (train, mask, label) in enumerate(train_loader):
    print(train.shape, mask.shape, label.shape)  ##torch.Size([8,512]) torch.Size([8,512]) torch.Size([8, 1])
    break

input_ids2 = [convert_text_to_ids(tokenizer, sen) for sen in process_mydata.test_samples]
input_labels2 = torch.unsqueeze(torch.tensor(process_mydata.test_labels), dim=1)
atten_tokens_eval = get_att_masks(input_ids2)
test_set = TensorDataset(torch.LongTensor(input_ids2), torch.LongTensor(atten_tokens_eval),
                         torch.LongTensor(input_labels2))
test_loader = DataLoader(dataset=test_set,
                         batch_size=BATCHSIZE, )

for i, (train, mask, label) in enumerate(test_loader):
    print(train.shape, mask.shape, label.shape)  #
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''预测函数，用于预测结果'''

def predict(logits):
    res = torch.argmax(logits, dim=1)  # 按行取每行最大的列下标
    return res


'''训练'''

def train_model(net, epoch=epoch):
    avg_loss = []
    net.train()  # 将模型设置为训练模式
    net.to(device)

    # optimizer = AdamW(net.parameters(), lr=5e-5)
    optimizer = AdamW(net.parameters(), lr=lr)
    gamma = 0.9  # 衰减率
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    accumulation_steps = 8
    writer = SummaryWriter(log_dir='logs/train')
    for e in range(epoch):
        for batch_idx, (data, mask, target) in enumerate(train_loader):
            # optimizer.zero_grad()
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            output = net(data, attention_mask=mask, labels=target)
            # output = net(data, token_type_ids=None, attention_mask=mask, labels=target)
            # logit是正负概率
            loss,logits=output[0],output[1]
            # L2
            # l2_reg = torch.tensor(0., device=device)
            # for param in net.parameters():
            #     l2_reg += torch.norm(param, 2)
            # loss = loss + 0.0001 * l2_reg
            loss = loss / accumulation_steps  # 梯度积累
            avg_loss.append(loss.item())
            loss.backward()

            if ((batch_idx + 1) % accumulation_steps) == 0:
                # 每 8 次更新一下网络中的参数
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if batch_idx % 5 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    e + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), np.array(avg_loss).mean()
                ))
                writer.add_scalar('loss', float(loss), batch_idx)
    print('Finished Training')
    writer.close()
    return net


def test_model(net):
    net.eval()
    net = net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        writer = SummaryWriter(log_dir='logs/test')
        for batch_idx, (data, mask, label) in enumerate(test_loader):
            # logging.info("test batch_id=" + str(batch_idx))

            data, mask, label = data.to(device), mask.to(device), label.to(device)
            # output = net(data, token_type_ids=None, attention_mask=mask)  # 调用model模型时不传入label值。
            output = net(data, attention_mask=mask)  # 调用model模型时不传入label值。
            # output的形式为（元组类型，第0个元素是每个batch中好评和差评的概率）
            # print(output[0],label)
            # print(predict(output[0]), label.flatten())
            total += label.size(0)  # 逐次按batch递增
            correct += (predict(output[0]) == label.flatten()).sum().item()
            if batch_idx%50==0:
                # print('start')
                accuracy = 100.*correct/total
                formatted_accuracy = "{:.3f}".format(accuracy)
                formatted_accuracy = float(formatted_accuracy)
                writer.add_scalar('accuracy', formatted_accuracy, batch_idx/50)
                logging.info('==============共{}轮,第{}轮执行,每50轮打印一次,当前正确率为{:.3f}%=============='.format(len(test_loader), batch_idx+1, 100.*correct/total))
        print(f"正确分类的样本数 {correct}，总数 {total},准确率 {100.*correct/total:.3f}%")
        writer.close()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    #  当使用预训练模型做迁移学习时，num_labels的指定非常重要
    pre_net = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_classes)
    for param in pre_net.parameters():
        param.requires_grad = False

    classifier = nn.Linear(pre_net.classifier.in_features, num_classes)
    pre_net.classifier = classifier
    # params_dir = './model/bert_base_model_mydata_beta.pkl'
    params_dir = './model/distilbert-base-multilingual-cased-sentiments-student.pkl'
    # params_dir = './model/bert-base-multilingual-uncased-sentiment.pkl'

    model = train_model(pre_net, epoch=epoch)
    os.makedirs(os.path.dirname(params_dir), exist_ok=True)
    torch.save(model.state_dict(), params_dir)  # 保存模型参数
    # os.system("/usr/bin/shutdown")


