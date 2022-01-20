# 20211101_174348

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# 1에폭 0.9927

import torch.nn as nn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
from transformers import BertTokenizer, RobertaForSequenceClassification

# from torch.optim import Adam
import torch.nn.functional as F
import time
import os
import datetime
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
import random  # 재현을 위해 랜덤시드 고정
import tqdm

#  # CUDA_VISIBLE_DEVICES=1,2,3 python training_server_2.py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # Uses GPU 0.

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Uses GPU 0.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print('Available device : ', torch.cuda.device_count())
print('Current cuda device :', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# Setting parameters
batch_size = 22
num_label = 2
learning_rate = 1e-5
max_grad_norm = 1
epochs = 10
# max_len = 512
num_workers = 8


start_time = time.time()

from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification
import torch

config = 'klue/roberta-large'
tokenizer = BertTokenizer.from_pretrained(config)
model = RobertaForSequenceClassification.from_pretrained(config, num_labels=num_label)



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# criterion = torch.nn.CrossEntropyLoss()
model = nn.DataParallel(model)
model.to(device)  # model.cuda()
# model = Model(bert.roberta).to(device)


train_path = './chitchat_train_dataset.csv'
valid_path = './chitchat_valid_dataset.csv'
# train_path ='../experiment_dataset/chitchat_train_dataset.csv'
# valid_path = '../experiment_dataset/chitchat_valid_dataset.csv'

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
# train_df = train_df[:50]
# valid_df = valid_df[:1]


class Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        #         print(text)

        label = self.df.iloc[idx, 1]
        #         print(label)
        return text, label


train_dataset = Dataset(train_df)
valid_dataset = Dataset(valid_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

work_date = datetime.today().strftime("%Y%m%d_%H%M%S")
print(work_date)
model_save_dir = './saved_model/' + work_date  # ./saved_model/2021712)
print(model_save_dir)




import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
createFolder(model_save_dir)


def train(epoch):
    model.train()
    total_loss = 0
    total_len = 0
    total_correct = 0
    itr = 1
    p_itr = 1000
    # p_itr = 1

    for text, label in train_loader:
        optimizer.zero_grad()
        #     if len(text) > 512:
        #         print(text)
        encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=512, truncation=True) for t in text]

        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]

        sample = torch.tensor(padded_list)

        sample, label = sample.to(device), label.to(device)

        labels = label.clone().detach()

        outputs = model(sample, labels=labels)

        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

        correct = pred.eq(labels)

        total_correct += correct.sum().item()

        total_len += len(labels)  # 배치사이즈

        loss = outputs[0].mean()
        total_loss += loss.item()

        loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        #     scheduler.step()

        if itr % p_itr == 0:  # p_itr=2이면 1에폭(itr)안에서 배치사이즈만큼 2번돌때 출력
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.4f}'.format \
                      (epoch + 1, epochs, itr, total_loss / p_itr, total_correct / total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0
        itr += 1

    model_epoch = 'epoch_{}.pt'.format(epoch + 1)
    #         print(model_epoch)
    final_save_dir = os.path.join(model_save_dir, model_epoch)
    #         print(final_save_dir)
    #         os.mkdir(final_save_dir)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, final_save_dir)

    print("=> {}/{} saving checkpoint".format(epoch + 1, epochs))


def test():
    model.eval()

    total_loss = 0
    total_len = 0
    total_correct = 0

    for text, label in valid_loader:
        # print("Running Validation...")

        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]

        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = label.clone().detach()

        with torch.no_grad():
            outputs = model(sample, labels=labels)
        _, logits = outputs
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)

    print('Test accuracy: {:.4f}'.format(total_correct / total_len))


# main
start_epoch = 0
for epoch in range(start_epoch, epochs):
    train(epoch)

    test()
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

train_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
print(train_date)

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

print("Training complete!")


"""
(pytorch1.7) yeji@abrlab:~/chitchat2$ CUDA_VISIBLE_DEVICES=1,2 python training.py
2021-11-26 12:09:24.504798: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
cuda
Available device :  2
Current cuda device : 0
Quadro GV100
Some weights of the model checkpoint at klue/roberta-large were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at klue/roberta-large and are newly initialized: ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
20211126_121016
./saved_model/20211126_121016
/abr/yeji/anaconda3/envs/pytorch1.7/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:64: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
[Epoch 1/10] Iteration 1000 -> Train Loss: 0.0298, Accuracy: 0.9865
[Epoch 1/10] Iteration 2000 -> Train Loss: 0.0046, Accuracy: 0.9990
[Epoch 1/10] Iteration 3000 -> Train Loss: 0.0007, Accuracy: 0.9998
[Epoch 1/10] Iteration 4000 -> Train Loss: 0.0041, Accuracy: 0.9992
[Epoch 1/10] Iteration 5000 -> Train Loss: 0.0016, Accuracy: 0.9996
[Epoch 1/10] Iteration 6000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 1/10] Iteration 7000 -> Train Loss: 0.0006, Accuracy: 0.9999
[Epoch 1/10] Iteration 8000 -> Train Loss: 0.0043, Accuracy: 0.9990
[Epoch 1/10] Iteration 9000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 1/10] Iteration 10000 -> Train Loss: 0.0005, Accuracy: 1.0000
[Epoch 1/10] Iteration 11000 -> Train Loss: 0.0070, Accuracy: 0.9978
[Epoch 1/10] Iteration 12000 -> Train Loss: 0.0010, Accuracy: 0.9999
[Epoch 1/10] Iteration 13000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 1/10] Iteration 14000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 1/10] Iteration 15000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 1/10] Iteration 16000 -> Train Loss: 0.0000, Accuracy: 1.0000
=> 1/10 saving checkpoint
Test accuracy: 0.9927
11 hours 43 mins 34 secs for training
[Epoch 2/10] Iteration 1000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 2000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 3000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 4000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 5000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 6000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 7000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 8000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 9000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 10000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 11000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 12000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 13000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 14000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 15000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 2/10] Iteration 16000 -> Train Loss: 0.0000, Accuracy: 1.0000
=> 2/10 saving checkpoint
Test accuracy: 0.9927
23 hours 25 mins 29 secs for training
[Epoch 3/10] Iteration 1000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 3/10] Iteration 2000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 3/10] Iteration 3000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 3/10] Iteration 4000 -> Train Loss: 0.0000, Accuracy: 1.0000
[Epoch 3/10] Iteration 5000 -> Train Loss: 0.0000, Accuracy: 1.0000

"""
