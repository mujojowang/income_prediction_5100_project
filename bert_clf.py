import torch
import numpy as np
from transformers import BertTokenizer

from torch import nn
from transformers import BertModel

from torch.optim import Adam
from tqdm import tqdm

from file_util import load_file
from CONST import BERT_TRAIN_PATH, BERT_VAL_PATH, BERT_TEST_PATH
from bert_model import BertClassifier
from dataset_util import Dataset


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = {'0':0, '1':1}
np.random.seed(112)

# 30561
trainset = load_file(BERT_TRAIN_PATH, split_tag="\t")
train_data = {'text':trainset[0], 'category':trainset[1]}

# 1000
validset = load_file(BERT_VAL_PATH, split_tag="\t")
valid_data = {'text':validset[0], 'category':validset[1]}

def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data, labels, tokenizer), Dataset(val_data, labels, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch_num in range(epochs):
        total_acc_train = []
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
          train_label = train_label.to(device)
          mask = train_input['attention_mask'].to(device)
          input_id = train_input['input_ids'].squeeze(1).to(device)
          output = model(input_id, mask)
          batch_loss = criterion(output, train_label.long())
          total_loss_train += batch_loss.item()

          acc = (output.argmax(dim=1) == train_label).sum().item() / train_label.size(0)
          total_acc_train.append(acc)

          model.zero_grad()
          batch_loss.backward()
          optimizer.step()

        total_acc_val = []
        total_loss_val = 0
        with torch.no_grad():
          for val_input, val_label in val_dataloader:
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, val_label.long())
            total_loss_val += batch_loss.item()

            acc = (output.argmax(dim=1) == val_label).sum().item() / val_label.size(0)
            total_acc_val.append(acc)

        print("Epoch=", epoch_num)
        print("Train Acc=", sum(total_acc_train) / len(total_acc_train))
        print("Valid Acc=", sum(total_acc_val) / len(total_acc_val))

EPOCHS = 5
model = BertClassifier()
LR = 1e-6

train(model, train_data, valid_data, LR, EPOCHS)
