import os
import pandas as pd
import numpy as np
import torch

from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader, RandomSampler
from tqdm import trange, tqdm
from torch.optim import AdamW
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
import argparse
import random

bert_model = './msmarco-distilbert-base-tas-b'  
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def data_read(bert_model, dataset_files, data_type, device):
    """
    read data from excel file
    :param data_type:train/test/dev
    :param dataset_files: the path of dataset
    :return: input_ids, labels, attention_masks
    """
    sentencses = []
    labels = []
    path = dataset_files[data_type]
    print("%s data loading------" % data_type)
    reader = pd.read_excel(path)
    for i in range(reader.shape[0]):
        column_name1 = reader.iloc[i][0].lower().strip()
        column_name2 = reader.iloc[i][1].lower().strip()
        label = reader.iloc[i][4]
        label = int(label)
        sent =  '[CLS] ' + column_name1 + ' : ' + reader.iloc[i][2].lower().strip() + ' [SEP] ' + column_name2 + " : " + reader.iloc[i][3].lower().strip() + ' [SEP] '

        sentencses.append(sent)
        labels.append(label)

    tokenizer = DistilBertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentencses]

    
    MAX_LEN = 512

    
    input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

    
    
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    
    
    input_ids = torch.tensor(input_ids).to(device)
    labels = torch.tensor(labels).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)

    return input_ids, labels, attention_masks


def data_load(bert_model, file_type, device, batch_size, dataset_files):
    """
    transform the data to the format that pytorch can recognize
    :param data_type:train/test/dev
    :param dataset_files: the path of dataset
    :param device: GPU 
    :param batch_size: the size of batch
    :return: DataLoader
    """
    inputs, labels, masks = data_read(bert_model, dataset_files, file_type, device)  

    
    data = TensorDataset(inputs, masks, labels)

    # sampler = RandomSampler(data)

    # Compute samples weight (each sample should get its own weight)
    class_sample_count = torch.tensor(
        [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def model_test(model,data,disable_bar:bool = False) -> float:
    input_ids, labels, attention_masks = data
    dataset = TensorDataset(input_ids, labels, attention_masks)

    TP = 0    # true positive label:1 predict:1
    TN = 0    # true negative label:0 predict:0
    FN = 0    # false negative label:1 predict:0
    FP = 0    # false positive label:0 predict:1
    
    for batch in tqdm(dataset, desc="step", disable=disable_bar):
        input_ids, labels, attention_masks = batch
        with torch.no_grad():
            logits = model(input_ids.unsqueeze(dim=0),attention_masks.unsqueeze(dim=0))[0]
        logits = logits.detach().cpu().numpy()
        # print(logits)

        if labels.item() == 0:
            if logits[0][0] > logits[0][1] : TN += 1
            else: FP += 1
        else :
            if logits[0][0] < logits[0][1] : TP += 1
            else: FN += 1
    if TP == 0 :
        recall,precision = 0,0
    else:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)

    print("recall :%.4f" % recall)
    print("precision :%.4f" % precision)
    
    if(precision + recall != 0):
        f1_score = 2*precision*recall/(recall+precision)
        print("f1-score :%.4f" % f1_score)
    else:
        f1_score = 0

    return recall,precision,f1_score

def train_classier(bert_model, opt, batch_size):
    """
    train the model on the dataset
    """
    epochs = opt.epoch
    device = opt.device 
    dataset_name = opt.dataset
    id = opt.dataset_id

    dataset_files = {
        'university':{
            'train': '../datasets/university/train_university.xlsx',
            'test': '../datasets/university/test_university.xlsx',
            'val': '../datasets/university/val_university.xlsx',
        },
        'mimic':{
            'train': '../datasets/mimic/train_mimic',
            'test': '../datasets/mimic/test_mimic',
            'val': '../datasets/mimic/val_mimic',
        },
        'cms':{
            'train': '../datasets/cms/train_cms',
            'test': '../datasets/cms/test_cms',
            'val': '../datasets/cms/val_cms',
        },
        'synthea':{
            'train': '../datasets/synthea/train_synthea',
            'test': '../datasets/synthea/test_synthea',
            'val': '../datasets/synthea/val_synthea',
        },
        'thalia':{
            'train': '../datasets/thalia/train_thalia0.xlsx',
            'test': '../datasets/thalia/test_thalia0.xlsx',
            'val': '../datasets/thalia/val_thalia0.xlsx',
        },
        'PO':{
            'train': '../datasets/PO/train_PO0.xlsx',
            'test': '../datasets/PO/test_PO0.xlsx',
            'val': '../datasets/PO/val_PO0.xlsx',
        },
        'harduni':{
            'train': '../datasets/harduni/train_harduni0.xlsx',
            'test': '../datasets/harduni/test_harduni0.xlsx',
            'val': '../datasets/harduni/val_harduni0.xlsx',
        },
    }

    filepath = dataset_files[dataset_name]
    if dataset_name in ['cms','mimic','synthea']:
        print("omop dataset load start---------")
        for key,value in filepath.items():
            filepath[key] = value + '{}.xlsx'.format(id)

    model_save_path = './saved/bert/{}-{}/'.format(dataset_name,id) 

    modelConfig = DistilBertConfig.from_pretrained(bert_model)
    modelConfig.num_labels = 2  
    model = DistilBertForSequenceClassification.from_pretrained(bert_model, config=modelConfig)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [  
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and p.requires_grad)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and p.requires_grad)],
        'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,
    )
        
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model)
    model.resize_token_embeddings(len(tokenizer))  

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.7)

    train_dataloader = data_load(bert_model, 'train', device, batch_size, filepath)
    val_dataset = data_read(bert_model, filepath, 'val', device)
    test_dataset = data_read(bert_model, filepath, 'test', device)
    model.to(device)
    
    log_info  = ""
    max_f1 = 0 

    for epoch in trange(epochs, desc="Epoch", disable=False):
        
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # for epoch in trange(epochs):
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iter", disable=False):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        scheduler.step()
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # validation
        model.eval()
        print('the current epoch is ',epoch)
        print('for val:')
        recall,precision,f1_score = model_test(model, val_dataset)
        print('for test:')
        model_test(model, test_dataset)
        if f1_score > max_f1:
            log_info = "recall:%.4f , precision:%.4f, f1_score:%.4f " % (recall,precision,f1_score)
            max_f1 = f1_score
            model.save_pretrained(model_save_path)

    print(log_info)
    print('\nThe final test:')
    model = DistilBertForSequenceClassification.from_pretrained(model_save_path, config=modelConfig)
    model.to(device)
    recall,precision,f1_score = model_test(model, test_dataset)

def main(opt):
    seed_everything(2021)
    batch_size = 8
    train_classier(bert_model, opt, batch_size)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='university', type=str, help='chemprot,imdb,yelp1,yelp2')
    parser.add_argument('--dataset_id', default=0, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda:7', type=str, help="cuda device")
    parser.add_argument('--test', action="store_true", help="whether to test the model")
    
    opt = parser.parse_args()
    print(opt)
    main(opt)

