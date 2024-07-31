import torch
import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,WeightedRandomSampler
from sentence_transformers import InputExample

data_path = '../datasets/'

def read_table(prefix_path,target_file):
    origin_dataset = ['cms','mimic','synthea']

    if(target_file in origin_dataset):
        path = prefix_path + target_file + '/omop_' + target_file + '_data.xlsx'
        books = pd.read_excel(path)
    elif(target_file in ['PO','harduni']):
        path = prefix_path + target_file + '/' + target_file + '.csv'
        books = pd.read_csv(path)
    else:
        path = prefix_path + target_file + '/' + target_file + '.xlsx'
        books = pd.read_excel(path)
    return books

def batch_to_device(batch, target_device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def data_read(dataset_name,data_type,with_sent:bool = True,
              id:int = 2) -> tuple:
    sentencses1 = []
    sentencses2 = []
    labels = []
    prepath = data_path + dataset_name + '/' + data_type + '_' + dataset_name
    path = prepath +'{}.xlsx'.format(id)

    print(dataset_name + " " + data_type +" data reading------")
    reader = pd.read_excel(path)
    for i in range(reader.shape[0]):
        column_name1 = reader.iloc[i][0].lower().strip()
        column_name2 = reader.iloc[i][1].lower().strip()
        label = reader.iloc[i][4]

        sent1 =  '[CLS] ' + column_name1 + ' [SEP] '   
        sent2 =  '[CLS] ' + column_name2 + ' [SEP] ' 

        if with_sent:
            text_raw1 = reader.iloc[i][2].lower().strip()
            text_raw2 = reader.iloc[i][3].lower().strip()
            sent1 += text_raw1
            sent2 += text_raw2
        sentencses1.append(sent1)
        sentencses2.append(sent2)
        try:
            labels.append(int(label))
        except:
            print(dataset_name,data_type,i)
    
    return sentencses1,sentencses2,labels


def data_load(dataset_name,data_type,with_sent:bool = True,
              id:int = 2) -> DataLoader:
    sentencses1,sentencses2,labels = data_read(dataset_name,data_type,with_sent,id)
    train_examples = [InputExample(texts=[sentencses1[i],sentencses2[i]],label=int(labels[i]))  for i in range(len(labels))]
    labels = torch.tensor(labels)

    # Compute samples weight (each sample should get its own weight)
    class_sample_count = torch.tensor(
        [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dataloader = DataLoader(train_examples, sampler=sampler, batch_size=16)

    return dataloader


def plot_res(title, x_data, y_data):
    plt.cla()

    plt.plot(x_data, y_data)
    plt.title(title)
    plt.savefig(title)
    plt.show()