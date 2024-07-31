## table_embedding module: add the information from the table that the attribute belongs
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader,WeightedRandomSampler,SequentialSampler
from sentence_transformers import SentenceTransformer,InputExample
from tqdm import trange,tqdm
from transformers import AutoTokenizer, DebertaV2Tokenizer
from math import sqrt
from torch.utils.data import DataLoader,Dataset

import my_distilbert
from TwinBERT import CompositeModel, Multi_CrossAttention
from utils import pad_and_trancate,cos_sim,flatten_list,get_table_embdi,MAX_STATE_NUM,DISTILBERT_PATH

sql_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)

class MyDataset(Dataset):
    def __init__(self,data_path,opt):
        with open(data_path, "rb") as f:
            datas = json.load(f)
        self.mydatas = []

        labels = []
        for data in datas:
            if data_path in ["data/harduni/train1.json","data/harduni/test1.json"]:
                print("university dataset")
                sent1 =  data["databaseA"] + ' [SEP] ' + data["descriptionA"]
                sent2 =  data["databaseB"] + ' [SEP] ' + data["descriptionB"]

            else:
                sent1 =  data["omop"] + ' [SEP] ' + data["des1"]
                sent2 =  data["table"] + ' [SEP] ' + data["des2"]
            mydata = {
                    'cnames1':data["omop"],
                    'cnames2':data["table"],
                    'sentences1':sent1,
                    'sentences2':sent2,
                    'query1': data["queryA"].replace('"','').replace('\'',''),
                    'query2': data["queryB"].replace('"','').replace('\'',''),
                    'label' : torch.tensor(int(data["label"]))
                }
            if opt.has_state:
                
                temp1 = pad_and_trancate(torch.tensor(data["stateA"]) , 512, 0) 
                state1 = F.one_hot(temp1 , num_classes = MAX_STATE_NUM)
                temp2 = pad_and_trancate(torch.tensor(data["stateB"]) , 512, 0) 
                state2 = F.one_hot(temp2 , num_classes = MAX_STATE_NUM)

                mydata['state1'] = state1
                mydata['state2'] = state2
                
            labels.append(int(data["label"]))
            self.mydatas.append(mydata)

        
        labels = torch.tensor(labels)
        class_sample_count = torch.tensor(
            [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
        weight = 1. / class_sample_count.float()
        self.samples_weight = torch.tensor([weight[t] for t in labels])
    
    def get_sample_weight(self):
        return self.samples_weight

    def __getitem__(self, index):
        return self.mydatas[index]

    def __len__(self):
        return len(self.mydatas)


def data_load(data_path,batch_size,opt,has_weight=True) -> DataLoader:
    TrainDataset = MyDataset(data_path,opt)
    if has_weight:
        samples_weight = TrainDataset.get_sample_weight()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        dataloader = DataLoader(TrainDataset, sampler=sampler, batch_size=batch_size)
    else:
        # print("SequentialSampler")
        sampler = SequentialSampler(TrainDataset)
        dataloader = DataLoader(TrainDataset,sampler=sampler,shuffle=False, batch_size=batch_size)
    return dataloader



def evaluate(model,test_loader,opt,bert_tokenizer,disable_bar = False,test = False) :
    dataset_name = opt.dataset
    dataset_id = opt.dataset_id
    device = opt.device
    alpha  = opt.alpha

    omop_table_embdi,target_table_embdi = get_table_embdi(dataset_name,dataset_id=dataset_id)

    all_y = []
    all_probs = []
    for data in tqdm(test_loader, desc="step", disable=disable_bar):
        sent1 = bert_tokenizer(data['sentences1'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        query1 = sql_tokenizer(data['query1'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        sent2 = bert_tokenizer(data['sentences2'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        query2 = sql_tokenizer(data['query2'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        with torch.no_grad():
            table_emb = [omop_table_embdi[item.split('-')[0]][:,0,:] for item in data['cnames1']]
            target_emb = [target_table_embdi[item.split('-')[0]][:,0,:] for item in data['cnames2']] 

            # print(table_names,target_names)
            tableembeddings = [
                torch.stack(table_emb).to(device),
                torch.stack(target_emb).to(device)
                ]
            
            if opt.has_state :
                embeddingA = model.table_embdi(sent1,query1,data['state1'].to(device),alpha*tableembeddings[0])
                embeddingB = model.table_embdi(sent2,query2,data['state2'].to(device),alpha*tableembeddings[1])
                del data['state1'],data['state2']
            else:
                embeddingA = model.table_embdi(sent1,query1,alpha*tableembeddings[0])
                embeddingB = model.table_embdi(sent2,query2,alpha*tableembeddings[1])

            # print(embeddingA,tableembeddings[0])
            cos_sim = torch.cosine_similarity(
                embeddingA ,
                embeddingB 
                )
            all_y.append(data['label'])
            all_probs.append(cos_sim)
            # print(all_y,all_probs)
            del sent1,query1,sent2,query2
            torch.cuda.empty_cache()

    # print(all_probs)
    all_y = flatten_list(all_y)
    all_probs = flatten_list(all_probs)

    best_th = 0.5
    f1 = 0.0 # metrics.f1_score(all_y, all_p)
    best_info = ""

    # print(all_probs)
    max_th = max(all_probs)
    for th in np.arange(0.2, max_th, 0.04):
        pred = [1 if p > th else 0 for p in all_probs]
        precision,recall,new_f1,_ = metrics.precision_recall_fscore_support(all_y, pred, labels=[1])
        info = "recall:%.4f, precision:%.4f, f1:%.4f" % (recall,precision,new_f1)
        # print("\t\t\t",info)
        if new_f1 > f1:
            f1 = new_f1
            best_th = th
            best_info = info
    print(best_info)
    if test:
        indics = []
        for i in range(len(all_probs)):
            if all_probs[i] > best_th and all_y[i] == 1:
                indics.append(i)
                TT_data = test_loader.dataset[i]
                print(TT_data.get('sentences1'),TT_data.get('query1'))
                print(TT_data.get('sentences2'),TT_data.get('query2'))
            
        print(indics)
    
    return f1, best_th, best_info


def sdata_read(data_path) -> tuple:
    with open(data_path, "rb") as f:
        datas = json.load(f)
    sentences1 = []
    sentences2 = []
    cnames1 = []
    cnames2 = []
    labels = []
    for data in datas:
        if data_path in ["data/harduni/train1.json","data/harduni/test1.json","data/harduni/val1.json"]:
            print("university dataset")
            sent1 =  data["databaseA"] + ' [SEP] ' + data["descriptionA"]
            sent2 =  data["databaseB"] + ' [SEP] ' + data["descriptionB"]
            column_name1 = data["databaseA"]
            column_name2 = data["databaseB"]
        else:
            sent1 =  data["omop"] + ' [SEP] ' + data["des1"]
            sent2 =  data["table"] + ' [SEP] ' + data["des2"]
            column_name1 = data["omop"]
            column_name2 = data["table"]

        label = int(data["label"])

        sentences1.append(sent1)
        sentences2.append(sent2)
        cnames1.append(column_name1)
        cnames2.append(column_name2)
        labels.append(label)
    
    return cnames1,cnames2,sentences1,sentences2,labels

def sdata_load(data_path,batch_size=16,has_weight=True) -> DataLoader:
    sentences1,sentences2,labels = sdata_read(data_path)
    train_examples = [InputExample(texts=[sentences1[i],sentences2[i]],label=labels[i])  for i in range(len(labels))]
    if has_weight:
        labels = torch.tensor(labels)
        # Compute samples weight (each sample should get its own weight)
        class_sample_count = torch.tensor(
            [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        dataloader = DataLoader(train_examples, sampler=sampler, batch_size=batch_size)
    else:
        sampler = SequentialSampler(train_examples)
        dataloader = DataLoader(train_examples,sampler=sampler,shuffle=False, batch_size=batch_size)
    return dataloader

def sevaluate(model:SentenceTransformer,data,
              omop_table_embdi,target_table_embdi,
              alpha,disable_bar:bool = False) -> float:
    column_name1,column_name2,sentences1,sentences2,labels = data

    all_y = []
    all_probs = []

    for i in trange(len(labels), desc="step", disable=disable_bar):
        embeddings = [torch.tensor(model.encode(sentences1[i])),
                      torch.tensor(model.encode(sentences2[i]))]
        # print(embeddings[0]) ## [list] -> 768
        table_name = column_name1[i].split('-')[0]
        target_name = column_name2[i].split('-')[0]
        # print(table_name,target_name)
        tableembeddings = [
            omop_table_embdi[table_name][:,0,:],
            target_table_embdi[target_name][:,0,:]
            ]
        # print(tableembeddings[0][0])
        ## tensor [[768]]
        
        output = cos_sim(embeddings[0] + alpha*tableembeddings[0][0],
                        embeddings[1] + alpha*tableembeddings[1][0])
        all_y.append(labels[i])
        all_probs.append(output)

    best_th = 0.5
    f1 = 0.0 # metrics.f1_score(all_y, all_p)
    best_info = ""

    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in all_probs]
        precision,recall,new_f1,_ = metrics.precision_recall_fscore_support(all_y, pred, labels=[1])
        if new_f1 > f1:
            f1 = new_f1
            best_th = th
            best_info = "recall:%.4f, precision:%.4f, f1:%.4f" % (recall,precision,f1)
    print(best_info)

    return f1, best_th, best_info


def DotMultiply(q,k,v,attention_mask):
    
    batch_size= q.size(0)
    num_heads = q.size(1)
    seq_len   = q.size(2)
    h_size    = q.size(3)

    attention_mask = attention_mask.eq(0)
    if num_heads > 1:
        attention_mask = torch.unsqueeze(attention_mask, 1).expand(-1, num_heads, -1) 
        attention_mask = torch.unsqueeze(attention_mask, -1).expand(-1, -1, -1, seq_len) 

    scores = torch.matmul(q,torch.transpose(k, -1, -2))  ##scores.shape -> torch.Size([32, 8, 512, 512])
    # use mask
    scores = scores.masked_fill(attention_mask, -1e9)
    weights = torch.softmax(scores / sqrt(q.size(-1)), dim=-1)
    # weights = self.dropout(weights)

    attention = torch.matmul(weights,v)

    # attention : [batch_size , seq_length , num_heads * h_size]
    attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * h_size)
    return attention


def tableembdding(model:my_distilbert.DistilBertModel,tokenizer,table_path,device):
    """
    使用表中所有属性的embedding来表示这个表
    使用model将属性embedding生成，并保存到一个字典中，key(表名),value(embedding)
    """
    df = pd.read_excel(table_path,sheet_name=0)

    
    k_dict = {}
    v_dict = {}
    des_dict = {}

    
    for index, row in df.iterrows():
        with torch.inference_mode():
            all_name = row['table']
            attr_des = row['d2']

            parts = all_name.split('-')
            if len(parts) == 2:
                table_name = parts[0]
                
                model_input = parts[1] + ' [SEP] ' + attr_des
                inputs = tokenizer(model_input, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
                outputs = model(**inputs,output_attentions=True)
                k = outputs['attentions'][-1][1]
                v = outputs['attentions'][-1][2]
                
                if table_name in k_dict:
                    k_dict[table_name].append(k.to('cpu'))
                    v_dict[table_name].append(v.to('cpu'))
                else:
                    k_dict[table_name] = [k.to('cpu')]
                    v_dict[table_name] = [v.to('cpu')]
                    des_dict[table_name] = row['d1']
                del inputs,outputs,k,v
                torch.cuda.empty_cache()
    
    
    avg_k = {}
    for key,value in k_dict.items():
        stacked_tensor = torch.stack(value)
        avg_k[key] = torch.sum(stacked_tensor, dim=0) / len(value)
    
    avg_v = {}
    for key,value in v_dict.items():
        stacked_tensor = torch.stack(value)
        avg_v[key] = torch.sum(stacked_tensor, dim=0) / len(value)

    table_embdi = {}
    
    for table_name in avg_k.keys():
        model_input = table_name + ' [SEP] ' + des_dict[table_name]
        inputs = tokenizer(model_input, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        attention_mask = inputs["attention_mask"]

        outputs = model(**inputs,output_attentions=True)
        q = outputs['attentions'][-1][0].to('cpu')
        value = DotMultiply(q,avg_k[table_name],avg_v[table_name],attention_mask.to('cpu'))
        table_embdi[table_name] = value

    return table_embdi
 
def generate_embdi(model_name,datasets_name,dataset_id,device):
    
    model = my_distilbert.DistilBertModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained('msmarco-distilbert-base-tas-b/')

    target_dict = tableembdding(model,tokenizer,'../../sbert/output/target_{}.xlsx'.format(datasets_name),device)
    with open('embdi/{}-{}.pickle'.format(datasets_name,dataset_id), 'wb') as file:
        pickle.dump(target_dict, file)

    omop_dict = tableembdding(model,tokenizer,'../../sbert/output/test_{}.xlsx'.format(datasets_name),device)
    with open('embdi/omop_{}-{}.pickle'.format(datasets_name,dataset_id), 'wb') as file:
        pickle.dump(omop_dict, file)

def sbert_embdi(opt):
    device = opt.device
    dataset_name = opt.dataset
    dataset_id = opt.dataset_id
    
    model_name = './saved/sentencebert/{}-model-cos-mean/{}-distilbert'.format(dataset_name,dataset_id)
    final_model = SentenceTransformer(model_name,device = device)
    final_model.to(final_model._target_device)
        
    
    omop_table_embdi,target_table_embdi = get_table_embdi(dataset_name,dataset_id=dataset_id)

    ## alpha from 0.1 to 1.0
    my_list = [i / 10 for i in range(1,11)]

    print(my_list)
    best_f1 = 0.0
    best_alpha = my_list[0]
    best_info = ""
    test_data_path = "data/" + dataset_name  + "/test{}.json".format(dataset_id)
    data = sdata_read(test_data_path)
    for alpha in my_list:
        f1, best_th, info = sevaluate(final_model,data,omop_table_embdi,target_table_embdi, alpha)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_info = info
    
    print(best_alpha,best_info)


def twinbert_embdi(opt):
    if opt.isDistilBert:   
        bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
    else:
        bert_tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta')

    # save_path = 'saved3/' + "{}-{}-state".format(opt.dataset,opt.dataset_id) +  "/" +  opt.dataset
    save_path = 'twinbert-weight/{}-{}-state/{}'.format(dataset_name,dataset_id,dataset_name)
    data_path = "data/" + opt.dataset + "/"
    test_loader = data_load(data_path=data_path+"test{}.json".format(opt.dataset_id),batch_size=4,opt=opt,has_weight=False)
    final_model = CompositeModel(
        sbert_path = save_path + '/sbert',
        sqlbert_path = save_path + '/sqlbert',
        hidden_size = 768,
        all_head_size = 768,
        head_num = 8,
        n_layers = opt.crosslayers, 
        isDistil = opt.isDistilBert,
        device = opt.device
    )
    final_model.load(save_path)
    final_model.to(opt.device)

     ## alpha from 0.1 to 0.5
    my_list = [i / 10 for i in range(1,6)]

    print(my_list)
    best_f1 = 0.0
    best_alpha = my_list[0]
    best_info = ""
    for alpha in my_list:
        opt.alpha = alpha
        f1, best_th, info = evaluate(final_model,test_loader,opt,bert_tokenizer)
        
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_info = info
    
    print(best_alpha,best_info)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cms', type=str, help='mimic,harduni')
    parser.add_argument('--dataset_id', default=0, type=int)
    parser.add_argument('--isDistilBert', action="store_true", help="whether to use distilBERT model")
    parser.add_argument('--crosslayers', default=2, type=int)
    parser.add_argument('--device', default='cuda:2', type=str, help="cuda device")
    parser.add_argument('--has_state', action="store_true", help="whether to use the state info")



    opt = parser.parse_args()
    opt.isDistilBert = True

    device = opt.device
    dataset_name = opt.dataset
    dataset_id = opt.dataset_id
    
    model_name = 'twinbert-weight/{}-{}-state/{}/sbert'.format(dataset_name,dataset_id,dataset_name)

    generate_embdi(model_name,dataset_name,dataset_id,device)

    # sbert_embdi(opt)
    twinbert_embdi(opt)
    

