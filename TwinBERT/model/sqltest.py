## sql similarity class : Test the similarity of different SQL statements
import argparse
import numpy as np
import csv
from tqdm import trange
from transformers import DistilBertTokenizer
import pandas as pd
from sklearn import metrics
import torch
import torch.nn.functional as F

from .TwinBERT import *
from .state_embdi import statement_embdi
from .utils import DISTILBERT_PATH,pad_and_trancate


# additionally defined token
EMP_TOKEN = "[EMP]"
# bert special token-strings
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

# sql state info
MAX_STATE_NUM = 64
SEPECIAL_STATE_NUM = 63
SEPECIAL_STATE = [63]
CLOZE_STATE = [62]


se = statement_embdi()

bert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)

bert_tokenizer.add_special_tokens({'additional_special_tokens':[EMP_TOKEN,"ASC", "DESC", "AVG"]})

# special_tokens in SQL query
sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "BY", "JOIN", "UNION", "INSERT", "UPDATE", "DELETE"] + ["COUNT", "SUM", "AVG", "MIN", "MAX"] + ["ASC", "DESC"] + ["LIMIT" ,"DISTINCT"] + ["AS", "ON", "USING"] + ["AND","OR"]
punctuation = ["<",">","=","!","'","`",",",";","*","(",")","%","\"",".","_"]
key_tokens = [CLS_TOKEN,SEP_TOKEN,PAD_TOKEN] + [token.lower() for token in sql_keywords] + punctuation
key_ids = [bert_tokenizer.convert_tokens_to_ids(token) for token in key_tokens]

def sqltest(model,query_list1,all_list1,query_list2,all_list2,labels,opt,csv_file,state_1=None,state_2=None) :
    """
    test sqlbert model, calculate similarity, and output similarity to an excel table
    :param model: TwinBERT model path
    :param query_list1,query_list2 : Two schema attribute corresponding sql query records
    :param all_list1,all_list2 : Two schema attribute names
    :param labels: ground truth
    :param csv_file: the save path of the csv file
    """
    device = opt.device

    header=['databaseA','databaseB','all-score','label']
    with open(csv_file,'w') as f:
        writer=csv.writer(f)
        writer.writerow(header)

        columns = len(all_list2)
        is_odd = columns % 2

        for group in [0,1]:
            columns_group = columns // 2
            emb_lists_2 = []
            start = columns_group*group
            end = start + columns_group
            if is_odd and group == 1:
                end = end + 1
            for j in trange(start,end, desc="Attr2",disable=False):
                sent2 = bert_tokenizer(all_list2[j], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
                query2 = bert_tokenizer(query_list2[j], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
                if opt.has_state :
                    temp2 = pad_and_trancate(torch.tensor(state_2[j]) , 512, 0) 
                    state2 = F.one_hot(temp2 , num_classes = MAX_STATE_NUM)
                    embeddingB = model.onlysql(sent2.to(device),query2.to(device),state2.to(device))
                else:
                    embeddingB = model.onlysql(sent2.to(device),query2.to(device))
                    # embeddingB = model(sent2.to(device),query2.to(device))
                emb_lists_2.append(embeddingB)

            for i in trange(len(all_list1), desc="Attr1"):
                sent1 = bert_tokenizer(all_list1[i], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
                query1 = bert_tokenizer(query_list1[i], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
                if opt.has_state :
                    temp1 = pad_and_trancate(torch.tensor(state_1[i]) , 512, 0) 
                    state1 = F.one_hot(temp1 , num_classes = MAX_STATE_NUM)
                    embedding = model.onlysql(sent1.to(device),query1.to(device),state1.to(device))
                else:
                    embedding = model.onlysql(sent1.to(device),query1.to(device))
                    # embedding = model(sent1.to(device),query1.to(device))
                
                for j,embeddingB in zip(range(start,end),emb_lists_2):
                    all_score = torch.cosine_similarity(embedding, embeddingB)
                    
                    label = labels[i*columns+j]
                    
                    data=[all_list1[i],
                        all_list2[j],
                        format(all_score.item(), "6.4f"),   
                        label]
                    writer.writerow(data)

def read_query(filename) -> tuple:
    """
    return attr_name,new_lines
    """
    
    file = open(filename, "r")
    
    lines = file.readlines()
    
    file.close()

    attr_list = []  
    new_lines = []  
    attr_name = []
    SQL_temp = ""
    
    for line in lines:
        if line.startswith("SELECT"):
            SQL_temp = SQL_temp + se.pre_process(line) + ";"
        else:
            line = line.rstrip()
            if line[-1] == ":":
                line = line[0:-1]
            attr_list.append(line)
            if SQL_temp != "":  
                new_lines.append(SQL_temp)
                SQL_temp = ""

    if SQL_temp != "":  
        new_lines.append(SQL_temp)

    for s in attr_list:
        attr_name.append(s)

    return attr_name,new_lines

def print_result(file_name):
    
    file = open("../../bert-sql/inference/twobert_mimic", "r")
    
    lines = file.readlines()
    
    file.close()

    sql_score = pd.read_csv('csv/' + file_name + ".csv")
    all_score = sql_score['all-score']

    for i in range(4):
        if i == 0:
            print("TT")
        elif i == 1:
            print("TF") 
        elif i == 2:
            print("FT") 
        elif i == 3:
            print("FF")
        ids = eval(lines[i])

        all_sum = []
        for id in ids:
            all_sum.append(all_score[id])
        
        print("个数 ",len(all_sum))
        print(format(np.mean(all_sum),".4f"))
        print(format(np.var(all_sum),".5f"))

def sql_evaluate(file_path, index_list):
    """
    read from csv file, use all-score as prediction, label as ground truth
    iterate threshold from 0 to 1, calculate the best threshold and the corresponding f1-score
    """
    sql_score = pd.read_csv(file_path)
    all_score = sql_score['all-score']
    all_label = sql_score['label']

    test_score = [all_score[i] for i in index_list if i < len(all_score)]
    test_label = [all_label[i] for i in index_list if i < len(all_label)]

    best_th = 0.5
    f1 = 0.0 # metrics.f1_score(all_y, all_p)
    best_info = ""

    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in test_score]
        precision,recall,new_f1,_ = metrics.precision_recall_fscore_support(test_label, pred, labels=[1])
        if new_f1 > f1:
            f1 = new_f1
            best_th = th
            best_info = "precision:%.4f, recall:%.4f, f1:%.4f" % (precision,recall,f1)
    print(best_info)
    print("best threshold :%.4f" % best_th)

def generate_csv(opt):
    device = opt.device
    datasets = opt.dataset
    id = opt.dataset_id
    
    
    model_path = './saved_final/{}-{}/{}'.format(datasets,id,datasets)
    bert_model = CompositeModel(
                sbert_path = model_path + '/sbert',
                sqlbert_path = model_path + '/sqlbert',
                hidden_size = 768,
                all_head_size = 768,
                head_num = 8,
                device = opt.device
            )
    bert_model.load(model_path)
    bert_model.to(device)
    
    all_list1,query_list1 = read_query("../../bert-sql/inference/omop.sql")
    all_list2,query_list2 = read_query("../../bert-sql/inference/"+datasets+".sql")

    attr_list1 = []
    attr_list2 = []

    
    for s in all_list1:
        attr_list1.append(s.split("-")[-1])

    for s in all_list2:
        attr_list2.append(s.split("-")[-1])

    labels = (pd.read_excel("../../datasets/"+datasets+"/omop_"+datasets+"_data.xlsx"))['label']
    csv_path = './csv/{}-{}.csv'.format(datasets,id)
    sqltest(bert_model,query_list1,all_list1,query_list2,all_list2,labels,opt,csv_path,state_1=None,state_2=None)

def do_test(opt):
    datasets = opt.dataset
    id = opt.dataset_id

    csv_path = './csv/{}-{}.csv'.format(datasets,id)
    index_path = './csv/index/{}-{}.txt'.format(datasets,id)
    with open(index_path, "r") as f:
        index_list = eval(f.read())
    sql_evaluate(csv_path,index_list)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-state','--has_state', action="store_true", help="whether to use the sql state")
    parser.add_argument('--model', default='clc-adapter-new', type=str, help="the model name")
    parser.add_argument('--device', default='cuda:2', type=str, help="the GPU device")
    parser.add_argument('--dataset', default='cms', type=str, help="cms/mimic/synthea")
    parser.add_argument('--dataset_id', default=0, type=int, help="the dataset id")

    opt = parser.parse_args()
    print(opt)
    generate_csv(opt)
    do_test(opt)

