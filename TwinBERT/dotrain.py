## TwinBERT train class : Train a TwinBERT model for different datasets
import argparse
import json
import random
from sklearn import metrics
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,DistilBertModel,DebertaV2Tokenizer

from model.TwinBERT import *
from model.sbert import *
from model.sqltest import *
from model.utils import pad_and_trancate,flatten_list,MAX_STATE_NUM,DISTILBERT_PATH
from model.loss import SoftmaxLoss,CosineSimilarityLoss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def plot_res(title, x_data, y_data):
    plt.cla()

    plt.plot(x_data, y_data)
    plt.title(title)
    plt.savefig(title)
    plt.show()

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
    device = opt.device
    all_y = []
    all_probs = []
    for data in tqdm(test_loader, desc="step", disable=disable_bar):
        sent1 = bert_tokenizer(data['sentences1'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        query1 = sql_tokenizer(data['query1'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        sent2 = bert_tokenizer(data['sentences2'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        query2 = sql_tokenizer(data['query2'], max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        with torch.no_grad():
            if opt.has_state :
                embeddingA = model(sent1,query1,data['state1'].to(device))
                embeddingB = model(sent2,query2,data['state2'].to(device))
                del data['state1'],data['state2']
            else:
                embeddingA = model(sent1,query1)
                embeddingB = model(sent2,query2)
        cos_sim = torch.cosine_similarity(embeddingA, embeddingB)
        all_y.append(data['label'])
        all_probs.append(cos_sim)
        del sent1,query1,sent2,query2
        torch.cuda.empty_cache()

    all_y = flatten_list(all_y)
    all_probs = flatten_list(all_probs)

    best_th = 0.5
    f1 = 0.0 # metrics.f1_score(all_y, all_p)
    best_info = ""

    # print(all_probs)
    min_th = min(all_probs)
    max_th = max(all_probs) - 0.04
    for th in np.arange(min_th, max_th, 0.04):
        pred = [1 if p > th else 0 for p in all_probs]
        precision,recall,new_f1,_ = metrics.precision_recall_fscore_support(all_y, pred, labels=[1])
        if new_f1 > f1:
            f1 = new_f1
            best_th = th
            best_info = "recall:%.4f, precision:%.4f, f1:%.4f" % (recall,precision,f1)
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

def twobert_train(opt,
        sbert_path = 'DISTILBERT_PATH',
        model_save_path = './saved',
        bert_tokenizer = None):
    num_epochs = opt.epoch
    batch_size = opt.batch_size
    device = opt.device
    sqlbert_name = opt.sqlbert

    if model_save_path is not None:   
        model_save_path = model_save_path + '/' + opt.dataset

    twobert = CompositeModel(
        sbert_path = sbert_path,
        sqlbert_path = '../../bert-sql/saved/distilbert/' + sqlbert_name,
        hidden_size = 768,
        all_head_size = 768,
        head_num = 8,
        n_layers = opt.crosslayers, 
        isDistil = opt.isDistilBert,
        device = device
    ).to(device)

    data_path = "data/" + opt.dataset + "/"
    train_loader = data_load(data_path=data_path+"train{}.json".format(opt.dataset_id),batch_size=batch_size,opt=opt,has_weight=True)
    val_loader = data_load(data_path=data_path+"val{}.json".format(opt.dataset_id),batch_size=1,opt=opt,has_weight=False)
    test_loader = data_load(data_path=data_path+"test{}.json".format(opt.dataset_id),batch_size=1,opt=opt,has_weight=False)
    if opt.softmax2:
        train_loss = SoftmaxLoss(model=twobert).to(device)
    else:
        train_loss = CosineSimilarityLoss(model=twobert).to(device)
    # train_loss = NewLoss(model=twobert).to(device)

    accumulation_steps = 2  
    steps_per_epoch = len(train_loader) / accumulation_steps
    num_train_steps = int(steps_per_epoch * num_epochs)

    # Prepare optimizers
    weight_decay = 0.01
    warmup_steps = int(num_train_steps * 0.1)
    optimizer_params = {'lr': 1e-5}
    scheduler = 'WarmupLinear'

    param_optimizer = list(train_loss.named_parameters()) 

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    f1_scores = []
    log_info  = ""
    max_f1 = 0 
    best_th = 0
    for epoch in trange(num_epochs, desc="Epoch"):
        total_loss = 0
        ## do test 
        if epoch > 0:
            twobert.eval()
            print('the current epoch is ',epoch)
            f1_score,th,info = evaluate(twobert,val_loader,opt,bert_tokenizer)
            f1_scores.append(f1_score)
            print("For test:")
            evaluate(twobert,test_loader,opt,bert_tokenizer)
        #save max f1-score model version
        if epoch > 0 and f1_score > max_f1:
            log_info = info
            max_f1 = f1_score
            best_th = th
            if model_save_path is not None:   
                twobert.save(model_save_path)

        train_loss.train()
        i = 0
        for batch in tqdm(train_loader,desc='Step'):
            train_loss.zero_grad()
            labels = batch['label'].to(device)
            sent1 = bert_tokenizer(batch['sentences1'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            query1 = sql_tokenizer(batch['query1'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            sent2 = bert_tokenizer(batch['sentences2'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            query2 = sql_tokenizer(batch['query2'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            
            if opt.has_state :
                sentence_features = [dict(sentence_input=sent1.to(device),query_input=query1.to(device),state_input=batch['state1'].to(device)),
                                    dict(sentence_input=sent2.to(device),query_input=query2.to(device),state_input=batch['state2'].to(device))]     
                # sentence_features = dict(sent1=sent1.to(device),sent2=sent2.to(device),query1=query1.to(device),query2=query2.to(device),state1=batch['state1'].to(device),state2=batch['state2'].to(device))         
            else:
                sentence_features = [dict(sentence_input=sent1.to(device),query_input=query1.to(device),),
                                    dict(sentence_input=sent2.to(device),query_input=query2.to(device),)]     
                # sentence_features = dict(sent1=sent1.to(device),sent2=sent2.to(device),query1=query1.to(device),query2=query2.to(device)) 
            loss = train_loss(sentence_features,labels)
            loss = loss / accumulation_steps
            
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item()
            i += 1
            
        print('\ntrain loss: ',total_loss)

    twobert.eval()
    f1_score,th,info = evaluate(twobert,val_loader,opt,bert_tokenizer)
    f1_scores.append(f1_score)
    print("For test:")
    evaluate(twobert,test_loader,opt,bert_tokenizer)
    #save max f1-score model version
    if f1_score > max_f1:
        log_info = info
        best_th = th
        max_f1 = f1_score
        if model_save_path is not None:   
            twobert.save(model_save_path)
    
    # print(log_info,"best_th: {:.4f}".format(best_th))

def main(opt):
    seed_everything(2021)
    if opt.isDistilBert:   
        bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
    else:
        bert_tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta')
    if opt.train:
        save_path = None
        if opt.store:
            save_path = opt.save_path
        sbert_path = './saved/sentencebert/' + opt.dataset
        if opt.softmax:
            sbert_path += "-model-3" 
        else:
            sbert_path += "-model-cos"
        if opt.strategy == 0:
            sbert_path += "-cls/"
        elif opt.strategy == 1:
            sbert_path += "-mean/"
        elif opt.strategy == 2:
            sbert_path += "-max/"
        else:
            sbert_path += "-sqrt/"
        sbert_path += str(opt.dataset_id)
        if opt.isDistilBert:
            sbert_path += "-distilbert"
        print(sbert_path)
        if opt.first_train:
            print("First train sbert:")
            sbert_train(opt,sbert_path)
        if opt.second_train:
            twobert_train(opt,sbert_path,save_path,bert_tokenizer)
            save_path += '/' + opt.dataset
            print('\nThe final test:')
            data_path = "data/" + opt.dataset + "/"
            test_loader = data_load(data_path=data_path+"test{}.json".format(opt.dataset_id),batch_size=opt.batch_size,opt=opt,has_weight=False)
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
            evaluate(final_model,test_loader,opt,bert_tokenizer)

    else:
        save_path = './saved_final/' + opt.dataset + '-' + str(opt.dataset_id) + '/' + opt.dataset
        # print('\nThe final test:')
        data_path = "data/" + opt.dataset + "/"
        test_loader = data_load(data_path=data_path+"test{}.json".format(opt.dataset_id),batch_size=opt.batch_size,opt=opt,has_weight=False)
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
        evaluate(final_model,test_loader,opt,bert_tokenizer,test=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='harduni', type=str, help='mimic,harduni')
    parser.add_argument('--sbert_epoch', default=10, type=int)
    parser.add_argument('--epoch', default=6, type=int)
    parser.add_argument('--device', default='cuda:2', type=str, help="cuda device")
    parser.add_argument('--save_path', default='./saved/', type=str, help="the path to save the model")
    parser.add_argument('--sqlbert', default='clc-adapter-new', type=str, help="sqlbert model path")
    parser.add_argument('--model_path', default='./saved/', type=str, help="the path to load the model")
    parser.add_argument('--strategy', default=1, type=int, help="the pooling strategy:0[CLS];1[MEAN];2[MAX];3[SQRT]")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--crosslayers', default=1, type=int)
    parser.add_argument('--has_state', action="store_true", help="whether to use the state info")
    parser.add_argument('--first_train', action="store_true", help="whether to train sbert")
    parser.add_argument('--second_train', action="store_true", help="whether to train sbert")
    parser.add_argument('--train', action="store_true", help="whether to train the model(twobert)")
    parser.add_argument('--store', action="store_true", help="whether to store the model")
    parser.add_argument('--softmax2', action="store_true", help="whether to use softmax in the twobert training")
    parser.add_argument('--softmax', action="store_true", help="whether to use softmax in the sbert training")
    parser.add_argument('--isDistilBert', action="store_true", help="whether to use distilBERT model")
    parser.add_argument('--dataset_id', default=1, type=int)

    opt = parser.parse_args()
    print(opt)
    main(opt)