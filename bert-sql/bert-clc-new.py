## python bert-clc-new.py --train --store_path ./saved/distilbert/clc-state
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

import os
import torch
import pickle
import random
import argparse
import configparser
from torch.optim import AdamW
from tqdm import trange, tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer,DistilBertConfig
from my_distilbert import DistilBertModel
from state_embdi import statement_embdi

random.seed(11)

se = statement_embdi()


bert_tokenizer = DistilBertTokenizer.from_pretrained('./bert/distilbert')

bert_tokenizer.add_special_tokens({'additional_special_tokens':[EMP_TOKEN,"ASC", "DESC", "AVG"]})


sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "BY", "JOIN", "UNION", "INSERT", "UPDATE", "DELETE"] + ["COUNT", "SUM", "AVG", "MIN", "MAX"] + ["ASC", "DESC"] + ["LIMIT" ,"DISTINCT"] + ["AS", "ON", "USING"] + ["AND","OR"]
punctuation = ["<",">","=","!","'","`",",",";","*","(",")","%","\"",".","_"]
key_tokens = [CLS_TOKEN,SEP_TOKEN,PAD_TOKEN] + [token.lower() for token in sql_keywords] + punctuation
key_ids = [bert_tokenizer.convert_tokens_to_ids(token) for token in key_tokens]

SEP_ID = torch.tensor([bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)])

class Config:
    def __init__(self,config_file='newconfig.ini'):
        config = configparser.ConfigParser() 
        config.read(config_file) 
        self.device = "cuda:" + str(config.getint('device', 'id'))
        ## clc task config
        self.probability = config.getfloat('clc', 'probability')
        self.max_length = config.getint('clc', 'max_length')
        self.max_choice = config.getint('clc', 'max_choice')
        self.num_sent = config.getint('clc', 'num_sent')
    
    def training_config(self,config_file='newconfig.ini'):
        config = configparser.ConfigParser() 
        config.read(config_file) 
        self.batch_size = config.getint('hyperparameters', 'batch_size')
        self.epochs = config.getint('hyperparameters', 'epoch')
        self.learning_rate = config.getfloat('hyperparameters', 'learning_rate')
        self.weight_decay = config.getfloat('hyperparameters', 'weight_decay')

    def __str__(self):
        return f'batch_size :{self.batch_size}\nepoch :{self.epochs}\nlearning_rate :{self.learning_rate}\nweight_decay :{self.weight_decay}\ndevice :{self.device}'
    
def pad_and_trancate(data,length,value):
    """
    For a tensor, limit its length to length, if it is greater than it is truncated, if it is less than it is filled with value
    """
    if data.shape[0] > length:
        return torch.narrow(data, dim=0, start=0, length=length)
    return F.pad(data,(0,length-data.shape[0]),mode='constant',value=value)

class TrainDataset(Dataset):
    """
    create training dataset, this time we need to fill in the words dug out at the end of the SQL statement
    """
    def __init__(self, input_texts, states, tokenizer, config):
        all_data = []
        for i in range(0,len(input_texts)-config.num_sent,int(config.num_sent/2)):
        # for i in range(0,20,int(config.num_sent/2)):
            all_text = []
            all_state  = []
            emp_labels = []
            emp_poses  = []
            emp_states = []

            
            choices = random.sample(range(config.num_sent),int(config.num_sent*config.probability))

            start_pos = 0

            for j in range(config.num_sent):
                batch_text = input_texts[i+j]
                
                new_state = states[i+j] + SEPECIAL_STATE    
                if j == 0: new_state = SEPECIAL_STATE + new_state 
                
                
                features = tokenizer(batch_text, max_length=config.max_length, truncation=True, padding=False, return_tensors='pt')
                
                text_ids = features['input_ids'][0]
                if j > 0: text_ids = text_ids[1:]   
                if j in choices:
                    special_tokens_mask = [1 if val in key_ids else 0 for val in text_ids.tolist()]
                    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

                    mask_matrix = torch.full(text_ids.shape, 1)
                    mask_matrix.masked_fill_(special_tokens_mask, value=0)
                    true_indices = torch.nonzero(mask_matrix)

                    emp_label = []
                    emp_state = []
                    for pos in true_indices:
                        if pos + start_pos < config.max_length :
                            emp_label.append(text_ids[pos])
                            # print(torch.tensor(new_state[pos]).unsqueeze(0))
                            emp_state.append(torch.tensor(new_state[pos]).unsqueeze(0))
                            
                            new_state[pos] = SEPECIAL_STATE_NUM

                    if len(emp_label) > 0:
                        # print(bert_tokenizer.convert_ids_to_tokens(text_ids))
                        emp_label = torch.cat(emp_label,dim=0)
                        emp_state = torch.cat(emp_state,dim=0)
                        
                        text_ids[true_indices] = bert_tokenizer.convert_tokens_to_ids(EMP_TOKEN)

                        emp_labels.append(emp_label)
                        emp_states.append(emp_state)
                        true_indices += start_pos
                        
                        true_indices = torch.clamp(true_indices,max=config.max_length-1)
                        emp_poses.append(true_indices)
                        
                        
                        idx = torch.randperm(emp_label.size(0))

                        
                        label_shuffled = emp_label[idx]
                        
                        new_text_ids = torch.cat((text_ids,SEP_ID,label_shuffled,SEP_ID),dim=0)
                        # print(bert_tokenizer.convert_ids_to_tokens(new_text_ids))
                        all_text.append(new_text_ids)

                        start_pos = start_pos + emp_label.shape[0] + 2
                        new_state = new_state + SEPECIAL_STATE + CLOZE_STATE*emp_label.shape[0] + SEPECIAL_STATE
                    else:
                        all_text.append(text_ids)
                else:
                    all_text.append(text_ids)
                
                new_state = torch.tensor(new_state)
                all_state.append(new_state) 

                
                start_pos += text_ids.shape[0]

            if len(emp_labels) == 0:
                continue 
            # padding 
            all_state  = torch.cat(all_state,dim=0)
            all_text   = torch.cat(all_text,dim=0)
            emp_labels = torch.cat(emp_labels,dim=0)
            emp_states = torch.cat(emp_states,dim=0)
            emp_poses = torch.cat(emp_poses,dim=0).squeeze(dim=1)

            if(all_text.shape[0] != all_state.shape[0]):
                print(all_text.shape[0],all_state.shape[0])
                print(bert_tokenizer.convert_ids_to_tokens(all_text))

            ## build attention_mask
            attention_mask = torch.full(all_text.shape, 1) 
            
            attention_mask = pad_and_trancate(attention_mask,config.max_length,0)
            all_state = pad_and_trancate(all_state,config.max_length,0)
            all_text = pad_and_trancate(all_text,config.max_length,0)
            emp_poses = pad_and_trancate(emp_poses,config.max_choice,0)
            emp_labels = pad_and_trancate(emp_labels,config.max_choice,0)
            emp_states = pad_and_trancate(emp_states,config.max_choice,0)

            data = {
                'input_ids': all_text,
                'state_ids': all_state,
                'attention_mask':attention_mask,
                'emp_pos': emp_poses,        
                'emp_label': emp_labels,     
                'emp_state' : emp_states
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
  
def train(model, train_dataloader, config, bert_tokenizer):
    """
    训练
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    """
    # model.cuda()
    device = torch.device(config.device)
    model.to(device)
    
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")
        
    
    adapter_param = ["resizelayer","stateweight"]
    # freeze all params except adapter params
    adapter_param_list = [p for n, p in model.named_parameters() if not any(nd in n for nd in adapter_param)]
    for param in adapter_param_list:
        param.requires_grad = False
    
    optimizer = AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=5e-4)

    # param_optimizer = list(model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    #     {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    # optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    softmax = torch.nn.Softmax(dim=0)
    criterion = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

    for i in trange(int(config.epochs), desc="Epoch"):
        training_loss = 0
        test(model,bert_tokenizer)
        model.train()

        for batch in tqdm(train_dataloader, desc='Step', disable = False):
            
            for key in batch: 
                batch[key] = batch[key].to(device)
                
            state_vector = F.one_hot(batch['state_ids'], num_classes = MAX_STATE_NUM)
            
            encode_output = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], state_vector = state_vector)
            # print(encode_output)
            encode_output = encode_output['last_hidden_state']
            
            loss = 0
            for embedding,emp_pos,emp_label,emp_state in zip(encode_output,batch['emp_pos'],batch['emp_label'],batch['emp_state']):
                
                batch_prediction = []
                for pos in emp_pos:
                    if pos == 0:
                        break
                    batch_prediction.append(embedding[pos])

                
                batch_label = []
                for i in range(len(batch_prediction)):
                    label_ids = emp_label[i].unsqueeze(0).unsqueeze(0)
                    state_vector = F.one_hot(emp_state[i].unsqueeze(0).unsqueeze(0), num_classes = MAX_STATE_NUM)
                    model_output = model(label_ids, state_vector = state_vector)['last_hidden_state'].squeeze(0)
                    batch_label.append(model_output)
                batch_label = torch.cat(batch_label,dim=0)

                ## dot-product attention
                batch_label = torch.transpose(batch_label,dim0=0,dim1=1)
                dotproduct = [softmax(torch.matmul(prediction,batch_label).squeeze(0)) for prediction in batch_prediction]
                dotproduct = torch.stack(dotproduct,dim=0)
                if len(batch_prediction) == 1:
                    dotproduct = dotproduct.unsqueeze(0)

                loss += criterion(dotproduct,torch.arange(0, dotproduct.shape[0],dtype=torch.long,device=device))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            model.zero_grad()
            training_loss += loss.item()
        print("Training loss: ", training_loss)

def pretrain(bert_model,bert_tokenizer,store_path):
    """
    pretrain bert model
    param bert_model : bert model to train
    param bert_tokenizer: bert tokenizer
    param store_path: model save path
    """
    ## Read config for pretraining
    my_config = Config(config_file='newconfig.ini')
    my_config.training_config(config_file='newconfig.ini')
    print(my_config)
    
    # bert_model = torch.nn.DataParallel(bert_model)
    ## load training texts from spider
    originsql = []
    tokens = []
    states = []
    with open('./spider/state/all_emb.txt','r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i % 3 == 0:
                sql = lines[i][4:]
                originsql.append(sql.replace('"','').replace('\'',''))
            elif i % 3 == 1:
                temp = eval(lines[i])
                tokens.append(temp)
            else:
                temp = eval(lines[i])
                states.append(temp)
    new_states = []
    
    for i in range(len(tokens)):
        token_list = tokens[i]
        state_list = states[i]
        new_state = []
        for token,state in zip(token_list,state_list):
            output = bert_tokenizer.encode(token)
            count = len(output) - 2 ## [CLS],[SEP]
            for _ in range(count):
                new_state.append(state)
        new_states.append(new_state)

    train_dataset = TrainDataset(originsql,new_states, bert_tokenizer, my_config)
    train_dataloader = DataLoader(train_dataset,batch_size=my_config.batch_size)
    train(bert_model, train_dataloader, my_config, bert_tokenizer)
    bert_model.save_pretrained(store_path)

class TestDataset(Dataset):
    def __init__(self, file_path, tokenizer, config, dat_fname):
        if os.path.exists(dat_fname):
            print('loading dataset -----',dat_fname)
            self.data = pickle.load(open(dat_fname, 'rb'))
        else:
            # load test texts from spider
            input_texts = []
            with open (file_path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    
                    input_texts.append(line)
            all_data = []
            for batch_text in input_texts:
                
                batch_text = se.pre_process(batch_text)
                emb_out = se.token_embdi(batch_text)
                state_list = SEPECIAL_STATE     ## [CLS]
                for tupledata in emb_out:
                    token = tupledata[0]
                    state = tupledata[2]
                    output = bert_tokenizer.encode(token)
                    state_list = state_list + [state] * (len(output) - 2) ## [CLS],[SEP]
                
                state_list = state_list + SEPECIAL_STATE ## [SEP]
                
                features = tokenizer(batch_text.replace('"','').replace('\'',''), max_length=config.max_length, truncation=True, pad_to_max_length = False, return_tensors='pt')
                # print(features)
                input_ids = features['input_ids']

                
                text_ids = input_ids[0]

                special_tokens_mask = [1 if val in key_ids else 0 for val in text_ids.tolist()]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

                probability_matrix = torch.full(text_ids.shape, config.probability)
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()

                true_indices = torch.nonzero(masked_indices)
                emp_label = [text_ids[pos] for pos in true_indices]
                emp_state = [torch.tensor(state_list[pos]).unsqueeze(0) for pos in true_indices]
                if len(emp_label) == 0:
                    continue
                emp_label = torch.cat(emp_label,dim=0)
                emp_state = torch.cat(emp_state,dim=0)

                text_ids[true_indices] = bert_tokenizer.convert_tokens_to_ids(EMP_TOKEN)

                
                idx = torch.randperm(emp_label.size(0))

                
                label_shuffled = emp_label[idx]

                
                # new_text_ids = torch.cat((text_ids,label_shuffled),dim=0)
                new_text_ids = torch.cat((text_ids,SEP_ID,label_shuffled,SEP_ID),dim=0)
                state_list = state_list + SEPECIAL_STATE + CLOZE_STATE*emp_label.shape[0] + SEPECIAL_STATE
                state_list = torch.tensor(state_list)
                data = {
                    'input_ids': new_text_ids.unsqueeze(0),
                    'state_ids': state_list.unsqueeze(0),
                    'emp_pos': true_indices.squeeze(1),   
                    'emp_label': emp_label,     
                    'emp_state': emp_state
                }
                all_data.append(data)
            pickle.dump(all_data,open(dat_fname, 'wb'))
            self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def test(model,tokenizer,has_state=True):
    ## Read config for pretraining
    my_config = Config(config_file='config.ini')
    file_path = 'spider/devset'
    test_dataset = TestDataset(file_path, tokenizer, my_config, 'dev-state.dat')

    device = torch.device(my_config.device)
    
    model.to(device)
    model.eval()

    predict_right = 0
    predict_wrong = 0
    for batch in test_dataset:
        
        for key in batch: 
            batch[key] = batch[key].to(device) 

        if has_state:
            state_vector = F.one_hot(batch['state_ids'], num_classes = MAX_STATE_NUM)
            encode_output = model(input_ids = batch['input_ids'],state_vector = state_vector)
        else:
            encode_output = model(input_ids = batch['input_ids'])
        # print(encode_output)
        encode_output = encode_output['last_hidden_state']
        
        embedding = encode_output[0]
        emp_pos = batch['emp_pos']
        emp_label = batch['emp_label']
        emp_state = batch['emp_state']

        
        batch_prediction = []
        for pos in emp_pos:
            if pos == 0:
                break
            batch_prediction.append(embedding[pos])

        
        batch_label = []
        for i in range(len(batch_prediction)):
            label_ids = emp_label[i].unsqueeze(0).unsqueeze(0)
            if has_state:
                state_vector = F.one_hot(emp_state[i].unsqueeze(0).unsqueeze(0), num_classes = MAX_STATE_NUM)
                model_output = model(label_ids, state_vector = state_vector)['last_hidden_state'].squeeze(0)
            else:
                model_output = model(label_ids)['last_hidden_state'].squeeze(0)
            batch_label.append(model_output)
        batch_label = torch.cat(batch_label,dim=0)

        ## dot-product attention
        batch_label = torch.transpose(batch_label,dim0=0,dim1=1)
        for i,prediction in zip(range(len(batch_label)),batch_prediction):
            dotproduct = torch.matmul(prediction,batch_label).squeeze(0)
            if torch.argmax(dotproduct) == i:
                predict_right += 1
            else:
                predict_wrong += 1

    accuracy = predict_right/(predict_right+predict_wrong)
    print('right , wrong :',predict_right,predict_wrong)
    print('Accuracy: %.4f' % accuracy)

def main(opt):
    if opt.train :
        model_path = './saved/distilbert/clc-new'

        modelConfig = DistilBertConfig.from_pretrained(model_path)
        modelConfig.statedim = MAX_STATE_NUM
        bert_model = DistilBertModel.from_pretrained(model_path,config=modelConfig)
        bert_model.resize_token_embeddings(len(bert_tokenizer))

        pretrain(bert_model,bert_tokenizer,opt.store_path)
    if opt.test :
        print('start testing -----')
        
        modelConfig = DistilBertConfig.from_pretrained(opt.model_path)
        modelConfig.statedim = MAX_STATE_NUM
        model = DistilBertModel.from_pretrained(opt.model_path,config=modelConfig)
        model.resize_token_embeddings(len(bert_tokenizer))
        # test(model,bert_tokenizer,opt.has_state)
        my_config = Config(config_file='config.ini')
        file_path = 'spider/devset'
        test_dataset = TestDataset(file_path, bert_tokenizer, my_config, 'dev-state.dat')
        batch = test_dataset[0]

        state_vector = F.one_hot(batch['state_ids'], num_classes = MAX_STATE_NUM)
        encode_output = model(input_ids = batch['input_ids'],state_vector = state_vector)

    se.write_down()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", help="whether to train the model")
    parser.add_argument('--test', action="store_true", help="whether to test the model")
    parser.add_argument('--has_state', action="store_true", help="whether to use the sql state")
    parser.add_argument('--model_path', default='./saved/clc', type=str, help="the path of the model to be tested")
    parser.add_argument('--store_path', default='./saved', type=str, help='the path to store the model')
    parser.add_argument('-disbar','--disable_bar', action="store_true", help="whether to show the progress bar")
    
    opt = parser.parse_args()
    print(opt)
    main(opt)