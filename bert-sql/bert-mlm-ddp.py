import argparse
import os
import json
import copy
from tqdm import trange,tqdm
 
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer,DistilBertConfig
from my_distilbert import DistilBertForMaskedLM
from state_embdi import statement_embdi

# sql state info
MAX_STATE_NUM = 64
SEPECIAL_STATE = [63]
CLOZE_STATE = [62]


se = statement_embdi()

def pad_and_trancate(data,length,value):
    """
    For a tensor, limit its length to length, if it is greater than truncation, if it is less than padding value
    """
    if data.shape[0] > length:
        return torch.narrow(data, dim=0, start=0, length=length)
    return F.pad(data,(0,length-data.shape[0]),mode='constant',value=value)

class TrainDataset(Dataset):
    """
    TrainDataset is a subclass of torch.utils.data.Dataset
    """
    def __init__(self, input_texts, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        max_length=512
        all_data = []
        for input_text in input_texts:
            batch_text = se.pre_process(input_text)
            # print(batch_text)
            emb_out = se.token_embdi(batch_text)
            state_list = SEPECIAL_STATE     ## [CLS]
            for tupledata in emb_out:
                token = tupledata[0]
                state = tupledata[2]
                output = tokenizer.encode(token)
                state_list = state_list + [state] * (len(output) - 2) ## [CLS],[SEP]
            
            state_list = state_list + SEPECIAL_STATE ## [SEP]
            state_list = pad_and_trancate(torch.tensor(state_list),max_length,0)
            features = tokenizer(input_text.replace('"','').replace('\'',''), max_length=max_length, truncation=True, pad_to_max_length=True, return_tensors='pt')
            inputs, labels = self.mask_tokens(features['input_ids'])
            # print(inputs.shape,labels.shape)
            batch = {"inputs": inputs.squeeze(0),
                     "labels": labels.squeeze(0),
                     "states": state_list.squeeze(0)}
            all_data.append(batch)
        self.data = all_data
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        
    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)
        if self.config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.config.special_tokens_mask.bool()
 
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
 
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
 
        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.config.prob_replace_rand / (1 - self.config.prob_replace_mask)
        indices_random = torch.bernoulli(torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
 
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class Config:
    def __init__(self):
        pass
    
    def mlm_config(
        self, 
        mlm_probability=0.15, 
        special_tokens_mask=None,
        prob_replace_mask=0.8,
        prob_replace_rand=0.1,
        prob_keep_ori=0.1,
    ):
        """
        :param mlm_probability: total probability of MLM
        :param special_token_mask: speical token
        :param prob_replace_mask: replace token with [MASK] probability
        :param prob_replace_rand: replace token with random token probability
        :param prob_keep_ori: keep token unchanged probability
        """
        assert sum([prob_replace_mask, prob_replace_rand, prob_keep_ori]) == 1, ValueError("Sum of the probs must equal to 1.")
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori
        
    def training_config(
        self,
        batch_size,
        epochs,
        learning_rate,
        weight_decay,
        device,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        
    def io_config(
        self,
        from_path,
        save_path,
    ):
        self.from_path = from_path
        self.save_path = save_path

def train(model, train_dataloader, config):
    """
    train model
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    """
    assert config.device.startswith('cuda') or config.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(config.device)
    
    # model.to(device)
    
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    
    # adapter_param = ["resizelayer","stateweight"]
    # # freeze all params except adapter params
    # adapter_param_list = [p for n, p in model.named_parameters() if not any(nd in n for nd in adapter_param)]
    # for param in adapter_param_list:
    #     param.requires_grad = False
    
    # optimizer = AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=5e-4)
        
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    for cur_epc in trange(int(config.epochs), desc="Epoch"):
        training_loss = 0
        # print("Epoch: {}".format(cur_epc+1))
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc='Step')):
            input_ids = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            state_vector = F.one_hot(batch['states'], num_classes = MAX_STATE_NUM)
            state_vector = state_vector.to(device)
            loss = model(input_ids=input_ids, labels=labels, state_vector=state_vector).loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            training_loss += loss.item()
            # print(loss.item())
        print("Training loss: ", training_loss)

def main(opt):
    dist.init_process_group(backend='nccl')
    rank = opt.local_rank
    print(f"Start running basic DDP example on rank {rank}.")

    device_id = rank % torch.cuda.device_count()

    config = Config()
    config.mlm_config()
    config.training_config(batch_size=8, epochs=6, learning_rate=1e-5, weight_decay=0, device='cuda:'+str(device_id))
    config.io_config(from_path='./saved/distilbert/ddp', 
                    save_path='./saved/distilbert/ddp-new-new')

    bert_tokenizer = DistilBertTokenizer.from_pretrained('./bert/distilbert')
    modelConfig = DistilBertConfig.from_pretrained(config.from_path)
    modelConfig.statedim = MAX_STATE_NUM
    bert_mlm_model = DistilBertForMaskedLM.from_pretrained(config.from_path,config=modelConfig)

    ## load training texts from spider
    training_texts = []
    with open ("spider/all_sql","r") as f:
        lines = f.readlines()
        for line in lines:
            training_texts.append(line)

    train_dataset = TrainDataset(training_texts, bert_tokenizer, config)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle = (train_sampler is None), sampler=train_sampler, pin_memory=True)
    
    bert_mlm_model = bert_mlm_model.to(device_id)
    bert_mlm_model = nn.parallel.DistributedDataParallel(bert_mlm_model, device_ids=[device_id], find_unused_parameters=True)
    train(bert_mlm_model, train_dataloader, config)

    if rank == 0 and (not os.path.exists(config.save_path)):
        os.mkdir(config.save_path)
    bert_mlm_model.module.save_pretrained(config.save_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int)
    opt = parser.parse_args()
    # print(opt)
    main(opt)

# CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 bert-mlm-ddp.py