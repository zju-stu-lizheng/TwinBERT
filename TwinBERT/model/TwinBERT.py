## TwinBERT model class
from math import sqrt
import os
import torch
import torch.nn as nn
from transformers import DebertaV2Model

from .my_distilbert import *
from .sbert import *
# from .sqltest import *

class Multi_CrossAttention(nn.Module):
    """
    Multi-Head Attention Layer
    when forward,
    the first parameter is used to calculate `value` and `key`,
    the second parameter is used to calculate `query`
    """
    def __init__(self,hidden_size,all_head_size,head_num,attention_dropout=0.1):
        super().__init__()
        self.hidden_size    = hidden_size       # input dim
        self.all_head_size  = all_head_size     # output dim
        self.num_heads      = head_num          # attention head num
        self.h_size         = all_head_size // head_num
        self.dropout = nn.Dropout(p=attention_dropout)

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        ## norm (before attention and FFN)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-12)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-12)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,sql,sentence,attention_mask):
        """
        cross-attention: 
        sql,sentence is two model's hidden layer, sql as k and v, sentence as q
        """
        batch_size = sql.size(0)
        seq_len    = sql.size(1)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(sentence).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(sql).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(sql).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        attention_mask = attention_mask.eq(0)
        if self.num_heads > 1:
            attention_mask = torch.unsqueeze(attention_mask, 1).expand(-1, self.num_heads, -1) 
            attention_mask = torch.unsqueeze(attention_mask, -1).expand(-1, -1, -1, seq_len) 

        scores = torch.matmul(q_s,torch.transpose(k_s, -1, -2))  ##scores.shape -> torch.Size([32, 8, 512, 512])
        # use mask
        scores = scores.masked_fill(attention_mask, -1e9)
        weights = torch.softmax(scores / sqrt(q_s.size(-1)), dim=-1)
        # weights = self.dropout(weights)
        
        attention = torch.matmul(weights,v_s)

        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        attention = attention + sentence
        attention = self.sa_layer_norm(attention)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        output = output + attention
        output = self.output_layer_norm(output)

        return output

class CompositeModel(nn.Module):
    """
    Using Cross-Attention to combine two models
    """
    def __init__(self,sbert_path,sqlbert_path,
                hidden_size=768,
                all_head_size=768,
                head_num=8,
                n_layers=1,
                isDistil=True,
                device='cuda:0'):
        super().__init__()
        self.device = device
        self.max_seqlen = 512
        self.sqlbert = DistilBertModel.from_pretrained(sqlbert_path)
        if isDistil:
            self.sbert = DistilBertModel.from_pretrained(sbert_path)
            self.SEP_TOKEN_IDS = 102
        else:
            self.sbert = DebertaV2Model.from_pretrained(sbert_path)
            self.SEP_TOKEN_IDS = 2

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.output_layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-12)
        self.n_layers = n_layers
        self.crosslayer = nn.ModuleList([Multi_CrossAttention(hidden_size,all_head_size,head_num) for _ in range(self.n_layers)])
        self.classifier = nn.Linear(3 * hidden_size,2)  ## final classifier


    def forward(self,sentence_input,query_input,state_input=None):
        """
        sbert and sqlbert receive sentence and query as input respectively, and the hidden layer performs 2-layer cross-attention calculation
        """
        ## 1. calculate the embedding of sentence and query respectively 
        with torch.no_grad():
            sbert_output = self.sbert(**sentence_input)['last_hidden_state']
        sqlbert_output = self.sqlbert(**query_input,state_vector = state_input)['last_hidden_state']

        all_mask = sentence_input['attention_mask']
        ## 2. sentence is divided into two parts: name (_x) and description (_y), and the mask is calculated
        sep_masks = []
        for batch in sentence_input["input_ids"]:
            for i in range(len(batch)):
                if batch[i] == self.SEP_TOKEN_IDS:
                    sep_mask = torch.zeros(self.max_seqlen,device=self.device)
                    sep_mask[0:i] = 1
                    sep_masks.append(sep_mask)
                    break
        sentence_mask = torch.stack(sep_masks)

        ## 3. through the mask, the sentence is intercepted before the first [SEP] (table name + attribute name: sentence_x)
        mask_before = (sentence_mask == 0).unsqueeze(-1).expand_as(sbert_output)
        sbert_before = sbert_output.masked_fill(mask_before, 0)
        
        ## 4. the cross-attention layer is used to calculate the sentence_x and sql input
        hidden_state = sqlbert_output
        for i, layer_module in enumerate(self.crosslayer):
            layer_outputs = layer_module(
                sql=hidden_state, sentence=sbert_before, attention_mask=sentence_mask
            )
            hidden_state = layer_outputs

        ## 5. the cross-attention result (sentence_x') is spliced with the original sentence_y
        output = hidden_state.masked_fill(mask_before, 0)
        output = self.output_layer_norm(sbert_output - sbert_before + self.alpha*output) 

        ## 6. the final result is calculated by pooling
        return output[:,0,:]    ## [CLS] pooling

        # input_mask_expanded = all_mask.unsqueeze(-1).expand(output.size()).float()
        # sum_embeddings = torch.sum(output * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        # return sum_embeddings / sum_mask

    def table_embdi(self,sentence_input,query_input,state_input=None,table_embdi=None):
        """
        sbert and sqlbert receive sentence (with table embedding) and query as input respectively, and the hidden layer performs 2-layer cross-attention calculation
        """
        ## 1. calculate the embedding of sentence and query respectively 
        with torch.no_grad():
            sbert_output = self.sbert(**sentence_input)['last_hidden_state'] + table_embdi
            sqlbert_output = self.sqlbert(**query_input,state_vector = state_input)['last_hidden_state']

            all_mask = sentence_input['attention_mask']
            ## 2. sentence is divided into two parts: name (_x) and description (_y), and the mask is calculated
            sep_masks = []
            for batch in sentence_input["input_ids"]:
                for i in range(len(batch)):
                    if batch[i] == self.SEP_TOKEN_IDS:
                        sep_mask = torch.zeros(self.max_seqlen,device=self.device)
                        sep_mask[0:i] = 1
                        sep_masks.append(sep_mask)
                        break
            sentence_mask = torch.stack(sep_masks)

            ## 3. through the mask, the sentence is intercepted before the first [SEP] (table name + attribute name: sentence_x)
            mask_before = (sentence_mask == 0).unsqueeze(-1).expand_as(sbert_output)
            sbert_before = sbert_output.masked_fill(mask_before, 0)
            
            ## 4. the cross-attention layer is used to calculate the sentence_x and sql input
            hidden_state = sqlbert_output
            for i, layer_module in enumerate(self.crosslayer):
                layer_outputs = layer_module(
                    sql=hidden_state, sentence=sbert_before, attention_mask=sentence_mask
                )
                hidden_state = layer_outputs

            ## 5. the cross-attention result (sentence_x') is spliced with the original sentence_y
            output = hidden_state.masked_fill(mask_before, 0)
            output = self.output_layer_norm(sbert_output - sbert_before + self.alpha*output) 

            return output[:,0,:]    ## [CLS] pooling
        
    def save(self,model_path):
        """
        save model, with two distilBERT and one Cross-Attention Layer
        """
        sbert_path = model_path+"/sbert"
        if not(os.path.exists(sbert_path)) :
            os.makedirs(sbert_path)
        sqlbert_path = model_path+"/sqlbert"
        if not(os.path.exists(sqlbert_path)) :
            os.makedirs(sqlbert_path)
        cross_path = model_path+"/cross.pkl"
        alpha_path = model_path+"/alpha.pt"
        class_path = model_path+"/class.pth"

        self.sbert.save_pretrained(sbert_path)
        self.sqlbert.save_pretrained(sqlbert_path)
        
        torch.save(self.classifier.state_dict(), class_path)
        torch.save(self.crosslayer,cross_path)
        torch.save(self.alpha,alpha_path)
    
    def load(self,model_path):
        """
        only need to load the parameters of Cross-Attention Layer
        """
        cross_path = model_path+"/cross.pkl"
        alpha_path = model_path+"/alpha.pt"
        class_path = model_path+"/class.pth"

        self.crosslayer = torch.load(cross_path)
        self.alpha      = torch.load(alpha_path)
        state_dict = torch.load(class_path)
        self.classifier.load_state_dict(state_dict)