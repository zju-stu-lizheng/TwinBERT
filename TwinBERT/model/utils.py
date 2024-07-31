import torch
import pickle
import torch.nn.functional as F
import numpy as np

MAX_STATE_NUM = 64
DISTILBERT_PATH = '../sbert/msmarco-distilbert-base-tas-b/'

def get_table_embdi(dataset_name,dataset_id):
    
    with open('embdi/omop_{}-{}.pickle'.format(dataset_name,dataset_id), 'rb') as file:
        omop_table_embdi = pickle.load(file)
        
    with open('embdi/{}-{}.pickle'.format(dataset_name,dataset_id), 'rb') as file:
        target_table_embdi = pickle.load(file)
        
    return omop_table_embdi,target_table_embdi

def flatten_list(tensor_list):
    """
    flatten a list of tensors into a one-dimensional numpy array
    """
    numpy_array = [tensor.cpu().numpy() for tensor in tensor_list]
    
    flattened_list = [element for sublist in numpy_array for element in sublist]
    return flattened_list

def pad_and_trancate(data,length,value):
    """
    For a tensor, limit its length to length, if it is greater than truncation, if it is less than padding value
    """
    if data.shape[0] > length:
        return torch.narrow(data, dim=0, start=0, length=length)
    return F.pad(data,(0,length-data.shape[0]),mode='constant',value=value)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))