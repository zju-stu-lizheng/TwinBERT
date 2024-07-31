## SentenceBERT class : Train a sbert model for different datasets
import json
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler,SequentialSampler
from sentence_transformers import SentenceTransformer,losses,InputExample
from tqdm import trange,tqdm

from .utils import DISTILBERT_PATH,cos_sim

def batch_to_device(batch, target_device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def sdata_read(data_path) -> tuple:
    """
    preprocess the data for schema matching task
    :param data_path: data path for schema matching task
    :return: sentences1,sentences2,labels
    """
    with open(data_path, "rb") as f:
        datas = json.load(f)
    sentences1 = []
    sentences2 = []
    labels = []
    for data in datas:
        if data_path in ["data/harduni/train1.json","data/harduni/test1.json","data/harduni/val1.json"]:
            print("university dataset")
            sent1 =  data["databaseA"] + ' [SEP] ' + data["descriptionA"]
            sent2 =  data["databaseB"] + ' [SEP] ' + data["descriptionB"]
        else:
            sent1 =  data["omop"] + ' [SEP] ' + data["des1"]
            sent2 =  data["table"] + ' [SEP] ' + data["des2"]

        label = int(data["label"])

        sentences1.append(sent1)
        sentences2.append(sent2)
        labels.append(label)
    
    return sentences1,sentences2,labels

def sdata_load(data_path,batch_size=16,has_weight=True) -> DataLoader:
    """
    load the data for schema matching task, and return a DataLoader
    :param data_path: data path for schema matching task
    :param batch_size: batch size
    :param has_weight: whether to assign weights according to label distribution
    :return: DataLoader
    """
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

def sevaluate(model:SentenceTransformer,data,disable_bar:bool = False) -> float:
    """
    test the model on the test dataset
    :param model: the transformer model
    :param data : test dataset
    :param disable_bar: cancel the progress bar

    :return: f1, best_th, best_info
    """
    sentences1,sentences2,labels = data

    all_y = []
    all_probs = []

    for i in trange(len(labels), desc="step", disable=disable_bar):
        embeddings = [model.encode(sentences1[i]),model.encode(sentences2[i])]
        output = cos_sim(embeddings[0],embeddings[1])
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

def sbert_train(opt,
          model_save_path: str = None):
    """
    train a sbert model for schema matching task
    :param opt: the options
    :param model_save_path: the path to save the model
    """
    num_epochs = opt.sbert_epoch
    dataset_name = opt.dataset
    device = opt.device
    ## init the model (read from path)
    if opt.isDistilBert:
        model_path = DISTILBERT_PATH
    else:
        model_path = 'microsoft/deberta/'
    model = SentenceTransformer(model_path,device = device)
    pooling = model[1]
    pooling.pooling_mode_cls_token = False
    pooling.pooling_mode_mean_tokens = False
    pooling.pooling_mode_max_tokens = False
    pooling.pooling_mode_mean_sqrt_len_tokens = False
    if opt.strategy == 0:
        pooling.pooling_mode_cls_token = True
    elif opt.strategy == 1:
        pooling.pooling_mode_mean_tokens = True
    elif opt.strategy == 2:
        pooling.pooling_mode_max_tokens = True
    else:
        pooling.pooling_mode_mean_sqrt_len_tokens = True

    ## load the train dataset & test dataset
    train_data_path = "data/" + dataset_name + "/train{}.json".format(opt.dataset_id)
    test_data_path = "data/" + dataset_name  + "/test{}.json".format(opt.dataset_id)
    val_data_path = "data/" + dataset_name  + "/val{}.json".format(opt.dataset_id)
    train_dataloader = sdata_load(train_data_path,has_weight=True)
    test_dataloader = sdata_read(test_data_path)
    val_dataloader = sdata_read(val_data_path)
    if opt.softmax:
        train_loss = losses.SoftmaxLoss(model,sentence_embedding_dimension=768,num_labels=2)
    else:
        train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    print("model training------")
    model.to(model._target_device)
    train_loss.to(model._target_device)

    # Use smart batching
    train_dataloader.collate_fn = model.smart_batching_collate
    # test_dataloader.collate_fn = model.smart_batching_collate
    # val_dataloader.collate_fn = model.smart_batching_collate

    steps_per_epoch = len(train_dataloader)
    num_train_steps = int(steps_per_epoch * num_epochs)

    # Prepare optimizers
    weight_decay = 0.01
    max_grad_norm = 1.0
    optimizer_params = {'lr': 2e-5}
    scheduler = 'WarmupLinear'

    param_optimizer = list(train_loss.named_parameters()) 

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
    scheduler_obj = model._get_scheduler(optimizer, scheduler=scheduler, 
                                        warmup_steps=steps_per_epoch,   ## one epoch for warmup
                                         t_total=num_train_steps)

    f1_scores = []
    train_losses = []
    log_info  = ""
    max_f1 = 0 
    best_th = 0
    for epoch in trange(num_epochs, desc="Epoch"):
        total_loss = 0
        train_loss.train()
        for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
            data = next(iter(train_dataloader))
            features, labels = data
            labels = labels.to(model._target_device)
            features = list(map(lambda batch: batch_to_device(batch, model._target_device), features))

            if opt.softmax:
                loss_value = train_loss(features, labels)
            else:
                loss_value = train_loss(features, labels.float())
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(train_loss.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler_obj.step()
            total_loss += loss_value.item()
        
        model.eval()
        print('the current epoch is ',epoch)
        print('for val:')
        f1_score,th,info = sevaluate(model,val_dataloader)
        f1_scores.append(f1_score)
        if epoch >= 0 and f1_score > max_f1:
            log_info = info
            best_th = th
            max_f1 = f1_score
            if model_save_path is not None and opt.store:   #save final model version
                model.save(model_save_path)
        # print('for test:')
        # sevaluate(model,test_dataloader)
        ## save max f1-score model version
        print('\ntrain loss: ',total_loss)
        train_losses.append(total_loss)
            
    print(log_info)
    print('\nThe final test:')
    final_model = SentenceTransformer(model_save_path,device = device)
    final_model.to(final_model._target_device)
    f1_score,th,info = sevaluate(final_model,test_dataloader)
    print("best_th: {:.4f}".format(best_th))
    
if __name__ == '__main__':
    device = 'cuda:5'
    test_data_path = "data/mimic/test1.json"
    test_dataloader = sdata_load(test_data_path,has_weight=False)
    val_data_path = "data/mimic/val1.json"
    val_dataloader = sdata_load(val_data_path,has_weight=False)
    
    final_model = SentenceTransformer('./saved/sentencebert/mimic-model-cos-mean/',device = device)
    final_model.to(final_model._target_device)
    test_dataloader.collate_fn = final_model.smart_batching_collate
    val_dataloader.collate_fn = final_model.smart_batching_collate
    # stest(final_model,test_dataloader)
    # stest(final_model,val_dataloader)