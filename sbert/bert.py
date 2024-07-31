import argparse
import os
import torch
import numpy as np
from tqdm import trange
from sentence_transformers import SentenceTransformer,losses, util
import sklearn.metrics as metrics
from utils import batch_to_device,data_read,data_load,plot_res

model_path = './msmarco-distilbert-base-tas-b/'

def train(opt,
        model_save_path: str = None):
    """
    训练模型
    :param opt:模型训练选项
    :param model_save_path: 模型保存路径
    """
    dataset_name = opt.dataset
    num_epochs   = opt.epoch
    with_sent   = opt.with_sent
    disable_bar = opt.disable_bar
    device      = opt.device

    ## init the model (read from path)
    model = SentenceTransformer(model_path,device = device)

    ## load the train dataset & test dataset
    train_dataloader = data_load(dataset_name,'train',with_sent,opt.dataset_id)
    val_data = data_read(dataset_name,'val',with_sent,opt.dataset_id)
    test_data= data_read(dataset_name,'test',with_sent,opt.dataset_id)

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

    steps_per_epoch = len(train_dataloader)
    num_train_steps = int(steps_per_epoch * num_epochs)

    # Prepare optimizers
    weight_decay = 0.01
    warmup_steps = int(num_train_steps * 0.1)
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
    scheduler_obj = model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

    f1_scores = []
    train_losses = []
    log_info  = ""
    max_f1 = 0 
    best_th = 0
    for epoch in trange(num_epochs, desc="Epoch",disable=disable_bar):
        #save max f1-score model version
        if epoch > 0 :
            model.eval()
            print('the current epoch is ',epoch)
            f1_score,th,info = evaluate(model,val_data,disable_bar)
            f1_scores.append(f1_score)
            print("test:")
            evaluate(model,test_data)
            if f1_score > max_f1:
                log_info = info
                best_th = th
                max_f1 = f1_score
                if model_save_path is not None:   #save final model version
                    model.save(model_save_path)
        total_loss = 0
        train_loss.train()
        for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=disable_bar):
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
        
        print('\ntrain loss: ',total_loss)
        train_losses.append(total_loss)

    ## final test
    model.eval()
    print('the current epoch is ',epoch+1)
    f1_score,th,info = evaluate(model,val_data,disable_bar)
    f1_scores.append(f1_score)
    print("test:")
    evaluate(model,test_data)
    #save max f1-score model version
    if f1_score > max_f1:
        log_info = info
        best_th = th
        max_f1 = f1_score
        if model_save_path is not None:   #save final model version
            model.save(model_save_path)

    print(log_info,"best_th: ",best_th)
    ## plot the picture
    # pre_path = dataset_name
    # if with_sent:
    #     pre_path = pre_path+'-sent'
        
    # plot_res(pre_path+'-train_loss',range(1,num_epochs+1),train_avglosses)
    # plot_res(pre_path+'-f1_score',range(0,num_epochs+1),f1_scores)

def evaluate(model:SentenceTransformer,data:tuple,disable_bar:bool = False) -> float:
    sentences1,sentences2,labels = data

    all_y = []
    all_probs = []

    for i in trange(len(labels), desc="step", disable=disable_bar):
        embeddings = [model.encode(sentences1[i]),model.encode(sentences2[i])]
        cos_sim = util.cos_sim(embeddings[0],embeddings[1])
        all_y.append(labels[i])
        all_probs.append(cos_sim)

    best_th = 0.5
    f1 = 0.0 # metrics.f1_score(all_y, all_p)
    best_info = ""

    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in all_probs]
        precision,recall,new_f1,_ = metrics.precision_recall_fscore_support(all_y, pred, labels=[1])
        if new_f1 > f1:
            f1 = new_f1
            best_th = th
            best_info = "precision:%.4f , recall:%.4f, f1:%.4f " % (precision,recall,f1)
    print(best_info)

    return f1, best_th, best_info

def main(opt:argparse) -> None:
    dataset_name = opt.dataset
    with_sent   = opt.with_sent
    device      = opt.device
    softmax     = opt.softmax
    dataset_id  = opt.dataset_id
    
    if with_sent:
        suffix = "-model"
    else:
        suffix = "-wosent-model"
    if softmax:
        suffix += "-3"
    else:
        suffix += "-cos"
    output_path = './saved_{}/'.format(dataset_id) + dataset_name + suffix
    if(opt.store):
        model_save_path = output_path
    else:
        model_save_path = None
    if not(model_save_path is None or os.path.exists(model_save_path)) :
        os.mkdir(model_save_path)
    if (opt.train):
        train(opt,model_save_path)
        print("The best model in test datasets:")
        model = SentenceTransformer(model_save_path,device = device)
        model.to(model._target_device)

        ## load the test dataset
        sentences1,sentences2,labels = data_read(dataset_name,'test',with_sent,opt.dataset_id)
        f1, best_th, best_info = evaluate(model,(sentences1,sentences2,labels))
        print(best_th, best_info)

    else:
        
        model = SentenceTransformer('saved2/mimic-model',device = device)
        model.to(model._target_device)

        ## load the test dataset
        sentences1,sentences2,labels = data_read(dataset_name,'test',with_sent)
        evaluate(model,(sentences1,sentences2,labels))

        # sentences1,sentences2,labels = data_read(dataset_name,'all',with_sent)
        # evaluate(model,(sentences1,sentences2,labels))
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mimic', type=str, help='cms,mimic,university')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--dataset_id', default=2, type=int)
    parser.add_argument('--device', default='cuda:3', type=str, help="cuda device")
    parser.add_argument('--train', action="store_true", help="whether to train the model")
    parser.add_argument('--store', action="store_true", help="whether to store the model")
    parser.add_argument('-sent','--with_sent', action="store_true", help="whether to use the description")
    parser.add_argument('-disbar','--disable_bar', action="store_true", help="whether to show the progress bar")
    parser.add_argument('--softmax', action="store_true", help="whether to use softmax in the sbert training")

    opt = parser.parse_args()
    print(opt)
    main(opt)
