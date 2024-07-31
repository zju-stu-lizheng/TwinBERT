## train_loss class : Calculate the training loss for different models
import torch
import torch.nn as nn
from typing import Callable

class NewLoss(nn.Module):
    def __init__(self,
        model,
        embedding_dimension: int = 768,
        num_label = 2,
        loss_fct: Callable = nn.CrossEntropyLoss()
        ):
        super().__init__()
        self.model = model
        self.num_label = num_label
        self.loss_fct = loss_fct

    def forward(self,sentence_features,labels=None):
        """
        sentence_features : sent1,sent2,query1,query2
        """
        ##sent1,sent2,query1,query2
        output = self.model.tripleFD(**sentence_features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return output

class SoftmaxLoss(nn.Module):
    def __init__(self,
        model,
        embedding_dimension: int = 768,
        num_label = 2,
        concatenation_sent_difference: bool = True,
        concatenation_sent_multiplication: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss()
        ):
        super().__init__()
        self.model = model
        self.num_label = num_label
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 2
        if concatenation_sent_difference :
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication :
            num_vectors_concatenated += 1
        
        self.classifier = nn.Linear(num_vectors_concatenated * embedding_dimension,num_label)
        self.loss_fct = loss_fct

    def forward(self,sentence_features,labels=None):
        """
        use triple (u,v,|u-v|) to classification
        sentence_features : sentence_input,query_input,state_input
        """
        embA,embB = [self.model(**sentence_feature) for sentence_feature in sentence_features]

        vectors_concat = [embA,embB]
        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(embA - embB))
        if self.concatenation_sent_multiplication:
            vectors_concat.append(embA * embB)
        features = torch.cat(vectors_concat, dim=1)
        output = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return output


class CosineSimilarityLoss(nn.Module):
    """
    cosine similarity loss
    """
    def __init__(self, 
        model, 
        loss_fct = nn.MSELoss()
        ):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct


    def forward(self,sentence_features,labels=None):
        embeddings = [self.model(**sentence_feature) for sentence_feature in sentence_features]
        output = torch.cosine_similarity(embeddings[0], embeddings[1])
        return self.loss_fct(output, labels.float())

