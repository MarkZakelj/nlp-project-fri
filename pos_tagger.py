# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:33:19 2022

@author: Kert PC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.legacy import data
from torchtext.legacy import datasets

from transformers import BertTokenizer, BertModel, BertForTokenClassification
import transformers

import numpy as np

import time
import random
import functools

class BERTPoSTagger(nn.Module):
    def __init__(self,
                 bert,
                 output_dim, 
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
  
        #text = [sent len, batch size]
    
        text = text.permute(1, 0)
        
        #text = [batch size, sent len]
        
        embedded = self.dropout(self.bert(text)[0])
        
        #embedded = [batch size, seq len, emb dim]
                
        embedded = embedded.permute(1, 0, 2)
                    
        #embedded = [sent len, batch size, emb dim]
        
        predictions = self.fc(self.dropout(embedded))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions

def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens

def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens
"""
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def categorical_accuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)




print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Found GPU device: {torch.cuda.get_device_name(i)}")


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)


text_preprocessor = functools.partial(cut_and_convert_to_id,
                                      tokenizer = tokenizer,
                                      max_input_length = max_input_length)

tag_preprocessor = functools.partial(cut_to_max_length,
                                     max_input_length = max_input_length)

TEXT = data.Field(use_vocab = False,
                  lower = True,
                  preprocessing = text_preprocessor,
                  init_token = init_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

UD_TAGS = data.Field(unk_token = None,
                     init_token = '<pad>',
                     preprocessing = tag_preprocessor)


fields = (("text", TEXT), ("udtags", UD_TAGS))

train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
UD_TAGS.build_vocab(train_data)


BATCH_SIZE = 4
OUTPUT_DIM = len(UD_TAGS.vocab)
DROPOUT = 0.25
LEARNING_RATE = 5e-5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)


bert = BertModel.from_pretrained('bert-base-uncased')

model = BERTPoSTagger(bert,
                      OUTPUT_DIM, 
                      DROPOUT)

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.udtags
                
        optimizer.zero_grad()
        
        #text = [sent len, batch size]
        
        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss = criterion(predictions, tags)
                
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.text
            tags = batch.udtags
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def tag_sentence(model, device, sentence, tokenizer, text_field, tag_field):
    
    model.eval()
    
    if isinstance(sentence, str):
        tokens = tokenizer.tokenize(sentence)
    else:
        tokens = sentence
    
    numericalized_tokens = tokenizer.convert_tokens_to_ids(tokens)
    numericalized_tokens = [text_field.init_token] + numericalized_tokens
        
    unk_idx = text_field.unk_token
    
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    
    token_tensor = torch.LongTensor(numericalized_tokens)
    
    token_tensor = token_tensor.unsqueeze(-1).to(device)
         
    predictions = model(token_tensor)
    
    top_predictions = predictions.argmax(-1)
    
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
    
    predicted_tags = predicted_tags[1:]
        
    assert len(tokens) == len(predicted_tags)
    
    return tokens, predicted_tags, unks

N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model/pos-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


model.load_state_dict(torch.load('model/pos-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')





sentence = 'The Queen will deliver a speech about the conflict in North Korea at 1pm tomorrow.'

tokens, tags, unks = tag_sentence(model, 
                                  device, 
                                  sentence,
                                  tokenizer,
                                  TEXT, 
                                  UD_TAGS)

print(unks)

print("Pred. Tag\tToken\n")

for token, tag in zip(tokens, tags):
    print(f"{tag}\t\t{token}")
    
    
"""
    
    
    
class POSTagger() :
    def __init__(self, model_path) :
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() :
            print('Running on CUDA')
            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']

        init_token = self.tokenizer.cls_token
        pad_token = self.tokenizer.pad_token
        unk_token = self.tokenizer.unk_token
        init_token_idx = self.tokenizer.convert_tokens_to_ids(init_token)
        pad_token_idx = self.tokenizer.convert_tokens_to_ids(pad_token)
        unk_token_idx = self.tokenizer.convert_tokens_to_ids(unk_token)


        text_preprocessor = functools.partial(cut_and_convert_to_id,
                                              tokenizer = self.tokenizer,
                                              max_input_length = max_input_length)

        tag_preprocessor = functools.partial(cut_to_max_length,
                                             max_input_length = max_input_length)

        self.text_field = data.Field(use_vocab = False,
                          lower = True,
                          preprocessing = text_preprocessor,
                          init_token = init_token_idx,
                          pad_token = pad_token_idx,
                          unk_token = unk_token_idx)

        self.tag_field = data.Field(unk_token = None,
                             init_token = '<pad>',
                             preprocessing = tag_preprocessor)
        
        fields = (("text", self.text_field ), ("udtags", self.tag_field))
        
        train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
        self.tag_field.build_vocab(train_data)
    
        
        OUTPUT_DIM = 18
        DROPOUT = 0.25
        
        bert = BertModel.from_pretrained('bert-base-uncased')

        self.model = BERTPoSTagger(bert,
                              OUTPUT_DIM, 
                              DROPOUT)
        
        self.model.load_state_dict(torch.load(model_path))
        
        self.model = self.model.to(self.device)
        
        
    def tag_sentence(self, sentence):
        
        self.model.eval()
        
        if isinstance(sentence, str):
            tokens = self.tokenizer.tokenize(sentence)
        else:
            tokens = sentence
        
        numericalized_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        numericalized_tokens = [self.text_field.init_token] + numericalized_tokens
            
        unk_idx = self.text_field.unk_token
        
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        
        token_tensor = torch.LongTensor(numericalized_tokens)
        
        token_tensor = token_tensor.unsqueeze(-1).to(self.device)
             
        predictions = self.model(token_tensor)
        
        top_predictions = predictions.argmax(-1)
        
        predicted_tags = [self.tag_field.vocab.itos[t.item()] for t in top_predictions]
        
        predicted_tags = predicted_tags[1:]
            
        assert len(tokens) == len(predicted_tags)
        
        return tokens, predicted_tags, unks