# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:07:33 2022

@author: Kert PC
"""
import os

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import gc

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Found GPU device: {torch.cuda.get_device_name(i)}")
    

df_data = pd.read_csv("data/tokenized_EN.csv", encoding="latin1").fillna(method="ffill")
df_data.shape



print(df_data.head)



tag_list = df_data.Tag.unique()
tag_list = np.append(tag_list, "PAD")
print(f"Tags: {', '.join(map(str, tag_list))}")


x_train, x_test = train_test_split(df_data, test_size=0.20, shuffle=False, random_state = 42)
x_val, x_test = train_test_split(x_test, test_size=0.50, shuffle=False, random_state = 42)

x_train.shape, x_val.shape, x_test.shape

agg_func = lambda s: [ [w,t] for w,t in zip(s["Word"].values.tolist(),s["Tag"].values.tolist())]


x_train_grouped = x_train.groupby("Sentence").apply(agg_func)
x_val_grouped = x_val.groupby("Sentence").apply(agg_func)
x_test_grouped = x_test.groupby("Sentence").apply(agg_func)

x_train_sentences = [[s[0] for s in sent] for sent in x_train_grouped.values]
x_val_sentences = [[s[0] for s in sent] for sent in x_val_grouped.values]
x_test_sentences = [[s[0] for s in sent] for sent in x_test_grouped.values]

x_train_tags = [[t[1] for t in tag] for tag in x_train_grouped.values]
x_val_tags = [[t[1] for t in tag] for tag in x_val_grouped.values]
x_test_tags = [[t[1] for t in tag] for tag in x_test_grouped.values]


x_train_sentences[0]
x_train_tags[0]

label2code = {label: i for i, label in enumerate(tag_list)}
code2label = {v: k for k, v in label2code.items()}
label2code

num_labels = len(label2code)
print(f"Number of labels: {num_labels}")


MAX_LENGTH = 128
BATCH_SIZE = 4

#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
#gc.collect()
#torch.cuda.empty_cache()


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def convert_to_input(sentences,tags):
    input_id_list = []
    attention_mask_list = []
    label_id_list = []
    
    for x,y in tqdm(zip(sentences,tags),total=len(tags)):
        tokens = []
        label_ids = []
        
        for word, label in zip(x, y):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label2code[label]] * len(word_tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_id_list.append(input_ids)
        label_id_list.append(label_ids)

    input_id_list = pad_sequences(input_id_list,
                          maxlen=MAX_LENGTH, dtype="long", value=0.0,
                          truncating="post", padding="post")
    label_id_list = pad_sequences(label_id_list,
                     maxlen=MAX_LENGTH, value=label2code["PAD"], padding="post",
                     dtype="long", truncating="post")
    attention_mask_list = [[float(i != 0.0) for i in ii] for ii in input_id_list]

    return input_id_list, attention_mask_list, label_id_list


input_ids_train, attention_masks_train, label_ids_train = convert_to_input(x_train_sentences, x_train_tags)
input_ids_val, attention_masks_val, label_ids_val = convert_to_input(x_val_sentences, x_val_tags)
input_ids_test, attention_masks_test, label_ids_test = convert_to_input(x_test_sentences, x_test_tags)


train_inputs = torch.tensor(input_ids_train)
train_tags = torch.tensor(label_ids_train)
train_masks = torch.tensor(attention_masks_train)

val_inputs = torch.tensor(input_ids_val)
val_tags = torch.tensor(label_ids_val)
val_masks = torch.tensor(attention_masks_val)

test_inputs = torch.tensor(input_ids_test)
test_tags = torch.tensor(label_ids_test)
test_masks = torch.tensor(attention_masks_test)



train_data = TensorDataset(train_inputs, train_masks, train_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label2code),
    output_attentions = False,
    output_hidden_states = False
)

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    model.cuda()
#stats = torch.cuda.memory_stats(device=device)
    
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"The model has {params} trainable parameters")

model_classifier_parameters = filter(lambda p: p.requires_grad, model.classifier.parameters())
params_classifier = sum([np.prod(p.size()) for p in model_classifier_parameters])
print(f"The classifier-only model has {params_classifier} trainable parameters")


from transformers import AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

epochs = 4
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup_steps=0,
    t_total=total_steps
)



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for epoch_id in range(epochs):
    print(f"Epoch {epoch_id+1}")
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0
    
    #torch.cuda.empty_cache()

    # Training loop
    for step, batch in tqdm(enumerate(train_dataloader)):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        b_input_ids = torch.tensor(b_input_ids, dtype=torch.long, device=device)
        b_input_mask = torch.tensor(b_input_mask, dtype=torch.long, device=device)
        b_labels = torch.tensor(b_labels, dtype=torch.long, device=device)
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        
        del b_input_ids
        del b_input_mask
        del b_labels
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += float(loss.item())
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        #scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    
    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
        

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [[code2label[p_i] for (p_i, l_i) in zip(p, l) if code2label[l_i] != "PAD"] 
                                  for p, l in zip(predictions, true_labels)]
    valid_tags = [[code2label[l_i] for l_i in l if code2label[l_i] != "PAD"] 
                                   for l in true_labels]
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()

torch.save(model, 'model/tagger_bert_dfd.pt')



