import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForTokenClassification, AdamW, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

MODEL_IDS = {'bert-base-cased', 'allenai/scibert_scivocab_cased', 'EMBEDDIA/sloberta', 'EMBEDDIA/crosloengual-bert'}

train_config = [
    {'experiment': 'EN_def+gen',
     'model_id': 'bert-base-cased',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 4},

    {'experiment': 'EN_def+gen+definitor_btag',
    'model_id': 'allenai/scibert_scivocab_cased',
    'max_length': 128,
    'batch_size': 4,
    'epochs': 4},
    
    {'experiment': 'SL_def+gen',
     'model_id': 'EMBEDDIA/crosloengual-bert',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 2},

    {'experiment': 'EN_def+gen_btag',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
    {'experiment': 'EN_def+gen_btag',
     'model_id': 'bert-base-cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},


    {'experiment': 'EN_nonhier+def_btag',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},

    {'experiment': 'EN_top4nonhier+def_btag',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},

    {'experiment': 'SL_def+gen+definitor_btag',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
    {'experiment': 'SL_def+gen+definitor_btag',
     'model_id': 'EMBEDDIA/sloberta',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},


    {'experiment': 'SL_def+gen_btag',
     'model_id': 'bert-base-cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
    {'experiment': 'SL_def+gen_btag',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
    {'experiment': 'SL_def+gen_btag',
     'model_id': 'EMBEDDIA/sloberta',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
    {'experiment': 'SL_def+gen_btag',
     'model_id': 'EMBEDDIA/crosloengual-bert',
     'max_length': 128,
     'batch_size': 6,
     'epochs': 6},

    {'experiment': 'SL_top4nonhier+def_btag',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
    {'experiment': 'SL_top4nonhier+def_btag',
     'model_id': 'EMBEDDIA/crosloengual-bert',
     'max_length': 128,
     'batch_size': 8,
     'epochs': 6},
]


def get_tokenizer_object(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=False)
    return tokenizer


def get_model_object(model_id, label2code):
    model = BertForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(label2code),
        output_attentions=False,
        output_hidden_states=False
    )
    return model


def model_id_to_path(model_id) :
    """prepare model_id so it can be a name of one directory - character / will create two directories
    example: EMBEDDIA/sloberta --> EMBEDDIA_sloberta"""
    model_path = model_id.replace('/', '_')
    return model_path


def check_config(configs) :
    must_have_keys = ['experiment', 'model_id', 'max_length', 'batch_size', 'epochs']
    experiments = {}
    for conf in configs:
        for key in must_have_keys:
            if key not in conf:
                raise KeyError(f'Missing key in the config dictionary: {key} not found in  {conf}')
        if conf['experiment'] not in experiments:
            experiments[conf['experiment']] = set()
        if conf['model_id'] in experiments[conf['experiment']]:
            raise NameError(
                f'This model is already a part of an experiment: {conf["model_id"]} already in {conf["experiment"]}')
        experiments[conf['experiment']].add(conf['model_id'])
    return True


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def group_predictions(tokens, tags):
    if len(tokens) != len(tags):
        raise ValueError('tokens list must be the same length as tags list')
    new_tokens = []
    new_tags = []

    new_token = ''
    new_tag = ''
    for i, token in enumerate(tokens):
        if token != '-' and '##' not in token:
            if new_token != '':
                if tokens[i - 1] != '-' and tokens[i - 1] != '\'':
                    if (tokens[i - 1].isnumeric() and token == '.') or (tokens[i - 1] == '.' and token.isnumeric()) or token == '%' or token == '\'':
                        new_token = new_token + token
                        if new_tag != 'O':
                            new_tag = tags[i]
                    else:
                        new_tokens.append(new_token)
                        new_tags.append(new_tag)
    
                        new_token = token
                        new_tag = tags[i]
                else :
                    if token.isnumeric() :
                        new_tokens.append(new_token)
                        new_tags.append(new_tag)
    
                        new_token = token
                        new_tag = tags[i]
                    else :
                        new_token = new_token + token
    
                        if new_tag != 'O':
                            new_tag = tags[i]
            else:
                new_token = token
                new_tag = tags[i]
        else:
            if token == '-' and is_float(new_token) :
                new_tokens.append(new_token)
                new_tags.append(new_tag)

                new_token = token
                new_tag = tags[i]
            else :
                if '##' in token:
                    token = token.replace('##', '')
                new_token = new_token + token
                
                
        
                if new_tag != 'O':
                    new_tag = tags[i]

    if tokens[-1] == '.' :
        new_tokens.append(tokens[-1])
        new_tags.append(tags[-1])
    else :
        new_tokens.append(new_token)
        new_tags.append(new_tag)

    return new_tokens, new_tags


def load_data(tokenizer, df_train, batch_size, max_length):
    tag_list = df_train['Tag'].unique()
    tag_list = np.append(tag_list, "PAD")
    print(f"Tags: {', '.join(map(str, tag_list))}")

    x_train, x_test = train_test_split(df_train, test_size=0.20, shuffle=False, random_state=42)
    x_val, x_test = train_test_split(x_test, test_size=0.50, shuffle=False, random_state=42)

    agg_func = lambda s: [[w, t] for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]

    x_train_grouped = x_train.groupby("Sentence").apply(agg_func)
    x_val_grouped = x_val.groupby("Sentence").apply(agg_func)
    x_test_grouped = x_test.groupby("Sentence").apply(agg_func)

    x_train_sentences = [[s[0] for s in sent] for sent in x_train_grouped.values]
    x_val_sentences = [[s[0] for s in sent] for sent in x_val_grouped.values]
    x_test_sentences = [[s[0] for s in sent] for sent in x_test_grouped.values]

    x_train_tags = [[t[1] for t in tag] for tag in x_train_grouped.values]
    x_val_tags = [[t[1] for t in tag] for tag in x_val_grouped.values]
    x_test_tags = [[t[1] for t in tag] for tag in x_test_grouped.values]

    label2code = {label: i for i, label in enumerate(tag_list)}
    code2label = {v: k for k, v in label2code.items()}

    num_labels = len(label2code)
    print(f"Number of labels: {num_labels}")

    def convert_to_input(sentences, tags):
        input_id_list = []
        attention_mask_list = []
        label_id_list = []
        tokens_list = []
        for x, y in tqdm(zip(sentences, tags), total=len(tags)):
            tokens = []
            label_ids = []

            for word, label in zip(x, y):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label2code[label]] * len(word_tokens))

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            tokens_list.append(tokens)
            input_id_list.append(input_ids)
            label_id_list.append(label_ids)

        input_id_list = pad_sequences(input_id_list,
                                      maxlen=max_length, dtype="long", value=0.0,
                                      truncating="post", padding="post")
        label_id_list = pad_sequences(label_id_list,
                                      maxlen=max_length, value=label2code["PAD"], padding="post",
                                      dtype="long", truncating="post")
        attention_mask_list = [[float(i != 0.0) for i in ii] for ii in input_id_list]

        return input_id_list, attention_mask_list, label_id_list, tokens_list

    input_ids_train, attention_masks_train, label_ids_train, _ = convert_to_input(x_train_sentences, x_train_tags)
    input_ids_val, attention_masks_val, label_ids_val, _ = convert_to_input(x_val_sentences, x_val_tags)
    input_ids_test, attention_masks_test, label_ids_test, tokens_list = convert_to_input(x_test_sentences, x_test_tags)

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
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader, label2code, code2label, tokens_list


def load_data_test(tokenizer, df_train, df_test, batch_size, max_length):
    tag_list = df_train.Tag.unique()
    tag_list = np.append(tag_list, "PAD")
    print(f"Tags: {', '.join(map(str, tag_list))}")

    x_train, x_val = train_test_split(df_train, test_size=0.10, shuffle=False, random_state=42)
    x_test = df_test

    # agg_func = lambda s: [ [w,p,t] for w,p,t in zip(s["Word"].values.tolist(),s["POS"].values.tolist(),s["Tag"].values.tolist())]
    agg_func = lambda s: [[w, t] for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]

    x_train_grouped = x_train.groupby("Sentence").apply(agg_func)
    x_val_grouped = x_val.groupby("Sentence").apply(agg_func)
    x_test_grouped = x_test.groupby("Sentence").apply(agg_func)

    x_train_sentences = [[s[0] for s in sent] for sent in x_train_grouped.values]
    x_val_sentences = [[s[0] for s in sent] for sent in x_val_grouped.values]
    x_test_sentences = [[s[0] for s in sent] for sent in x_test_grouped.values]

    x_train_tags = [[t[1] for t in tag] for tag in x_train_grouped.values]
    x_val_tags = [[t[1] for t in tag] for tag in x_val_grouped.values]
    x_test_tags = [[t[1] for t in tag] for tag in x_test_grouped.values]

    label2code = {label: i for i, label in enumerate(tag_list)}
    code2label = {v: k for k, v in label2code.items()}

    num_labels = len(label2code)
    print(f"Number of labels: {num_labels}")

    def convert_to_input(sentences, tags):
        input_id_list = []
        attention_mask_list = []
        label_id_list = []
        tokens_list = []
        for x, y in tqdm(zip(sentences, tags), total=len(tags)):
            tokens = []
            label_ids = []

            for word, label in zip(x, y):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label2code[label]] * len(word_tokens))

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            tokens_list.append(tokens)
            input_id_list.append(input_ids)
            label_id_list.append(label_ids)

        input_id_list = pad_sequences(input_id_list,
                                      maxlen=max_length, dtype="long", value=0.0,
                                      truncating="post", padding="post")
        label_id_list = pad_sequences(label_id_list,
                                      maxlen=max_length, value=label2code["PAD"], padding="post",
                                      dtype="long", truncating="post")
        attention_mask_list = [[float(i != 0.0) for i in ii] for ii in input_id_list]

        return input_id_list, attention_mask_list, label_id_list, tokens_list

    input_ids_train, attention_masks_train, label_ids_train, _ = convert_to_input(x_train_sentences, x_train_tags)
    input_ids_val, attention_masks_val, label_ids_val, _ = convert_to_input(x_val_sentences, x_val_tags)
    input_ids_test, attention_masks_test, label_ids_test, tokens_list = convert_to_input(x_test_sentences, x_test_tags)

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
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    df_test = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(df_test)
    test_dataloader = DataLoader(df_test, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader, label2code, code2label, tokens_list


def train_model(model, train_dataloader, valid_dataloader, code2label, epochs):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # stats = torch.cuda.memory_stats(device=device)

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

    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for epoch_id in range(epochs):
        print(f"Epoch {epoch_id + 1}")
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # torch.cuda.empty_cache()

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
            # b_input_ids = torch.tensor(b_input_ids, dtype=torch.long, device=device)
            # b_input_mask = torch.tensor(b_input_mask, dtype=torch.long, device=device)
            # b_labels = torch.tensor(b_labels, dtype=torch.long, device=device)

            b_input_ids = b_input_ids.clone().detach().type(torch.long).to(device)
            b_input_mask = b_input_mask.clone().detach().type(torch.long).to(device)
            b_labels = b_labels.clone().detach().type(torch.long).to(device)

            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

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
            scheduler.step()
            del b_input_ids
            del b_input_mask
            del b_labels

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
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            b_input_ids = b_input_ids.clone().detach().type(torch.long).to(device)
            b_input_mask = b_input_mask.clone().detach().type(torch.long).to(device)
            b_labels = b_labels.clone().detach().type(torch.long).to(device)

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

            del b_input_ids
            del b_input_mask
            del b_labels

        eval_loss = eval_loss / nb_eval_steps
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        pred_tags = [[code2label[p_i] for (p_i, l_i) in zip(p, l) if code2label[l_i] != "PAD"]
                     for p, l in zip(predictions, true_labels)]
        valid_tags = [[code2label[l_i] for l_i in l if code2label[l_i] != "PAD"]
                      for l in true_labels]
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()

    # torch.save(model, model_path)

    # Loading a model (see docs for different options)
    # model = torch.load('model/tagger_bert_dfd.pt', map_location=torch.device('cuda'))

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o', label="training loss")
    plt.plot(validation_loss_values, 'r-o', label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    # plt.savefig("training.png")

    return model


FORCE = True


def main():
    print(f'FORCE is {FORCE} - models {"will" if FORCE else "wont"} be retrained')
    check_config(train_config)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Found GPU device: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    for conf in train_config:
        if conf['model_id'] not in MODEL_IDS:
            raise NameError(
                f"""{conf['model_id']} not in recognized MODEL_IDS. Either add it to the
                exsisting MODEL_IDS, or change the model_id in the configuration""")
        model_id_path = model_id_to_path(conf['model_id'])
        experiment_dir = os.path.join('data', 'experiments', conf['experiment'])
        Path(experiment_dir, model_id_path).mkdir(parents=False, exist_ok=True)
        df_train = pd.read_csv(os.path.join(experiment_dir, 'train.csv'))
        test_file_path = os.path.join(experiment_dir, 'test.csv')
        model_path = os.path.join(experiment_dir, model_id_path, 'model.pt')
        anno_path = os.path.join(experiment_dir, model_id_path, 'annotation.csv')

        if not os.path.exists(model_path) or (not os.path.exists(anno_path) and os.path.exists(model_path)) or FORCE:
            json.dump(conf, open(os.path.join(experiment_dir, model_id_path, 'config_dict.json'), 'w'), indent=4)
            tokenizer = get_tokenizer_object(conf['model_id'])
            if os.path.exists(test_file_path):
                df_test = pd.read_csv(test_file_path)
                train_dataloader, valid_dataloader, test_dataloader, label2code, code2label, test_sentences = load_data_test(
                    tokenizer, df_train, df_test, conf['batch_size'], conf['max_length'])
            else:
                train_dataloader, valid_dataloader, test_dataloader, label2code, code2label, test_sentences = load_data(
                    tokenizer, df_train, conf['batch_size'], conf['max_length'])

        if not os.path.exists(model_path) or FORCE:
            print(f"""TRAINING {conf["model_id"]} on experiment: {conf["experiment"]}""")
            model_object = get_model_object(conf['model_id'], label2code)
            model = train_model(model_object, train_dataloader, valid_dataloader, code2label, conf['epochs'])

            torch.save(model, model_path)

        if (not os.path.exists(anno_path) and os.path.exists(model_path)) or FORCE:
            # TEST
            if not os.path.exists(test_file_path):
                continue
            print(f"""TESTING {conf["model_id"]} on experiment: {conf["experiment"]}""")
            model = torch.load(model_path, map_location=device)
            predictions, true_labels = [], []
            sentences = []
            for batch in tqdm(test_dataloader):
                b_input_ids, b_input_mask, b_labels = batch
                sentences.extend(b_input_ids)

                b_input_ids = b_input_ids.clone().detach().type(torch.long).to(device)
                b_input_mask = b_input_mask.clone().detach().type(torch.long).to(device)
                b_labels = b_labels.clone().detach().type(torch.long).to(device)

                b_input_ids.to(device)
                b_input_mask.to(device)
                b_labels.to(device)

                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)

                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

                del b_input_ids
                del b_input_mask
                del b_labels
            results_predicted = [[code2label[p_i] for (p_i, l_i) in zip(p, l) if code2label[l_i] != "PAD"]
                                 for p, l in zip(predictions, true_labels)]
            results_true = [[code2label[l_i] for l_i in l if code2label[l_i] != "PAD"] for l in true_labels]
            report = classification_report(results_true, results_predicted)
            with open(os.path.join(experiment_dir, model_id_path, 'results.txt'), 'w') as fl:
                fl.write(report)

            tokens = []
            tags = []
            sentence_ids = []
            for i, result_sentence in enumerate(results_predicted):
                new_tokens, new_tags = group_predictions(test_sentences[i], result_sentence)
                sentence_ids.extend([i] * len(new_tokens))
                tokens.extend(new_tokens)
                tags.extend(new_tags)
            ann_df = pd.DataFrame(data={'Sentence': sentence_ids, 'Word': tokens, 'Tag': tags})
            ann_df.to_csv(os.path.join(experiment_dir, model_id_path, 'annotation.csv'), index=False)


if __name__ == '__main__':
    main()
