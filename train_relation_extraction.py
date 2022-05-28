# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:16:37 2022

@author: Kert PC
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertPreTrainedModel, AdamW, AutoTokenizer, BertConfig, BertModel
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import seaborn as sns

from rbert_model import RBERT 
from rbert_data_loader import load_and_cache_examples

from train_sequence_tagging import MODEL_IDS, model_id_to_path, check_config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

train_config = [
    {'experiment': 'EN_reg_nonhier+def',
     'model_id': 'bert-base-cased',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 5},
    
    {'experiment': 'EN_reg_nonhier+def',
     'model_id': 'allenai/scibert_scivocab_cased',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 5},
    
    {'experiment': 'SL_reg_nonhier+def',
     'model_id': 'bert-base-cased',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 5},
    
    {'experiment': 'SL_reg_nonhier+def',
     'model_id': 'EMBEDDIA/crosloengual-bert',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 5},

    {'experiment': 'SL_reg_nonhier+def',
     'model_id': 'EMBEDDIA/sloberta',
     'max_length': 128,
     'batch_size': 4,
     'epochs': 5}
    
    
]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args['data_dir'], 'labels.txt'), "r", encoding="utf-8")]

def compute_metrics(preds, labels, label_list):
    assert len(preds) == len(labels)
    f1 = metrics.f1_score(labels, preds, average='macro', zero_division=0)
    pr = metrics.precision_score(labels, preds, average='macro', zero_division=0)
    re = metrics.recall_score(labels, preds, average='macro', zero_division=0)
    
    f1_all = metrics.f1_score(labels, preds, average=None, zero_division=0)
    pr_all = metrics.precision_score(labels, preds, average=None, zero_division=0)
    re_all = metrics.recall_score(labels, preds, average=None, zero_division=0)
    
    result = {"acc": simple_accuracy(preds, labels),
              "f1" : f1,
              "pr" : pr,
              "re" : re,
              "supp" : len(labels)}
    
    per_class_result = {}
    
    for i in range(1, max(labels) + 1) :
        per_class_result[i] = {"f1" : f1_all[i-1],
                               "pr" : pr_all[i-1],
                               "re" : re_all[i-1],
                               "supp" : np.count_nonzero(labels == i)}
    
    return result, per_class_result

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def write_prediction(args, output_file, preds) :
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f :
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))
            

def get_tokenizer(tokenizer_id) :
    tokenizer = None
    if tokenizer_id == 'EMBEDDIA/sloberta' :
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, do_lower_case=False)
    else :
        tokenizer = BertTokenizer.from_pretrained(tokenizer_id, do_lower_case=False)
        
        
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

class RelationExtractorTrainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.save_steps = 250
        self.logging_steps = 250

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config = BertConfig.from_pretrained(
            args['model_id'],
            num_labels=self.num_labels,
            finetuning_task=args['experiment'],
            id2label={str(i): label for i, label in enumerate(self.label_lst)},
            label2id={label: i for i, label in enumerate(self.label_lst)},
        )
        self.model = RBERT.from_pretrained(args['model_id'], config=self.config, args=args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args['batch_size'],
        )

        t_total = len(train_dataloader) * self.args['epochs']

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=2e-5,
            eps=1e-8,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args['epochs'])
        logger.info("  Total train batch size = %d", self.args['batch_size'])
        logger.info("  Gradient Accumulation steps = %d", 1)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.logging_steps)
        logger.info("  Save steps = %d", self.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        max_grad_norm = 1.0
        
        train_iterator = trange(int(self.args['epochs']), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]
                gradient_accumulation_steps = 1
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        self.evaluate("test")  # There is no dev set for semeval task

                    if self.save_steps > 0 and global_step % self.save_steps == 0:
                        self.save_model()


        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args['batch_size'])

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args['batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=1)
        write_prediction(self.args, os.path.join(self.args['eval_dir'], "annotation.txt"), preds)

        result, per_class_results = compute_metrics(preds, out_label_ids, get_label(self.args))
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results, per_class_results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args['model_dir']):
            os.makedirs(self.args['model_dir'])
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(os.path.join(self.args['model_dir'], 'model.pt'))

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args['model_dir'], "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args['model_dir'])

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(os.path.join(self.args['model_dir'], 'model.pt')):
            raise Exception("Model doesn't exists! Train first!")

        self.args = torch.load(os.path.join(self.args['model_dir'], "training_args.bin"))
        self.model = RBERT.from_pretrained(os.path.join(self.args['model_dir'], 'model.pt'), args=self.args)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")



def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def writeout_results(results, class_results, args) :
    labels = get_label(args)
    writeout = ''
    
    with open(args['model_dir'] + '/results.txt', "w", encoding="utf-8") as f:
        writeout += "{}\t{}\t{}\t{}\t{}\n".format(' ', 'Precision', 'Recall', 'F1', 'Support')
        
        
        for classs in class_results :
            writeout += "{}\t{}\t{}\t{}\t{}\n".format(labels[classs],
                                              round(class_results[classs]['pr'], 2),
                                              round(class_results[classs]['re'], 2),
                                              round(class_results[classs]['f1'], 2),
                                              class_results[classs]['supp'])
            
        writeout += "{}\t{}\t{}\t{}\t{}\n".format('Macro AVG',
                                              round(results['pr'], 2),
                                              round(results['re'], 2),
                                              round(results['f1'], 2),
                                              results['supp'])
            
        f.write(writeout)
        
    return writeout


FORCE = False

def main():
    check_config(train_config)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Found GPU device: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)
    
    do_train = True
    do_test = False

    for conf in train_config:
        conf['model_dir'] = os.path.join('data', 'experiments', conf['experiment'], model_id_to_path(conf['model_id']))
        conf['eval_dir'] = conf['model_dir']
        conf['data_dir'] = os.path.join('data', 'experiments', conf['experiment'])
        
        init_logger()
        set_seed(1)
        Path(conf['model_dir']).mkdir(parents=False, exist_ok=True)
        
        if (not os.path.exists(os.path.join(conf['model_dir'], 'model.pt')) or FORCE) and do_train:
            print(f'TRAINING {model_id_to_path(conf["model_id"])}')
            json.dump(conf, open(os.path.join(conf['model_dir'], 'config_dict.json'), 'w'), indent=4)
            
            tokenizer = get_tokenizer(conf['model_id'])
            
            train_dataset = load_and_cache_examples(conf, tokenizer, mode="train")
            test_dataset = load_and_cache_examples(conf, tokenizer, mode="test")
            
            trainer = RelationExtractorTrainer(conf, train_dataset=train_dataset, test_dataset=test_dataset)
            trainer.train()    
            # create model dir if it doesn't exist

            trainer.save_model()

            results, class_results = trainer.evaluate('test')
            print(writeout_results(results, class_results, conf)) 
        else :
            if do_train :
                print(f'Already trained {model_id_to_path(conf["model_id"])}')
            
            
        if do_test and os.path.exists(os.path.join(conf['model_dir'], 'model.pt')) :
            print(f'Testing {model_id_to_path(conf["model_id"])}')
            tokenizer = get_tokenizer(conf['model_id'])
        
            test_dataset = load_and_cache_examples(conf, tokenizer, mode="test")
            trainer = RelationExtractorTrainer(conf, train_dataset=None, test_dataset=test_dataset)  
            trainer.load_model()
            
            results, class_results = trainer.evaluate('test')
            print(writeout_results(results, class_results, conf)) 
            

if __name__ == '__main__':
    main()
