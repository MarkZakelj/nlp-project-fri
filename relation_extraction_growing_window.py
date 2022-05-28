# -*- coding: utf-8 -*-
"""
Created on Sat May 28 15:55:02 2022

@author: Kert PC
"""
import pandas as pd
from transformers import BertTokenizer, BertPreTrainedModel, AdamW, AutoTokenizer, BertConfig, BertModel
from rbert_model import RBERT
import os
import numpy as np
from rbert_data_loader import load_and_cache_examples
from train_relation_extraction import RelationExtractorTrainer, get_tokenizer, model_id_to_path, get_label, compute_metrics
from rbert_data_loader import TermFrameProcessor, convert_examples_to_features
import torch
from scipy.special import softmax
from tqdm import tqdm


def predict_line(line) :
    examples = processor._create_examples(line, 'train')
    features = convert_examples_to_features(
        examples, conf['max_length'], tokenizer, add_sep_token=False
    )
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device=device)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device=device)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long, device=device)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long, device=device)  # add e2 mask

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long, device=device)
    
    # for i in range(len(all_input_ids))
    with torch.no_grad():
        outputs = model(all_input_ids, all_attention_mask, all_token_type_ids, None, all_e1_mask, all_e2_mask)
        logits = outputs[0].detach().cpu().numpy()
        probs = softmax(logits, axis=1)
    detection = 0
    max_val = np.max(logits[0])
    if max_val > 7:
        detection = np.argmax(logits[0])
    
    return detection, max_val


if __name__ == '__main__':
    device = torch.device('cuda')
    
    conf = {'experiment': 'EN_reg_nonhier+def',
            'model_id': 'allenai/scibert_scivocab_cased',
            'max_length': 128,
            'batch_size': 4,
            'epochs': 5}
    conf['model_dir'] = os.path.join('data', 'experiments', conf['experiment'], model_id_to_path(conf['model_id']))
    conf['eval_dir'] = conf['model_dir']
    conf['data_dir'] = os.path.join('data', 'experiments', conf['experiment'])
    
    tokenizer = get_tokenizer(conf['model_id'])
    processor = TermFrameProcessor(conf)
    
    args = torch.load(os.path.join(conf['model_dir'], "training_args.bin"))
    model = RBERT.from_pretrained(os.path.join(conf['model_dir'], 'model.pt'), args=args)
    model.to(device);
    model.eval()
    
    
    #sentence = "Geer moraines or <e1> washboard moraines </e1> are series of small and roughly parallel ridges of till that are ordinarily associated with lakes or former lakes ."
    
    sentences = []
    test_sentences = {}
    
    with open(os.path.join(conf['data_dir'], 'test.tsv'), 'r') as fp:
        lines = fp.readlines()
        for line in lines :
            sentence_raw = line.split('\t')[1]
            sentence_proc = sentence_raw.replace('<e2> ', '')
            sentence_proc = sentence_proc.replace('</e2> ', '')
            if sentence_proc not in test_sentences :
                test_sentences[sentence_proc] = []
            test_sentences[sentence_proc].append([line.split('\t')[0], line.split('\t')[1]])
                
    arr_test_sent = []
    
    for test_sent in test_sentences :
        sent_group = []
        
        for sent in test_sentences[test_sent] :
            sent_group.append(sent)
        
        arr_test_sent.append(sent_group)
    
    
    with open(os.path.join(conf['data_dir'], 'test.tsv'), 'r') as fp:
        lines = fp.readlines()
        for line in lines :
            sentence_raw = line.split('\t')[1]
            sentence_proc = sentence_raw.replace('<e2> ', '')
            sentence_proc = sentence_proc.replace('</e2> ', '')
            if sentence_proc not in sentences :
                sentences.append(sentence_proc)
            
    all_preds = []
    
    labels = get_label(conf)
    
    for sentence in tqdm(sentences) :
        preds = []
        i1 = sentence.find('<e1>')
        i2 = sentence.find('</e1>')
        window_size_start = 2
        window_size = window_size_start
        words_before = sentence[:i1].strip().split(' ')
        words_after = sentence[i2:].strip().split(' ')
        
        idx1 = 0
        while idx1 < (len(words_before) - window_size + 1):
            idx2 = idx1 + window_size
            e2_before = words_before[:idx1] + ['<e2>'] + words_before[idx1:idx2] + ['</e2>'] + words_before[idx2:]
            line = [['Other', ' '.join(e2_before) + ' ' + sentence[i1:]]]
            prediction, confidence = predict_line(line)
            
            if prediction != 0 :
                max_confidence = confidence
                nu_prediction = prediction
                nu_confidence = confidence
                e2_before_nu = e2_before
                
                window_size += 1
                while prediction == nu_prediction and idx2 < len(words_before) and max_confidence >= nu_confidence - 1 :
                    e2_before = e2_before_nu
                    
                    idx2 = idx1 + window_size
                    e2_before_nu = words_before[:idx1] + ['<e2>'] + words_before[idx1:idx2] + ['</e2>'] + words_before[idx2:]
                    nu_line = [['Other', ' '.join(e2_before_nu) + ' ' + sentence[i1:]]]
                    nu_prediction, nu_confidence = predict_line(nu_line)
                    if nu_confidence > max_confidence :
                        max_confidence = nu_confidence
                    
                    window_size += 1
                    
                preds.append([labels[prediction], ' '.join(e2_before) + ' ' + sentence[i1:]])
                idx1 += window_size - 1
                window_size = window_size_start
            
            idx1 += 1
        
        idx1 = 0
        while idx1 < len(words_after):
            idx2 = idx1 + window_size
            e2_after = words_after[:idx1] + ['<e2>'] + words_after[idx1:idx2] + ['</e2>'] + words_after[idx2:]
            line = [['Other', sentence[:i2] + ' ' + ' '.join(e2_after)]]
            prediction, confidence = predict_line(line)
            
            if prediction != 0 :
                max_confidence = confidence
                nu_prediction = prediction
                nu_confidence = confidence
                e2_after_nu = e2_after
                
                window_size += 1
                idx2 = idx1 + window_size
                while prediction == nu_prediction and idx2 < len(words_after) and max_confidence >= nu_confidence - 1 :
                    e2_after = e2_after_nu
                    
                    e2_after_nu = words_after[:idx1] + ['<e2>'] + words_after[idx1:idx2] + ['</e2>'] + words_after[idx2:]
                    nu_line = [['Other', sentence[:i2] + ' ' + ' '.join(e2_after_nu)]]
                    nu_prediction, nu_confidence = predict_line(nu_line)
                    if nu_confidence > max_confidence :
                        max_confidence = nu_confidence
                    
                    window_size += 1
                    idx2 = idx1 + window_size
                    
                preds.append([labels[prediction], sentence[:i2] + ' ' + ' '.join(e2_after)])
                idx1 += window_size - 2
                window_size = window_size_start
            else :
                idx1 += 1
                
        all_preds.append(preds)
        # lines.append(['HAS_FUNCTION', '<e1> Grab samplers </e1> are buckets or segments that <e2> drive into the sediment layer and enclose and retain a layer </e2> .'])
     
    backup_preds = all_preds   
    
    
    all_preds = arr_test_sent
    all_preds = backup_preds
    gt = []
    prediction = []
    
    
    with open(os.path.join(conf['model_dir'], 'test_reg.csv'), 'w') as fa:
        fa.write('Sentence,Word,Tag\n')
        
        for sent_idx, preds in enumerate(all_preds) :
            if len(preds) > 0:
                tagged_sentence = ['O'] * (len(preds[0][1].replace('  ', ' ').split(' ')) - 4)
                tokens = []
                
                for i, pred in enumerate(preds) :
                    is_def = False
                    is_rel = False
                    test = pred[1].replace('  ', ' ').split(' ') 
                    relation = pred[0]
                    idx = 0
                    for word in pred[1].replace('  ', ' ').split(' ') :
                        if not is_def and '<e1>' in word :
                            is_def = True
                            continue
                        elif is_def and '</e1>' in word :
                            is_def = False
                            continue
                            
                        if not is_rel and '<e2>' in word :
                            is_rel = True
                            continue
                        elif is_rel and '</e2>' in word :
                            is_rel = False
                            continue
                        
                        if is_def :
                            tagged_sentence[idx] = 'DEFINIENDUM'
                        if is_rel :
                            tagged_sentence[idx] = relation
                        
                        if i == 0 :
                            tokens.append(word)
                        idx += 1
                
                for idx, toke in enumerate(tokens) :
                    fa.write(str(sent_idx) + ',' + toke.replace('\n', '') + ',' + tagged_sentence[idx] + '\n')
                
                #prediction.append(tagged_sentence)
                
    prediction.insert(8, ['O'] * len(gt[8]))
    prediction.insert(15, ['O'] * len(gt[15]))  
    
    mapper = {
        'O' : 0,
        'HAS_CAUSE' : 1,
        'HAS_LOCATION' : 2,
        'HAS_FORM' : 3,
        'COMPOSITION_MEDIUM' : 4,
        'HAS_FUNCTION' : 5,
        'HAS_SIZE' : 6,
        'DEFINIENDUM' : 7
        }
    
    def label_2_code(label) :
        return mapper[label]

    from sklearn import metrics
    
    
    nu_gt = []
    nu_pred = []
    
    for i, g in enumerate(gt) :
        nu_gt.append(list(map(label_2_code, gt[i])))
        nu_pred.append(list(map(label_2_code, prediction[i])))
        
    nu_gt = sum(nu_gt, [])
    nu_pred = sum(nu_pred, [])
    
    report = metrics.classification_report(nu_gt, nu_pred, target_names=list(mapper))
    with open(os.path.join(conf['model_dir'], 'results_reg.txt'), 'w') as fl:
        fl.write(report)
    
    
            
    """
    gt = []
    prediction =
    with open(os.path.join(conf['model_dir'], 'test_reg.csv'), 'r') as fa:
    """