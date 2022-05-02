# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:23:11 2022

@author: Kert PC
"""

import os
import csv
import re
import copy
import time
import argparse
from collections import defaultdict

from pos_tagger import POSTagger

import pandas as pd
import numpy as np


def group_pos_prediction(tokens, tags) :
    new_tokens = []
    new_tags = []
    
    new_token = ''
    new_tag = ''
    for i, token in enumerate(tokens):
        if '-' != token and '##' not in token :
            if new_token != '' :
                if tokens[i-1] != '-' :
                    new_tokens.append(new_token)
                    new_tags.append(new_tag)
                    
                    if token == '^' :
                        token = '-'
                    new_token = token
                    new_tag = tags[i]
                else :
                    new_token = new_token + token

                    if not (new_tag == 'NOUN' or new_tag == 'ADV' or new_tag == 'VERB'):
                        new_tag = tags[i]
            else :
                new_token = token
                new_tag = tags[i]
        else :
            if '##' in token :
                token = token.replace('##', '')
            new_token = new_token + token 
            
            if not (new_tag == 'NOUN' or new_tag == 'ADV' or new_tag == 'VERB'):
                new_tag = tags[i]
        
    new_tokens.append(tokens[-1])
    new_tags.append(tags[-1])

    return new_tokens, new_tags

model_path = 'model/pos-model.pt'
pos_tagger = POSTagger(model_path)
          
data = pd.read_csv('data/raw_csv.csv', encoding='latin-1')    
rows = []
i = 0
for dat in data.iterrows():
    sentence = dat[1]['SENTENCE']
    #print(sentence)
    
    preds = pos_tagger.tag_sentence(sentence)
    
    nu_sent, nu_tags = group_pos_prediction(preds[0], preds[1])
    
    #sentence = sentence.replace(' - ', '-')
    sentence = sentence.replace(',', ' , ')
    sentence = sentence.replace('^', '-')
    for ch in '.:;!?)]}%':
        sentence = sentence.replace(ch, ' ' + ch)
    for ch in '([{.':
        sentence = sentence.replace(ch, ch + ' ')
    
    sentence = sentence.strip()
    split = sentence.split(' ')
    #print(len(split))
    split = list(filter(lambda el: el != '' and el != ' ', split))
    
    #print(preds)
    
    i += 1
    #print('Our split: ' + str(len(split)) + ' Concat split: ' + str(len(nu_tags)))
    if len(split) != len(nu_tags) :
        print('ERROR at ' + str(i))
    
    definiendum = []
    genus = []
    
    if type(dat[1]['DEFINIENDUM']) != float :
        definis = dat[1]['DEFINIENDUM'].split(' ')
        for defi in definis: 
            for de in defi.split('|') :
                definiendum.append(de)
    if type(dat[1]['GENUS']) != float :
        geni = dat[1]['GENUS'].split(' ')
        for gen in geni: 
            for ge in gen.split('|'):
                genus.append(ge)
                

    for j, token in enumerate(split): 
        token = token.strip()
        
        if token == '' :
            continue
        
        word = defaultdict(list)
        if j == 0 :
            word['Sentence'] = 'Sentence ' + str(dat[0] + 1)
        word['Word'] = token
        word['POS'] = nu_tags[j]
            
        if token in definiendum :
            word['Tag'] = 'DFD'
            definiendum.remove(token)
        elif token in genus :
            word['Tag'] = 'GEN'
            genus.remove(token)
        else :
            word['Tag'] = 'O'

        rows.append(word)
    if len(genus) > 0 :
        print(genus)
    if len(definiendum) > 0:
        print(definiendum)
    

outfile = 'data/tokenized_pos_EN.csv'
with open(outfile, 'w', encoding="latin-1", newline="") as csvfile:
    fieldnames = ['Sentence', 'Word', 'POS', 'Tag']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    i = 0
    for row in rows:
        out = {k: row[k] for k in fieldnames if k in row}
        writer.writerow(out)
        i += 1
     