# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:05:52 2022

@author: Kert PC
"""

import os
import csv
import re
import copy
import time
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np


            
data = pd.read_csv('data/raw_csv.csv', encoding='latin-1')    
rows = []

for dat in data.iterrows():
    sentence = dat[1]['SENTENCE']
    for ch in ',.:;!?)]}':
        sentence = sentence.replace(ch, ' ' + ch)
    for ch in '([{':
        sentence = sentence.replace(ch, ch + ' ')
    
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
                
    i = 0
    for token in sentence.split(' '): 
        token = token.strip()
        
        if token == '' :
            continue
        
        word = defaultdict(list)
        if i == 0 :
            word['Sentence'] = 'Sentence ' + str(dat[0] + 1)
        word['Word'] = token
            
        if token in definiendum :
            word['Tag'] = 'DFD'
            definiendum.remove(token)
        elif token in genus :
            word['Tag'] = 'GEN'
            genus.remove(token)
        else :
            word['Tag'] = 'O'
        i += 1

        rows.append(word)
    if len(genus) > 0 :
        print(genus)
    if len(definiendum) > 0:
        print(definiendum)
    

outfile = 'data/tokenized_EN.csv'
with open(outfile, 'w', encoding="latin-1", newline="") as csvfile:
    fieldnames = ['Sentence', 'Word', 'Tag']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    i = 0
    for row in rows:
        out = {k: row[k] for k in fieldnames if k in row}
        writer.writerow(out)
        i += 1