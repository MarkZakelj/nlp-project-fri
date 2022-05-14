# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:10:27 2022

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

            
data = pd.read_csv('data/raw_csv_regions_new.csv', encoding='utf-8')    
rows = []

regions = ['HAS_CAUSE', 'HAS_FORM', 'HAS_FUNCTION', 'HAS_LOCATION', 'HAS_SIZE']

for dat in data.iterrows():
    sentence = dat[1]['SENTENCE']
    for ch in ',.:;!?)]}':
        sentence = sentence.replace(ch, ' ' + ch)
    for ch in '([{.':
        sentence = sentence.replace(ch, ch + ' ')
    
    definiendum = []
    genus = []
    
    if type(dat[1]['DEFINIENDUM']) != float :
        definis = dat[1]['DEFINIENDUM'].split(' ')
        for defi in definis: 
            for de in defi.split('|') :
                definiendum.append(de)
    geni = []
    for reg in regions:
        if type(dat[1][reg]) != float :
            geni += dat[1][reg].split(' ')
            
    if len(geni) > 0 :
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
            word['Tag'] = 'REG'
            genus.remove(token)
        else :
            word['Tag'] = 'O'
        i += 1

        rows.append(word)
    if len(genus) > 0 :
        print(genus)
    if len(definiendum) > 0:
        print(definiendum)
    

outfile = 'data/tokenized_reg_EN_new.csv'
with open(outfile, 'w', encoding="utf-8", newline="") as csvfile:
    fieldnames = ['Sentence', 'Word', 'Tag']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    i = 0
    for row in rows:
        out = {k: row[k] for k in fieldnames if k in row}
        writer.writerow(out)
        i += 1