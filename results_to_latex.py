# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:29:24 2022

@author: Kert PC
"""

import os

experiment = 'SL_reg_nonhier+def'
model = 'bert-base-cased'
#model = 'allenai_scibert_scivocab_cased'
model = 'EMBEDDIA_crosloengual-bert'
model = 'EMBEDDIA_sloberta'


path = './data/experiments/' + experiment + '/' + model

with open(os.path.join(path, 'results.txt'), 'r') as fp:    
    lines = fp.readlines()
    
    latex = ''
    
    for line in lines[1:] :
        split = line.split('\t')
        
        latex += split[0].replace('_', '\\_') + ' & ' + split[1] + ' & ' + split[2] + ' & ' + split[3] + ' \\\\ \\hline \n' 
        

"""
		\hline
        HAS_CAUSE & 0.95 & 0.96 & 0.96 \\
        \hline
        GENUS & 0.63 & 0.66 & 0.64 \\
        \hline
        Average & 0.79 & 0.81 & 0.80 \\
        \hline
"""