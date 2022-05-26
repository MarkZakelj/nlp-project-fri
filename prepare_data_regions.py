# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:41:59 2022

@author: Kert PC
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import os

experiment_config = [

    {'name': 'nonhier+def',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM', 'HAS_FUNCTION', 'HAS_SIZE']}
]


ALLOWED_LANGUAGES = ['EN', 'SL']


def get_language(filename, check=True):
    lang = Path(filename).stem[-2:]
    if check:
        if lang not in ALLOWED_LANGUAGES:
            raise ValueError(f"""File doesnt end with 2-character code of one of allowed languages.
                                     Allowed languages are: {ALLOWED_LANGUAGES}""")
    return lang


def prepare_dataframe(dataframe: pd.DataFrame, hier_cols: list, non_hier_cols: list) -> pd.DataFrame:
    """merge hierarchical column and non-hierarchical column into one Tag column"""
    df = dataframe.copy()
    # set unwanted tags to NaN
    df.loc[~df['hierarchical'].isin(hier_cols), ['hierarchical']] = np.nan
    df.loc[~df['non-hierarchical'].isin(non_hier_cols), ['non-hierarchical']] = np.nan

    def merge_tags(hier_tag, non_hier_tag):
        # if the hierarchical tag is NaN, return non-hierarchical tag (even if NaN)
        if type(hier_tag) == float and math.isnan(hier_tag):
            return non_hier_tag
        return hier_tag

    df['Tag'] = df.apply(lambda x: merge_tags(x['hierarchical'], x['non-hierarchical']), axis=1)
    df = df.drop(columns=['hierarchical', 'non-hierarchical'])
    # set Nan values is Tag column to 'O' - Other
    df.loc[df['Tag'].isna(), ['Tag']] = 'O'
    return df


def prepare_regions(dataframe: pd.DataFrame):
    dfs = []
    keys = []
    
    agg_func = lambda s: [[w, t] for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
    df_grouped = dataframe.groupby("Sentence").apply(agg_func)
    
    
    for _, sentence in df_grouped.items():
        dfds = []
        regs = []
        
        start = -1
        end = -1
        typ = None
        
        for i, word in enumerate(sentence) :
            if start == -1 :
                if word[1] != 'O' :
                    start = i
                    typ = word[1]
            else :
                if word[1] != typ :
                    end = i
                    
                    nu_reg = {'start' : start, 'end' : end, 'type' : typ}
                    if typ == 'DEFINIENDUM' :
                        dfds.append(nu_reg)
                    else :
                        regs.append(nu_reg)
                    
                    start = -1
                    end = -1
                    typ = None
        
        for dfd in dfds :
            for reg in regs :
                nu_sent = sentence.copy()
                
                if dfd['end'] < reg['start'] :
                    nu_sent.insert(reg['end'], ['</e2>', reg['type']])
                    nu_sent.insert(reg['start'], ['<e2>', reg['type']])
                    
                    nu_sent.insert(dfd['end'], ['</e1>', 'DEFINIENDUM'])
                    nu_sent.insert(dfd['start'], ['<e1>', 'DEFINIENDUM'])
                else :
                    nu_sent.insert(dfd['end'], ['</e1>', 'DEFINIENDUM'])
                    nu_sent.insert(dfd['start'], ['<e1>', 'DEFINIENDUM'])
                    
                    nu_sent.insert(reg['end'], ['</e2>', reg['type']])
                    nu_sent.insert(reg['start'], ['<e2>', reg['type']])
                
                tsv_row = [reg['type'], '']
                
                for word in nu_sent :
                    tsv_row[1] += word[0] + ' '
                
                tsv_row[1] = tsv_row[1].strip()
                dfs.append(pd.DataFrame([tsv_row], columns=['relation', 'sentence']))
                keys.append(tsv_row[0])

    df = pd.concat(dfs)

    return df, keys

def prepare_experiment(config: dict, as_test=False):
    """
    Create experiment folder based on language and experiment configuration
    :param full_dataframe_path: path to the .csv file containing all possible tags
    :param config: config dictionary specifying experiment name and which tags to keep
    :param as_test: if true, save the resulting dataframe as test.csv, otherwise save it as train.csv
    :return:
    """
    full_dataframe_path = config['train'] if not as_test else config['test']
    language = get_language(full_dataframe_path)
    df = pd.read_csv(full_dataframe_path)
    df_with_tag = prepare_dataframe(df, config['hierarchical'], config['non-hierarchical'])
    experiment_name = language + "_reg_" + config['name']
    
    df_with_reg, reg_keys = prepare_regions(df_with_tag)
    # create experiment dir if it doesnt exsist
    Path('data', 'experiments', experiment_name).mkdir(parents=True, exist_ok=True)
    out_filename = 'test.tsv' if as_test else 'train.tsv'
    df_with_reg.to_csv(os.path.join('data', 'experiments', experiment_name, out_filename), sep="\t", index=False, header=False)
    
    with open(os.path.join('data', 'experiments', experiment_name, 'answer_keys.txt'), 'w') as fp:
        for i, key in enumerate(reg_keys):
            fp.write("%s\n" % (str(8001+i) + '\t' + key))
            
            
    with open(os.path.join('data', 'experiments', experiment_name, 'labels.txt'), 'w') as fp:    
        fp.write("%s\n" % 'Other')
        for rel_tag in config['non-hierarchical'] :
            fp.write("%s\n" % rel_tag)
    

def main():
    for conf in experiment_config:
        # prepare train data
        prepare_experiment(conf, as_test=False)
        if conf['test']:
            if conf['train'] == conf['test']:
                raise NameError(
                    """Both train and test files are the same.
                       If you dont need the test set, set the test key in config to None or empty string""")
            # check if its the same language on the train set
            if get_language(conf['train']) != get_language(conf['test']):
                raise ValueError('Language of train and test set in the same experiment must be the same')
            prepare_experiment(conf, as_test=True)


if __name__ == '__main__':
    main()
