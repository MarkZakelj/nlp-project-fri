"""
Convert webanno .tsv annotations to csv files with four columns:
- Sentence: unique number for each sentence
- Word: word (token) in the sentence represented as string
- hierarchical: hierarchical tag. One of ['DEFINIENDUM', 'DEFINITOR', 'GENUS']
- non-hierarchical: non-hierarchical tag. One of many. (check in EDA.ipynb file)
"""

import pandas as pd
import os
import csv
import re
import copy
import numpy as np
import time
import argparse
from collections import defaultdict, namedtuple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


COLNAME2POS = {val: k for k, val in enumerate(
    ['TOKENID', 'POSITION', 'TOKEN', 'CANONICAL', 'CATEGORY', 'DEF_ELEMENT', 'RELATION', 'REL_VERB_FRAME'])}


def uncapitalize(s):
    res = s[:1].lower() + s[1:] if s else ''
    return res


def read_sentences(fname):
    '''
    Reads Webanno csv and splits into smaller tables according to "sentence_index-token_index" column.
    Returns a list of DataFrame objects.
    '''
    table = pd.read_csv(fname, sep='\t', quoting=csv.QUOTE_NONE, comment='#', header=None, na_values='_', index_col=0,
                        encoding='utf-8')
    table = table.replace('\xa0', ' ', regex=True)
    table.drop([colid for colid in table.columns if table[colid].isnull().all()], inplace=True,
               axis='columns')  # drop empty columns
    if len(table.columns) + 1 < len(COLNAME2POS):
        raise SyntaxError('Invalid number of columns in file {}'.format(fname))
    groups = [pd.DataFrame(data) for gid, data in table.groupby(by=lambda s: int(s.split('-')[0]))]
    return groups


def read_data(path, extensions=None):
    '''
    Reads the data from Webanno csv and returns a list where every element is a dictionary representing one annotated sentence.
    '''
    # default extensions to read if path is directory
    if extensions is None:
        extensions = ['.tsv', '.csv']

    nfinished = 0
    errors = []
    if os.path.isfile(path):
        groups = read_sentences(path)
        nfinished += 1
    elif os.path.isdir(path):
        with os.scandir(path) as it:
            groups = []
            for entry in it:
                if entry.is_file() and not entry.name.startswith('.') \
                        and os.path.splitext(entry.name)[1].lower() in extensions:
                    try:
                        groups.extend(read_sentences(os.path.join(path, entry.name)))
                        print(entry.name)
                    except:
                        errors.append(os.path.join(path, entry.name))
                    else:
                        nfinished += 1

    else:
        raise IOError('The input must be a file or folder')

    print('{} file(s) loaded.'.format(nfinished))
    if errors:
        print('{} file(s) not loaded due to errors:\n\t{}'.format(len(errors), '\n\t'.join(errors)))

    # first, expand all cells with multiple values separated with "|" into new rows
    datalines = []
    new_groups = []
    for g in groups:
        new_rows = []
        for colid in g.columns[3:]:  # in first four columns this cannot happen
            ismulti = g[colid].str.contains('|', na=False, regex=False)
            multirows = g[ismulti]
            if not multirows.empty:
                for idx, row in multirows.iterrows():
                    values = [x.strip() for x in row[colid].split('|')]
                    g[colid].loc[idx] = values[0]

                    # # print('--->', values)
                    # This is commented out to ommit duplicated words
                    # for i, val in enumerate(values[1:]):
                    #     # create a new row which only contains this column and row index (name), everything else is NaN
                    #     new = pd.Series(index=row.index, dtype=np.object, name=row.name + '-{}-{}'.format(colid, i + 1))
                    #     new[colid] = val
                    #     new[COLNAME2POS['TOKEN']] = row[COLNAME2POS['TOKEN']]
                    #     new_rows.append(new)
        g = g.append(new_rows)
        # g = pd.concat(g, new_rows)
        new_groups.append(g)
    groups = new_groups

    for g in groups:
        for colid in g.columns[3:]:
            nonempty = g[~g[colid].isna()]
            for idx, row in nonempty.iterrows():
                # print(colid, idx, g[colid].loc[idx], type(g[colid].loc[idx]))
                # print(g[colid].loc[idx])
                if not g[colid].loc[idx].endswith(']'):
                    g[colid].loc[idx] += '[{}]'.format(str(time.time()).replace('.', ''))
                    # print(colid, idx, g[colid].loc[idx])

    for g in groups:
        linedata = defaultdict(list)
        # build sentence from tokens
        sentence = ' '.join([x.strip() for x in g[COLNAME2POS['TOKEN']].values.tolist()])
        sentence = re.sub('[ ]+[ ]+', ' ', sentence)
        for ch in ',.:;!?)]}':
            sentence = sentence.replace(' ' + ch, ch)
        for ch in '([{':
            sentence = sentence.replace(ch + ' ', ch)
        linedata['SENTENCE'] = [sentence]

        for colid in [COLNAME2POS[x] for x in ['DEF_ELEMENT', 'RELATION', 'REL_VERB_FRAME']]:
            for val in list(g[colid].value_counts().index):
                rows = g.loc[g[colid] == val]
                string = ' '.join([x.strip() for x in rows[COLNAME2POS['TOKEN']].values.tolist() if x.strip()])
                val = re.sub('\[[0-9]+\]$', '', val)  # strip ending number in brackets
                val = val.replace('\\', '')
                linedata[val].append(string)

        # category is separate because it yields two columns
        col = 'CATEGORY'
        colid = COLNAME2POS[col]
        for val in list(g[colid].value_counts().index):
            rows = g.loc[g[colid] == val]
            string = ' '.join([x.strip() for x in rows[COLNAME2POS['TOKEN']].values.tolist() if x.strip()])
            val = re.sub('\[[0-9]+\]$', '', val)  # strip ending number in brackets
            val = val.replace('\\', '')
            # linedata[col].append((val, string))
            linedata[col].append(val)
            linedata['CATEGORY_TEXT'].append(string)

        datalines.append(linedata)
    return datalines, new_groups


def concatenate_groups(groups):
    # concatenate all term-frames into single pandas dataframe
    df_full = None
    for i, sentence in enumerate(groups):
        df = sentence.iloc[:, [1, 3, 4, 5, 6]].copy()
        df.loc[:, 1] = i
        if df_full is None:
            df_full = df
        else:
            df_full = pd.concat([df_full, df], axis=0)
    for col_num in [4, 5, 6, 7]:
        # get rid of '\' character in the tag and [<num>]
        df_full[col_num] = df_full[col_num].apply(
            lambda x: x[:x.find('[')].replace('\\', '').replace(' ', '_') if not type(x) == float else x)

    df_full = df_full.reset_index(drop=True).loc[:, [1, 2, 4, 5, 6, 7]]
    df_full.columns = ['Sentence', 'Word', 'category', 'hierarchical', 'non-hierarchical', 'non-hierarchical-definitor']
    return df_full


def main():
    # First batch of karst definitions:
    tsv_file_or_folder = 'data/Termframe/AnnotatedDefinitions/EN'
    datalines, groups = read_data(tsv_file_or_folder, extensions=['.tsv'])
    df_full = concatenate_groups(groups)
    df_full.to_csv('data/full_data_EN.csv', index=False)

    # First batch of karst definitions SL:
    tsv_file_or_folder = 'data/Termframe/AnnotatedDefinitions/SL'
    datalines, groups = read_data(tsv_file_or_folder, extensions=['.tsv'])
    df_full = concatenate_groups(groups)
    df_full.to_csv('data/full_data_SL.csv', index=False)

    # First batch of karst definitions SL:
    tsv_file_or_folder = 'data/Termframe/AnnotatedDefinitions/HR'
    datalines, groups = read_data(tsv_file_or_folder, extensions=['.tsv'])
    df_full = concatenate_groups(groups)
    df_full.to_csv('data/full_data_HR.csv', index=False)

    # New Definitions - used for testing
    tsv_file_or_folder = 'data/Termframe/NewDefinitions/en'
    datalines, groups = read_data(tsv_file_or_folder, extensions=['.tsv'])
    df_full = concatenate_groups(groups)
    df_full.to_csv('data/full_data_new_EN.csv', index=False)

    # New Definitions SL - used for testing
    tsv_file_or_folder = 'data/Termframe/NewDefinitions/sl'
    datalines, groups = read_data(tsv_file_or_folder, extensions=['.tsv'])
    df_full = concatenate_groups(groups)
    df_full.to_csv('data/full_data_new_SL.csv', index=False)


if __name__ == '__main__':
    main()
