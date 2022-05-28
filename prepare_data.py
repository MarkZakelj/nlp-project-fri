import math
from pathlib import Path
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

HIERARCHICAL_TAGS = ['DEFINIENDUM', 'GENUS', 'DEFINITOR']
NON_HIERARCHICAL_TAGS = ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM', 'HAS_FUNCTION', 'HAS_SIZE']

experiment_config = [
    {'name': 'def+gen+definitor',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS', 'DEFINITOR'],
     'non-hierarchical': [],
     'non-hierarchical-definitor': [],
     'B-tags': True},
    {'name': 'def+gen+definitor',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS', 'DEFINITOR'],
     'non-hierarchical': [],
     'non-hierarchical-definitor': [],
     'B-tags': True},

    {'name': 'def+gen',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS'],
     'non-hierarchical': [],
     'non-hierarchical-definitor': [],
     'B-tags': False},
    
    {'name': 'def+gen',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS'],
     'non-hierarchical': [],
     'non-hierarchical-definitor': [],
     'B-tags': False},
    
    {'name': 'def',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': [],
     'non-hierarchical-definitor': [],
     'B-tags': False},
    
    {'name': 'def',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': [],
     'non-hierarchical-definitor': [],
     'B-tags': False},


    {'name': 'top4nonhier+def',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM'],
     'non-hierarchical-definitor': [],
     'B-tags': True},
    
    {'name': 'top4nonhier+def',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM'],
     'non-hierarchical-definitor': [],
     'B-tags': True},

    {'name': 'nonhier+def',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM', 'HAS_FUNCTION', 'HAS_SIZE'],
     'non-hierarchical-definitor': [],
     'B-tags': False},
]

ALLOWED_LANGUAGES = ['EN', 'SL', 'HR']


def get_language(filename, check=True):
    lang = Path(filename).stem[-2:]
    if check:
        if lang not in ALLOWED_LANGUAGES:
            raise ValueError(f"""File doesnt end with 2-character code of one of allowed languages.
                                     Allowed languages are: {ALLOWED_LANGUAGES}""")
    return lang


def prepare_dataframe(dataframe: pd.DataFrame, hier_cols: list, non_hier_cols: list, non_hier_def_cols: list, btags=True) -> pd.DataFrame:
    """merge hierarchical, non-hierarchical and non-hierarchical-definitor columns into one Tag column"""
    df = dataframe.copy()
    # set unwanted tags to NaN
    df.loc[~df['hierarchical'].isin(hier_cols), ['hierarchical']] = np.nan
    df.loc[~df['non-hierarchical'].isin(non_hier_cols), ['non-hierarchical']] = np.nan
    df.loc[~df['non-hierarchical-definitor'].isin(non_hier_def_cols), ['non-hierarchical-definitor']] = np.nan

    def merge_tags(hier_tag, non_hier_tag, non_hier_def_tag):
        # if the hierarchical tag is NaN, return non-hierarchical-definitor tag.
        # If non-hierarchical-definitor tag is NaN, return non-hierarchical tag, even if its NaN.
        # Priority: hierarchical --> non-hierarchical-definitor --> non-hierarchical
        if type(hier_tag) == float and math.isnan(hier_tag):
            if type(non_hier_def_tag) == float and math.isnan(non_hier_def_tag):
                return non_hier_tag
            return non_hier_def_tag
        return hier_tag

    df['Tag'] = df.apply(lambda x: merge_tags(x['hierarchical'], x['non-hierarchical'], x['non-hierarchical-definitor']), axis=1)
    df = df[['Sentence', 'Word', 'Tag']]
    df_cpy = df.copy()

    if btags:
        # add B- and I- prefixes to tags
        last_tag = None
        for row in df.iterrows():
            tag = row[1]['Tag']
            if type(tag) == str:
                prefix = 'B-'
                if tag == last_tag:
                    prefix = 'I-'
                df_cpy.loc[row[0], 'Tag'] = prefix + tag
                last_tag = tag
            else:
                last_tag = None
    # set Nan values is Tag column to 'O' - Other
    df_cpy.loc[df['Tag'].isna(), ['Tag']] = 'O'
    return df_cpy


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
    df_with_tag = prepare_dataframe(df, config['hierarchical'], config['non-hierarchical'],
                                    config['non-hierarchical-definitor'], config['B-tags'])
    btag_name = "_btag" if config['B-tags'] else ''
    experiment_name = f"{language}_{config['name']}{btag_name}"
    # create experiment dir if it doesnt exsist
    Path('data', 'experiments', experiment_name).mkdir(parents=True, exist_ok=True)
    out_filename = 'test.csv' if as_test else 'train.csv'
    df_with_tag.to_csv(os.path.join('data', 'experiments', experiment_name, out_filename), index=False)


def main():
    for conf in tqdm(experiment_config):
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
