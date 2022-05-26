import math
from pathlib import Path
import numpy as np
import pandas as pd
import os

experiment_config = [
    {'name': 'def+gen+definitor',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS', 'DEFINITOR'],
     'non-hierarchical': [],
     'B-tags': True},
    {'name': 'def+gen',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS'],
     'non-hierarchical': [],
     'B-tags': True},
    {'name': 'top4nonhier+def',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM'],
     'B-tags': True},
    {'name': 'nonhier+def',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM', 'HAS_FUNCTION', 'HAS_SIZE'],
     'B-tags': True},
    {'name': 'has-form',
     'train': 'data/full_data_EN.csv',
     'test': 'data/full_data_new_EN.csv',
     'hierarchical': [],
     'non-hierarchical': ['HAS_FORM'],
     'B-tags': True},

    {'name': 'def+gen+definitor',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS', 'DEFINITOR'],
     'non-hierarchical': [],
     'B-tags': True},
    {'name': 'def+gen',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM', 'GENUS'],
     'non-hierarchical': [],
     'B-tags': True},
    {'name': 'top4nonhier+def',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': ['DEFINIENDUM'],
     'non-hierarchical': ['HAS_CAUSE', 'HAS_LOCATION', 'HAS_FORM', 'COMPOSITION_MEDIUM'],
     'B-tags': True},
    {'name': 'has-form',
     'train': 'data/full_data_SL.csv',
     'test': 'data/full_data_new_SL.csv',
     'hierarchical': [],
     'non-hierarchical': ['HAS_FORM'],
     'B-tags': True},
    
    {'name': 'def+gen',
     'train': 'data/full_data_HR.csv',
     'test': '',
     'hierarchical': ['DEFINIENDUM', 'GENUS'],
     'non-hierarchical': [],
     'B-tags': True},
]



ALLOWED_LANGUAGES = ['EN', 'SL', 'HR']


def get_language(filename, check=True):
    lang = Path(filename).stem[-2:]
    if check:
        if lang not in ALLOWED_LANGUAGES:
            raise ValueError(f"""File doesnt end with 2-character code of one of allowed languages.
                                     Allowed languages are: {ALLOWED_LANGUAGES}""")
    return lang


def prepare_dataframe(dataframe: pd.DataFrame, hier_cols: list, non_hier_cols: list, btags=True) -> pd.DataFrame:
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
    df = df[['Sentence', 'Word', 'Tag']]
    df_cpy = df.copy()

    in_region = False
    for row in df.iterrows():
        tag = row[1]['Tag']
        if type(tag) == str:
            prefix = 'B-'
            if in_region:
                prefix = 'I-'
            df_cpy.loc[row[0], 'Tag'] = prefix + tag
            in_region = True
        else:
            in_region = False
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
    df_with_tag = prepare_dataframe(df, config['hierarchical'], config['non-hierarchical'], config['B-tags'])
    experiment_name = language + "_" + config['name']
    # create experiment dir if it doesnt exsist
    Path('data', 'experiments', experiment_name).mkdir(parents=False, exist_ok=True)
    out_filename = 'test.csv' if as_test else 'train.csv'
    df_with_tag.to_csv(os.path.join('data', 'experiments', experiment_name, out_filename), index=False)


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
