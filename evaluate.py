import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
plt.rcParams['font.size'] = '17'

import config_util

def curve_precision(experiment, model, drop_def=False):
    df_true = pd.read_csv(os.path.join('data/experiments', experiment, 'test.csv'))
    df_anno = pd.read_csv(os.path.join('data/experiments', experiment, model, 'annotation.csv'))
    df_anno['Tag'] = df_anno['Tag'].apply(lambda x: x.lstrip('B-').lstrip('I-'))
    entities = []
    entity_id = -1
    for sentence_id in df_anno['Sentence'].unique():
        sentence = df_anno[df_anno['Sentence'] == sentence_id]
        last_tag = None
        for row in sentence.iterrows():
            add_entity = False
            row_id = row[0]
            vals = row[1]
            tag = vals['Tag']
            if tag != 'O':
                if tag != last_tag:
                    entity_id += 1
                add_entity = True
                last_tag = tag
            else:
                last_tag = None

            if add_entity:
                entities.append((*vals, entity_id))
            else:
                entities.append((*vals, None))
    df_true_new = pd.DataFrame.from_records(entities, columns=['Sentence', 'Word', 'Tag', 'Entity'])
    df_true_new['New'] = df_true['Tag']
    df_true_new['New'] = df_true_new['New'].apply(lambda x: x.lstrip('B-').lstrip('I-'))

    tags = []
    fig = plt.figure(figsize=(10, 8))
    for tag in sorted(df_true_new['Tag'].unique()):
        cond = False
        if drop_def:
            cond = tag == 'DEFINIENDUM'
        if tag == 'O' or cond:
            continue
        tags.append(tag)
        percents = df_true_new[df_true_new['Tag'] == tag].groupby('Entity').apply(
            lambda x: sum(x['Tag'] == x['New']) / len(x)).to_numpy()
        percents = np.sort(percents)
        perc_unique = np.unique(percents)
        x = [0]
        y = [1]
        for perc in perc_unique:
            x.append(perc)
            y.append(np.mean(percents >= perc))
        plt.plot(x, y)
        print(f'{tag}: precision AUC: {auc(x, y)}')
    plt.xlabel('Threshold', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.legend(tags, prop={'size': 13}, loc="lower left")

    plt.grid()
    plt.title('Precision curves')
    plt.ylim(0, 1.1)
    plt.savefig('precision.png', bbox_inches='tight')


    return entities


def curve_recall(experiment, model, drop_def=False):
    df_true = pd.read_csv(os.path.join('data/experiments', experiment, 'test.csv'))
    df_anno = pd.read_csv(os.path.join('data/experiments', experiment, model, 'annotation.csv'))

    entities = []
    entity_id = -1
    for sentence_id in df_true['Sentence'].unique():
        sentence = df_true[df_true['Sentence'] == sentence_id]

        for row in sentence.iterrows():
            new_entity = False
            same_entity = False
            row_id = row[0]
            vals = row[1]
            if vals['Tag'].startswith('B-'):
                new_entity = True
            elif vals['Tag'].startswith('I-'):
                same_entity = True
            if new_entity:
                entity_id += 1
            if new_entity or same_entity:
                entities.append((*vals, entity_id))
            else:
                entities.append((*vals, None))
    df_true_new = pd.DataFrame.from_records(entities, columns=['Sentence', 'Word', 'Tag', 'Entity'])
    df_true_new['Anno'] = df_anno['Tag']
    df_true_new['Anno'] = df_true_new['Anno'].apply(lambda x: x.lstrip('B-').lstrip('I-'))
    df_true_new['Tag'] = df_true_new['Tag'].apply(lambda x: x.lstrip('B-').lstrip('I-'))
    tags = []
    fig = plt.figure(figsize=(10, 8))
    for tag in sorted(df_true_new['Tag'].unique()):
        cond = False
        if drop_def:
            cond = tag == 'DEFINIENDUM'
        if tag == 'O' or cond:
            continue
        tags.append(tag)
        percents = df_true_new[df_true_new['Tag'] == tag].groupby('Entity').apply(lambda x: sum(x['Tag'] == x['Anno']) / len(x)).to_numpy()
        percents = np.sort(percents)
        perc_unique = np.unique(percents)
        x = [0]
        y = [1]
        for perc in perc_unique:
            x.append(perc)
            y.append(np.mean(percents >= perc))
        plt.plot(x, y)
        print(f'{tag}: recall AUC: {auc(x, y)}')
    plt.xlabel('Threshold', fontsize=25)
    plt.ylabel('Recall', fontsize=25)
    plt.legend(tags, prop={'size': 13}, loc="lower left")
    plt.grid()
    plt.title('Recall curves')
    plt.ylim(0, 1.1)
    # plt.show()
    plt.savefig('recall.png', bbox_inches='tight')





def main():
    exps = config_util.list_experiments()
    exp = exps[4]
    mods = config_util.list_models(exp)
    mod = mods[0]
    print(exp, mod)
    print()
    curve_precision(exp, mod, True)
    print()
    curve_recall(exp, mod, True)


if __name__ == '__main__':
    main()
