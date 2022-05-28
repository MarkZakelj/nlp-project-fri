import pandas as pd
from transformers import BertTokenizer, BertPreTrainedModel, AdamW, AutoTokenizer, BertConfig, BertModel
from rbert_model import RBERT
import os
import numpy as np
from rbert_data_loader import load_and_cache_examples
from train_relation_extraction import RelationExtractorTrainer, get_tokenizer, model_id_to_path
from rbert_data_loader import TermFrameProcessor, convert_examples_to_features
import torch
from scipy.special import softmax
from seqeval.metrics import classification_report
import re
from tqdm import tqdm

def add_e1_tag(word, tag):
    if tag == 'DEFINIENDUM':
        return '<e1> '+word.strip()+' </e1>'
    return word.strip()

def main():
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
    model.to(device)
    model.eval()
    test_df = pd.read_csv(os.path.join(conf['data_dir'].replace('_reg', ''), 'test.csv'))
    test_df['SentSTRe1'] = test_df.apply(lambda x: add_e1_tag(x['Word'], x['Tag']), axis=1)
    sentence_e1 = test_df.groupby('Sentence')['SentSTRe1'].apply(lambda x: ' '.join(list(x)).replace('</e1> <e1>', ''))

    df_ann = None
    for sentence_idx in tqdm(sentence_e1.index):
        sentence = sentence_e1[sentence_idx]
        e1_occ = [m.start() for m in re.finditer('</e1>', sentence)]
        if len(e1_occ) > 1:
            sentence = sentence.replace('<e1> ', '', 1).replace('</e1> ', '', 1)

        pure_sentence = sentence.replace('<e1> ', '').replace('</e1> ', '').replace('  ', ' ').split(' ')

        i1 = sentence.find('<e1>')
        i2 = sentence.find('</e1>')
        words_before = sentence[:i1].strip().split(' ')
        words_inside = sentence[i1 + 5:i2].strip().split(' ')
        words_after = sentence[i2 + 5:].strip().split(' ')
        for wo in [words_before, words_inside, words_after]:
            if '' in wo:
                wo.remove('')
        word_class_scores = np.zeros((len(pure_sentence), len(processor.relation_labels)))
        for window_size in [1, 2]:
            lines = []
            word_masks = []
            idx1 = 0
            for i in range(len(words_before) - window_size + 1):
                idx2 = idx1 + window_size
                e2_before = words_before[:idx1] + ['<e2>'] + words_before[idx1:idx2] + ['</e2>'] + words_before[idx2:]
                lines.append(['Other', ' '.join(e2_before) + ' ' + sentence[i1:]])
                word_masks.append(list(range(idx1, idx2)))
                idx1 += 1
            idx1 = 0
            offset = len(words_before) + len(words_inside)
            for i in range(window_size, len(words_after)):
                idx2 = idx1 + window_size
                e2_after = words_after[:idx1] + ['<e2>'] + words_after[idx1:idx2] + ['</e2>'] + words_after[idx2:]
                lines.append(['Other', sentence[:i2 + 5] + ' ' + ' '.join(e2_after)])
                word_masks.append(list(range(idx1 + offset, idx2 + offset)))
                idx1 += 1
            examples = processor._create_examples(lines, 'train')
            features = convert_examples_to_features(
                examples, conf['max_length'], tokenizer, add_sep_token=False
            )

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)
            all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long).to(device)  # add e1 mask
            all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long).to(device)  # add e2 mask


            # for i in range(len(all_input_ids))
            with torch.no_grad():
                outputs = model(all_input_ids, all_attention_mask, all_token_type_ids, None, all_e1_mask, all_e2_mask)
                logits = outputs[0].detach().cpu().numpy()

            logits[logits < 7] = 0
            for idx in range(logits.shape[0]):
                word_class_scores[word_masks[idx]] += logits[idx, :] / window_size
            torch.cuda.empty_cache()
        res = [(sentence_idx, word, processor.relation_labels[np.argmax(score)]) for word, score in
               zip(pure_sentence, word_class_scores)]
        df = pd.DataFrame.from_records(res, columns=["Sentence", 'Word', 'Tag'])
        if df_ann is None:
            df_ann = df
        else:
            df_ann = pd.concat([df_ann, df], axis=0)
    df_ann.loc[df_ann['Tag'] == 'Other', 'Tag'] = 'O'
    results_pred = df_ann.groupby('Sentence')['Tag'].apply(list).values.tolist()
    results_true = test_df.groupby('Sentence')['Tag'].apply(list).values.tolist()
    print(classification_report(results_true, results_pred))
    results_true = [[e] for row in results_true for e in row]
    results_pred = [[e] for row in results_pred for e in row]
    report_tbt = classification_report(results_true, results_pred)
    print(report_tbt)
    # with open(os.path.join(experiment_dir, model_id_path, 'results_tbt.txt'), 'w') as fl:
    #     fl.write(report_tbt)

    # df_ann.to_csv('df_ann.csv', index=False)


if __name__ == '__main__':
    main()
