"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import spacy
import nltk


def process_file(fname):
    return [" ".join(elem.strip().replace("###", "").\
        replace("##", "").split()) for elem in open(fname, 'r')]


def parse_args():
    parser = argparse.ArgumentParser(description="NER Precision/ Recall Evaluation")
    parser.add_argument("--source_doc", default="data/xsum/train.source", \
        help="Source Articles")
    parser.add_argument("--target_summary", default="data/xsum/train.target", \
        help="Target Summaries")
    parser.add_argument("--predict_summary", default="data/xsum/train.target", \
        help="Predicted Summaries")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_lg")
    nltk.download('stopwords')
    sws = set(nltk.corpus.stopwords.words('english'))
    args = parse_args()
    text_target = process_file(args.target_summary)
    text_source = process_file(args.source_doc)
    text_predict = process_file(args.predict_summary)
    assert len(text_target) == len(text_predict) == len(text_source)
    print("Total Samples: {0} and {1} and {2}".format(len(text_target), \
        len(text_predict), len(text_source)))

    docs_target = nlp.pipe(text_target)
    docs_source = nlp.pipe(text_source)
    docs_predict = nlp.pipe(text_predict)

    tot_prd_micro, tp_prd_src_micro, tp_prd_tgt_micro, tot_tgt_micro = 0., 0., 0., 0.
    tgt_macro_p, tgt_macro_r, tgt_macro_f, src_macro_p = 0., 0., 0., 0.

    for tgt, src, prd in zip(docs_target, docs_source, docs_predict):
        target_entity = set([x.text.lower() for x in tgt if x.ent_type_ \
            != '' and x.text.lower() not in sws])
        source_entity = set([x.text.lower() for x in src if x.ent_type_ \
            != '' and x.text.lower() not in sws])
        predict_entity = set([x.text.lower() for x in prd if x.ent_type_ \
            != '' and x.text.lower() not in sws])
        src_overlap = len(source_entity.intersection(predict_entity))
        tgt_overlap = len(target_entity.intersection(predict_entity))

        tot_prd_micro += len(predict_entity)
        tot_tgt_micro += len(target_entity)
        tp_prd_tgt_micro += tgt_overlap
        tp_prd_src_micro += src_overlap

        macro_p = tgt_overlap/(0.0001+len(predict_entity))
        macro_r = tgt_overlap/(0.0001+len(target_entity))
        tgt_macro_f += 2*macro_p*macro_r/(0.0001+(macro_r+macro_p))
        tgt_macro_p += macro_p
        tgt_macro_r += macro_r
        src_macro_p += src_overlap/(0.0001+len(predict_entity))

    micro_tgt_rec = tp_prd_tgt_micro/tot_tgt_micro
    micro_tgt_prec = tp_prd_tgt_micro/tot_prd_micro
    micro_src_prec = tp_prd_src_micro/tot_prd_micro

    print(f'File: {args.predict_summary},'
          f'Micro: Target P {micro_tgt_prec}, R {micro_tgt_rec}, '
          f'F1 {2*micro_tgt_prec*micro_tgt_rec/(micro_tgt_prec+micro_tgt_rec)}; '
          f'Source P {micro_src_prec} | Macro: Target P {tgt_macro_p/len(text_target)}, '
          f'R {tgt_macro_r/len(text_target)}, F1 {tgt_macro_f/len(text_target)}; '
          f'Source P {src_macro_p/len(text_target)}, \
            #OVERLAPPING ENTITY WITH SOURCE {tp_prd_src_micro}')
