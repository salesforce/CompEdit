"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import random
from collections import defaultdict
import argparse
import spacy

random.seed(0)
nlp = spacy.load("en_core_web_lg")


def process_file(fname):
    return [" ".join(elem.strip().replace("###", "").replace("##", "").split()) for elem in open(fname, 'r')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process src/tgt for \
        perturbation with sentence expansion model")
    parser.add_argument("--folder", type=str, default="./")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--output_fn", type=str, default=None, \
            help="Path to jsonl file that will store input to generate perturbed examples")

    args = parser.parse_args()

    fname_target = f"{args.folder}/{args.subset}.target"
    fname_source = f"{args.folder}/{args.subset}.source"

    text_target = process_file(fname_target)
    text_source = process_file(fname_source)

    docs_target = nlp.pipe(text_target,
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

    docs_source = nlp.pipe(text_source,
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

    targets = list(docs_target)
    sources = list(docs_source)

    with open(args.output_fn, "w") as out:
        for source, target in zip(sources, targets):
            dont_add = set()
            labs = set()
            for ent in target.ents:
                lab = ent.label_
                dont_add.add(ent.text)
                labs.add(lab)

            ent2label = {}
            label2ents = defaultdict(set)
            source_ents = set()
            for ent in source.ents:
                lab = ent.label_
                ent2label[ent.text] = lab
                label2ents[lab].add(ent.text)
                # and lab in labs
                if ent.text not in dont_add:
                    source_ents.add(ent.text)
            source_ents = list(source_ents)
            sources = []
            for i in range(1, 4):
                try:
                    cur_source_ents = random.choices(source_ents, k=i)
                    cur_source_ents_str = " | ".join(cur_source_ents)
                    new_source = f"{target.text} || {cur_source_ents_str}"
                    sources.append(new_source)
                except:
                    continue
            cur = {"source": source.text, "target": target.text, "perturber_input": sources}
            json.dump(cur, out)
            out.write("\n")
