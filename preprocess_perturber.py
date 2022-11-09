"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import csv
import os
import glob
import random
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_lg")

random.seed(0)

threshold = 0.75

#location of uncompressed files from https://github.com/google-research-datasets/sentence-compression/data
#TODO
DIR=""

with open(os.path.join(DIR, "train.csv"), "w") as outt, open(os.path.join(DIR, "val.csv"), "w") as outv:
    writert = csv.DictWriter(outt, fieldnames=["text", "summary"])
    writert.writeheader()

    writerv = csv.DictWriter(outv, fieldnames=["text", "summary"])
    writerv.writeheader()
    datas = []

    # after gunzip'ing the files
    for fname in glob.iglob(f"{DIR}/*.json"):
        file_data = open(fname).read().split("\n\n")
        file_data = [x for x in file_data if len(x.strip()) > 0]
        for ex in tqdm(file_data, total=len(file_data)):
            data = json.loads(ex)
            comp_sent = data['compression']['text']
            sent = data['graph']['sentence']
            ratio = data['compression_ratio']
            if ratio > threshold:
                doc = nlp(sent)
                ents = [x.text for x in doc.ents]
                missing_ents = []
                for ent in ents:
                    # here did string match different from entity precision calculation
                    if ent.lower() not in comp_sent.lower():
                        missing_ents.append(ent)
                if missing_ents:
                    missing_ents = list(dict.fromkeys(missing_ents))
                    missing_ents_str = " | ".join(missing_ents)
                    text = f"{comp_sent} <s> {missing_ents_str}".strip()
                    summary = sent.strip()
                    datas.append({"text": text, "summary": summary})

    idxs = list(range(len(datas)))
    random.shuffle(idxs)
    train = [datas[x] for x in idxs[:-500]]
    val = [datas[x] for x in idxs[-500:]]
    for ex in train:
        writert.writerow(ex)
    for ex in val:
        writerv.writerow(ex)
