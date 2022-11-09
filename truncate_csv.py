"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import csv
from tqdm import tqdm
from transformers import BartTokenizer

tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

fname = sys.argv[1]
with open(fname) as f, open(fname.replace(".csv", ".tok_trunc.csv"), "w") as out:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(out, fieldnames=["text", "summary"])
    writer.writeheader()
    for row in tqdm(reader, desc='Processing'):
        splits = row['text'].split("</s>")
        source = splits[0].strip()
        bart_summ = splits[1].strip()
        summary = row['summary']

        source_toks = tok(source)['input_ids']
        to_change_toks = tok(bart_summ)['input_ids']
        total_len = len(source_toks) + len(to_change_toks)
        if total_len > 1022:
            num_change = abs(1022 - total_len)
            source_truncated = source_toks[:-num_change]
            source_final = tok.decode(source_truncated, skip_special_tokens=True, \
                clean_up_tokenization_spaces=True).replace("<n>", "").strip()
        else:
            source_final = source
        text = f"{source_final}</s> {bart_summ}"
        row = {"text": text, "summary": summary}
        writer.writerow(row)
