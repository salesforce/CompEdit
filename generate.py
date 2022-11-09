"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import csv
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

def get_gen(inputs, model, tokenizer, args_dict):
    inputs_tok = tokenizer(inputs, max_length=args_dict['max_enc_len'],  \
        truncation=True, padding="max_length", return_tensors="pt")
    summaries = model.generate(input_ids=inputs_tok["input_ids"].to(args_dict['device']),
        attention_mask=inputs_tok["attention_mask"].to(args_dict['device']),
        length_penalty=args_dict['length_penalty'], num_beams=args_dict['num_beams'], \
        max_length=args_dict['max_gen_len'], min_length=args_dict['min_gen_len'])
    decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, \
        clean_up_tokenization_spaces=True) for s in summaries]
    decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
    return decoded_summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Decode from checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_enc_len", type=int, default=1024)
    parser.add_argument("--min_gen_len", type=int, default=11)
    parser.add_argument("--num_beams", type=int, default=6)
    parser.add_argument("--max_gen_len", type=int, default=60)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_fn", type=str, default=None, \
            help="Path to input csv(for summary generation)/jsonl(for perturbed output generation).")
    parser.add_argument("--output_fn", type=str, default=None, \
            help="Path to output")
    parser.add_argument("--model_ckpt", type=str, default=None, \
            help="Path to model checkpoint")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_ckpt).to(args.device)

    with open(args.input_fn) as f, open(args.output_fn, "w") as out:
        inputs = []
        all_sources = []
        all_targets = []
        if args.input_fn[-4:] == ".csv":
            reader = csv.DictReader(f)
            for row in reader:
                text = row['text'].strip()
                inputs.append(text)

        elif args.input_fn[-6:] == ".jsonl":
            for line in f:
                data = json.loads(line)
                cur_input = data["perturber_input"]
                inputs.extend(cur_input)
                all_sources.extend([data["source"]] * len(cur_input))
                all_targets.extend([data["target"]] * len(cur_input))


        input_batches = list(chunks(inputs, args.batch_size))
        all_summaries = []
        for input_batch in tqdm(input_batches, total=len(input_batches)):
            decoded_summaries = get_gen(input_batch, model, tokenizer, vars(args))
            decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
            for summary in decoded_summaries:
                summary = summary.replace("\n", " ").strip()
                out.write(summary + "\n")
                all_summaries.append(summary)

        if all_sources:
            output_fn = args.output_fn + ".perturbed.jsonl"
            output_fn_csv = args.output_fn + ".perturbed.csv"
            with open(output_fn, "w") as outf, open(output_fn_csv, "w") as outcsv:
                writer = csv.DictWriter(outcsv, fieldnames=["text", "summary"])
                writer.writeheader()
                sums = set()
                for source, target, summ, model_input in \
                        zip(all_sources, all_targets, all_summaries, inputs):

                    cur_ents = [x.strip() for x in model_input.split("||")[-1].split("|")]
                    cur_sum = summ
                    for ent in cur_ents:
                        cur_sum = cur_sum.replace(ent, f" ## {ent} ## ")
                        cur_sum = " ".join(cur_sum.split())
                    if cur_sum in sums:
                        continue
                    sums.add(cur_sum)

                    cur_json = {"source": source, "target_orig": target, "perturber_output_annot": \
                        cur_sum, "perturber_output": summ, "perturber_input": model_input}
                    json.dump(cur_json, outf)
                    outf.write("\n")

                    source_str = f"{source}</s> {cur_sum}".replace("\n", " ")
                    cur_csv = {"text": source_str, "summary": target}
                    writer.writerow(cur_csv)
