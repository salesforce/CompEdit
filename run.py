"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from generate import get_gen

sentence_to_perturb = "It was the season in which Chelsea played by their own record book. || 18 | the Champions League"
SRC = "The Blues topped the table while, at the other end, Sunderland could not escape the drop after 110 straight days at the bottom. Records tumbled through to the last day of the campaign, when we saw 33 goalscorers, more than ever before in a single day of a 38-game season. Goals scored from outside the penalty area fell to a Premier League low of 11.6% so, if you like a goalmouth scramble, this was your year."
sentence_to_postedit = f"{SRC} </s> It was the ## 18 ## th consecutive season in which Chelsea played by their own record book in ## the Champions League ##."

perturb_model_name = "PATH_TO_PERTURBER"
posteditor_model_name = "PATH_TO_POSTEDITOR"
gen_args = {"max_enc_len": 1024, "device": "cuda", "length_penalty": 1.0, "num_beams": 6, "min_gen_len": 10, "max_gen_len": 60}

perturber_tok = AutoTokenizer.from_pretrained(perturb_model_name)
perturber_model = AutoModelForSeq2SeqLM.from_pretrained(perturb_model_name).to(gen_args["device"])

posteditor_tok = AutoTokenizer.from_pretrained(posteditor_model_name)
posteditor_model = AutoModelForSeq2SeqLM.from_pretrained(posteditor_model_name).to(gen_args["device"])

perturbed_output = get_gen([sentence_to_perturb], perturber_model, perturber_tok, gen_args)[0]
print(perturbed_output)
#It was the 18th season in which Chelsea played by their own record book in the Champions League.

postedited_output = get_gen([sentence_to_postedit], posteditor_model, posteditor_tok, gen_args)[0]
print(postedited_output)
#It was the season in which Chelsea played by their own record book.
