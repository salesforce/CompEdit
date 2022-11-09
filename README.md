# Exploring Neural Models for Query-Focused Summarization

This is the official code repository for [Improving Factual Consistency in Summarization with Compression-Based Post-Editing](TBD)
by [Alexander R. Fabbri](https://twitter.com/alexfabbri4), [Prafulla Choubey](https://sites.google.com/view/prafulla-choubey/), [Jesse Vig](https://twitter.com/jesse_vig), [Chien-Sheng Wu](https://twitter.com/jasonwu0731), and
[Caiming Xiong](https://twitter.com/caimingxiong). 

We present code and instructions for running inference from the models introduced in the paper.

## Table of contents
- [Introduction](#introduction)
- [Perturber](#perturber)
- [Post-Editor](#post-editor)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Introduction
Post-editing is a model-agnostic approach to improve summary quality. In our paper, we propose a compression-based method that post-edits summaries to improve factual consistency.  
Our model takes as input the source document along with an initial summary with entities not found in the source according to named-entity recognizition marked with special tokens.  
The model produces a compressed output with these entities removed, improving entity precision with respect to the input by up to 25\% while retaining informativeness. 

</br> 

## Perturber

For training the perturber model, we use the data from the paper [Overcoming the Lack of Parallel Data in Sentence Compression](https://aclanthology.org/D13-1155.pdf) found [here](https://github.com/google-research-datasets/sentence-compression).

We provide the script `./preprocess_perturber.py` for processing this data in a format suitable for model training.  
Examples are filtered such that the compressed sentence is at least 75\% of the length of the uncompressed sentence.  
We provide the script `./train.sh` for model training, which makes use of the following files: [train.py](https://github.com/salesforce/query-focused-sum/blob/master/multiencoder/train.py) and [select_checkpoints.py](https://github.com/salesforce/query-focused-sum/blob/master/multiencoder/select_checkpoints.py).  
Our trained perturber checkpoint is found [here](TBD).



### Applying the perturber for post-editor training
We provide the script `./preprocess_posteditor.py` that takes in a folder containing `{subset}.source` and `{subset}.target` dataset files and prepares the input to apply the perturber on.  
The data should first be filtered according to entity precision (see below).

Then, run `./generate.py` on the output of this preprocessing step to produce training data for the post-editor. 

</br> 

## Post-Editor

The script `truncate_csv.py` should be run on the above csv file to ensure that model summaries that we learn to post-edit are not truncated. 

The above `./train.sh` can be used for training the post-editor on the output of the above scripts. Our trained post-editor checkpoint is found [here](TBD).

</br> 

## Pretrained Model Usage

The code below from `run.py` shows how to use the pretrained models:
```python
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
```

</br> 

## Evaluation

See `./entity_score.py` for entity precision and recall calculations.

</br> 

## Citation

</br> 

## License

This repository is released under the [BSD-3 License](LICENSE.txt).





