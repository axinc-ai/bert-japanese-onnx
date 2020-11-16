import torch
import numpy
import argparse

MODEL_LISTS = [
    'bert-base-cased',
    'bert-base-uncased',
    'cl-tohoku/bert-base-japanese',
    'cl-tohoku/bert-base-japanese-whole-word-masking',
    'cl-tohoku/bert-base-japanese-char-whole-word-masking',
]

parser = argparse.ArgumentParser(
    description='masklm proofreading sample'
)
parser.add_argument(
    '-i', '--input', metavar='VIDEO',
    default="terms.txt",
    help='The input video path.'
)
parser.add_argument(
    '-s', '--sudjest',
    action='store_true',
    help='Show sudjestion)'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='cl-tohoku/bert-base-japanese-whole-word-masking', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = parser.parse_args()

from transformers import BertTokenizer, BertJapaneseTokenizer, BertForMaskedLM
from onnxruntime import InferenceSession, SessionOptions, get_all_providers

is_english = args.arch=='bert-base-cased' or args.arch=='bert-base-uncased'
show_sudject = False

if is_english:
    tokenizer = BertTokenizer.from_pretrained(args.arch)
else:
    tokenizer = BertJapaneseTokenizer.from_pretrained(args.arch)

cpu_model = InferenceSession("onnx/"+args.arch)

import codecs

with codecs.open(args.input, 'r', 'utf-8', 'ignore') as f:
    s = f.readlines()

for text in s:
    tokenized_text = tokenizer.tokenize(text)
    score = numpy.zeros((len(tokenized_text)))
    sujest = {}

    #print("Tokenized text : ",tokenized_text)

    result_text = tokenizer.tokenize(text)

    for i in range(0,len(tokenized_text)):
        masked_index = i
        tokenized_text_saved = tokenized_text[masked_index] 

        tokenized_text[masked_index] = '[MASK]'
        #print("Masked text : ",tokenized_text)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        #print("Indexed tokens : ",indexed_tokens)

        token_type_ids = numpy.zeros((len(tokenized_text)))
        attention_mask = numpy.zeros((len(tokenized_text)))

        inputs_onnx = {"token_type_ids":[token_type_ids],"input_ids":[indexed_tokens],"attention_mask":[attention_mask]}

        #print("Input : ",inputs_onnx)

        #print("Predicting...")
        outputs = cpu_model.run(None, inputs_onnx)

        #print("Output : ",outputs)
        def softmax(x):
            u = numpy.sum(numpy.exp(x))
            return numpy.exp(x)/u

        outputs[0][0, masked_index] = softmax(outputs[0][0, masked_index])

        target_ids = tokenizer.convert_tokens_to_ids([tokenized_text_saved])
        index = target_ids[0]
        score[masked_index] = outputs[0][0, masked_index][index]

        predictions = torch.from_numpy(outputs[0][0, masked_index]).topk(1)
        index = predictions.indices[0]
        top_token = tokenizer.convert_ids_to_tokens([index])[0]
        sujest[masked_index] = top_token

        tokenized_text[masked_index] = tokenized_text_saved

    if is_english:
        space=" "
    else:
        space=""

    fine_text = ""
    for i in range(0,len(tokenized_text)):
        prob_yellow = 0.0001
        prob_red = 0.00001
        if score[i]<prob_red:
            fine_text=fine_text+'\033[31m'+space+tokenized_text[i]+'\033[0m'
            if args.sudjest:
                fine_text=fine_text+' ->\033[34m'+space+sujest[i]+'\033[0m'
        elif score[i]<prob_yellow:
            fine_text=fine_text+'\033[33m'+space+tokenized_text[i]+'\033[0m'
            if args.sudjest:
                fine_text=fine_text+' ->\033[34m'+space+sujest[i]+'\033[0m'
        else:
            fine_text=fine_text+space+tokenized_text[i]

    if is_english:
        fine_text = fine_text.replace(' ##', '')
    else:
        fine_text = fine_text.replace('##', '')

    print(fine_text)
