import torch
import numpy

from transformers import BertJapaneseTokenizer, BertForMaskedLM
from onnxruntime import InferenceSession, SessionOptions, get_all_providers

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
text = '私はお金で動く。'
print("Input text : "+text)

tokenized_text = tokenizer.tokenize(text)
print("Tokenized text : ",tokenized_text)

masked_index = 2
tokenized_text[masked_index] = '[MASK]'
print("Masked text : ",tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print("Indexed tokens : ",indexed_tokens)

cpu_model = InferenceSession("onnx/cl-tohoku/bert-base-japanese-whole-word-masking.onnx")

token_type_ids = numpy.zeros((len(tokenized_text)))
attention_mask = numpy.zeros((len(tokenized_text)))

inputs_onnx = {"token_type_ids":[token_type_ids],"input_ids":[indexed_tokens],"attention_mask":[attention_mask]}

print("Input : ",inputs_onnx)

print("Predicting...")
outputs = cpu_model.run(None, inputs_onnx)

print("Output : ",outputs)

predictions = torch.from_numpy(outputs[0][0, masked_index]).topk(5)

print("Predictions : ")
for i, index_t in enumerate(predictions.indices):
    index = index_t.item()
    token = tokenizer.convert_ids_to_tokens([index])[0]
    print(i, token)
