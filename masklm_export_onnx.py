import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

model = 'cl-tohoku/bert-base-japanese-whole-word-masking'
#model = 'cl-tohoku/bert-base-japanese-char-whole-word-masking'
#model = 'cl-tohoku/bert-base-japanese'
#model = 'bert-base-cased'
#model = 'bert-base-uncased'

output_path="onnx/"+model+".onnx"
convert(pipeline_name="fill-mask", framework="pt", model=model, tokenizer=model, output=Path(output_path), opset=11)
