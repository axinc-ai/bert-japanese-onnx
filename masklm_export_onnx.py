import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

output_path="onnx/masklm/bert.onnx"
convert(pipeline_name="fill-mask", framework="pt", model='cl-tohoku/bert-base-japanese-whole-word-masking', tokenizer='cl-tohoku/bert-base-japanese-whole-word-masking', output=Path(output_path), opset=11)
