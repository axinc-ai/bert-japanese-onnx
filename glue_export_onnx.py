import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

output_path="onnx/glue/bert.onnx"
training_output_dir="output/original"
convert(pipeline_name="sentiment-analysis", framework="pt", model=training_output_dir, tokenizer=training_output_dir, output=Path(output_path), opset=11)
