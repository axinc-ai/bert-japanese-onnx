import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

output_path="onnx/translation/t5.onnx"
convert(pipeline_name="translation_en_to_de", framework="pt", model="t5-base", output=Path(output_path), opset=11)

# Issue https://github.com/huggingface/transformers/issues/5948
