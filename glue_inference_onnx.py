from transformers import BertJapaneseTokenizer, BertForMaskedLM
from onnxruntime import InferenceSession, SessionOptions, get_all_providers
import numpy

cpu_model = InferenceSession("onnx/glue/bert.onnx",)
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

text = "楽しかった"
#text = "ダメだった"

model_inputs = tokenizer.encode_plus(text, return_tensors="pt")

inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

print("Input : ",inputs_onnx)

score = cpu_model.run(None, inputs_onnx)
print("Output : ",score)

label_name=["positive","negative"]

print("Label : ",label_name[numpy.argmax(numpy.array(score))])