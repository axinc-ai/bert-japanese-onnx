from transformers import BertJapaneseTokenizer, BertForMaskedLM

import numpy
import ailia

# require ailia SDK 1.2.5 and later

ailia_model = ailia.Net("onnx/glue/bert.onnx.prototxt","onnx/glue/bert.onnx")
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

text = "楽しかった"
#text = "ダメだった"

model_inputs = tokenizer.encode_plus(text, return_tensors="pt")

inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

print("Input : ",inputs_onnx)

score = ailia_model.predict(inputs_onnx)
print("Output : ",score)

label_name=["positive","negative"]

print("Label : ",label_name[numpy.argmax(numpy.array(score))])
