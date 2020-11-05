from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer

input_text = "Hugging Face is a technology company based in New York and Paris"

# from pipeline
translator = pipeline("translation_en_to_de")
output_text = translator(input_text, max_length=40)
print("Translated : ",output_text)

# from tokenizer ( T5(Text-to-Text Transfer Transformer) )
model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
output_text = tokenizer.decode(outputs[0])
print("Translated : ",output_text)
