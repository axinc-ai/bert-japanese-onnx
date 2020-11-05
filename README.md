# BERT japanese to ONNX

Tutorial to learn using Japanese bert, export to onnx and infer

## Requirements

```
python3==3.7.8
torch==1.6.0
transformers==3.4.0
mecab-python3
unidic-lite
fugashi
ipadic
datasets
onnxruntime
```

## MaskLM

### Inference

```
python3 masklm_pytorch.py
```

### Export to ONNX

```
python3 masklm_export_onnx.py
```

### Inference

```
python3 masklm_inference_onnx.py
```

## Glue

### Dataset

TSV file is separated by tab.

- data/original/train.tsv : Training Data
- data/original/dev.tsv : Validation Data
- data/original/test.tsv : Test Data

## Train

We added below code to original transformers/run_glue.py.

```
# regist original processor
from glue_processor import ClassificationProcessor
from glue_processor import classification_metrics
glue_processors[data_args.task_name]=ClassificationProcessor
glue_output_modes[data_args.task_name]="classification"
glue_tasks_num_labels[data_args.task_name]=2
glue_compute_metrics = classification_metrics
```

Training script.

```
python3 run_glue.py --data_dir ./data/original/ --model_name_or_path cl-tohoku/bert-base-japanese-whole-word-masking --task_name original --do_train --do_eval --output_dir output/original
```

## Inference

Inference script.

```
python3 run_glue.py --data_dir ./data/original/ --model_name_or_path cl-tohoku/bert-base-japanese-whole-word-masking --task_name original --do_predict --output_dir output/original
```

## Export to ONNX

```
python3 glue_export_onnx.py
```

## Inference using ONNX

```
python3 glue_inference_onnx.py
```
