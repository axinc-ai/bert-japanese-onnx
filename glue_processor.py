from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from torch.utils.data.dataset import Dataset
from transformers.data.datasets import GlueDataTrainingArguments
from transformers.data.datasets.glue import Split
from typing import List, Optional, Union
from transformers.data.metrics import simple_accuracy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import is_sklearn_available, requires_sklearn

def classification_metrics(task_name, preds, labels):
    requires_sklearn(classification_metrics)
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    metrics = {"acc": simple_accuracy(preds, labels)}
    return metrics

class ClassificationProcessor(DataProcessor):
    """Processor for the classification data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Ignore TSV header
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            print(line)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples