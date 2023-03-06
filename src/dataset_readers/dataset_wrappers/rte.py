from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from src.utils.app import App
from src.dataset_readers.dataset_wrappers.base import *
from src.datasets.labels import get_mapping_token

field_getter = App()

# e.g. label2text = {0: "negative", 1: "positive"}
label2text = get_mapping_token("rte")


@field_getter.add("X")
def get_X(entry):
    return entry['sentence1'] if 'X' not in entry.keys() else entry['X']


@field_getter.add("Y_TEXT")  # 获得标签对应的文本
def get_Y_TEXT(entry):
    # print(entry)
    return label2text[entry['label']] if 'Y_TEXT' not in entry.keys() else entry['Y_TEXT']


@field_getter.add("C")  # 数据集有sentence1,sentence2,label 时，用C表示sentence1
def get_C(entry):
    return entry['sentence2'] if 'C' not in entry.keys() else entry['C']


@field_getter.add("Y")  # 获得原始标签 int
def get_Y(entry):
    return entry['label'] if 'Y' not in entry.keys() else entry['Y']


@field_getter.add("ALL")
def get_ALL(entry):
    return f"{entry['sentence1']}\t {entry['sentence2']}" if 'ALL' not in entry.keys() else entry['ALL']


class RTEDatasetWrapper(DatasetWrapper):
    name = "rte"
    premise_field = "premise"
    hypothesis_field = "hypothesis"
    label_field = "label"

    def __init__(self, dataset_path=None, dataset_split=None, ds_size=None):
        def _abs_label(ex):
            ex['label'] = abs(ex['label'])
            return ex
        super().__init__()
        self.task_name = "rte"
        self.field_getter = field_getter
        self.postfix = ""  # for inference
        if dataset_path is None:
            self.dataset = load_dataset("glue", "rte", split=dataset_split)
            self.dataset = self.dataset.map(_abs_label, batched=False, load_from_cache_file=False)
        else:
            self.dataset = Dataset.from_pandas(pd.DataFrame(data=pd.read_json(dataset_path)))

        if dataset_split is not None and isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset[dataset_split]

        if ds_size is not None:
            self.dataset = load_partial_dataset(self.dataset, size=ds_size)
