from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from src.utils.app import App
from src.dataset_readers.dataset_wrappers.base import *

field_getter = App()
letter2num = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


@field_getter.add("X")
def get_X(entry):
    return entry['question'] if 'X' not in entry.keys() else entry['X']


@field_getter.add("C")  # 数据集有sentence1,sentence2,label 时，用C表示sentence1
def get_C(entry):
    if 'C' in entry.keys():
        return entry['C']
    choices = entry['choices']['text']
    return f'{choices[0]}, {choices[1]}, {choices[2]}, {choices[3]} or {choices[4]}?'


@field_getter.add("Y")  # 用于取template
def get_Y(entry):
    return letter2num[entry["answerKey"]] if 'Y' not in entry.keys() else entry['Y']


@field_getter.add("Y_TEXT")
def get_Y_TEXT(entry):
    return entry['choices']['text'][entry['Y']] if 'Y_TEXT' not in entry.keys() else entry['Y_TEXT']


@field_getter.add("ALL")
def get_ALL(entry):
    return f"" if 'ALL' not in entry.keys() else entry['ALL']


class CommonsenseQADatasetWrapper(DatasetWrapper):
    name = "commonsense_qa"
    question_field = "question"
    answer_field = "answerKey"

    def __init__(self, dataset_path=None, dataset_split=None, ds_size=None):
        super().__init__()
        self.field_getter = field_getter
        self.postfix = ""  # for inference
        if dataset_path is None:
            self.dataset = load_dataset("commonsense_qa", split=dataset_split)
        else:
            self.dataset = Dataset.from_pandas(pd.DataFrame(data=pd.read_json(dataset_path)))
            if dataset_split is not None and isinstance(self.dataset, DatasetDict):
                self.dataset = self.dataset[dataset_split]

        if ds_size is not None:
            self.dataset = load_partial_dataset(self.dataset, size=ds_size)
