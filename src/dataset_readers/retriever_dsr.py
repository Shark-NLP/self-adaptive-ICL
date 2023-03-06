import logging

import torch
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper


def get_length(tokenizer, text):
    tokenized_example = tokenizer.encode_plus(text, truncation=False, return_tensors='pt')
    return int(tokenized_example.input_ids.shape[1])


def set_length(example, **kwargs):
    tokenizer = kwargs['tokenizer']
    set_field = kwargs['set_field']
    field_getter = kwargs['field_getter']

    field_text = field_getter.functions[set_field](example)
    example[f'{set_field}_len'] = get_length(tokenizer, field_text)
    if set_field not in example:
        example[set_field] = field_text
    return example


def set_field(example, **kwargs):
    set_fields = ['C', 'X', 'Y', 'Y_TEXT', 'ALL']
    field_getter = kwargs['field_getter']
    for set_field in set_fields:
        field_text = field_getter.functions[set_field](example)
        if set_field not in example:
            example[set_field] = field_text
    return example


class RetrieverDatasetReader(torch.utils.data.Dataset):

    def __init__(self, model_name, task_name, index_split, dataset_path, n_tokens=700,
                 tokenizer=None, index_data_path=None):
        self.dataset_wrapper = get_dataset_wrapper(task_name)(dataset_path=dataset_path)

        self.index_dataset = get_dataset_wrapper(task_name)(dataset_split=index_split, dataset_path=index_data_path)

        self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.map(
            set_field,
            fn_kwargs={'field_getter': self.dataset_wrapper.field_getter}
        )

        self.index_dataset.dataset = self.index_dataset.dataset.map(
            set_field,
            fn_kwargs={'field_getter': self.index_dataset.field_getter}
        )

        self.n_tokens_in_prompt = n_tokens
        self.num_processes = 1
        self.process_index = 0

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        C, X, Y, Y_TEXT, example_list = self.get_ctxs_inputs(entry)
        return {
            "metadata": {'id': self.num_processes * index + self.process_index,
                         'C': C,
                         'X': X,
                         'Y': Y,
                         'Y_TEXT': Y_TEXT,
                         "examples": example_list}
        }

    def shard(self, accelerator):
        self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index
        )
        self.num_processes = accelerator.num_processes
        self.process_index = accelerator.process_index
        logging.info(f'accelerator.num_processes={accelerator.num_processes}')
        logging.info(f'accelerator.process_index={accelerator.process_index}')
        logging.info(f'after shard,len={len(self.dataset_wrapper)}')

    def __len__(self):
        return len(self.dataset_wrapper)

    def get_ctxs_inputs(self, entry):
        C = self.dataset_wrapper.get_field(entry, 'C')
        X = self.dataset_wrapper.get_field(entry, 'X')
        Y = self.dataset_wrapper.get_field(entry, 'Y')
        Y_TEXT = self.dataset_wrapper.get_field(entry, 'Y_TEXT')
        ctxs_candidates = [self.index_dataset.dataset[i[0]] for i in entry['ctxs_candidates']]
        example_list = [{'C': i['C'], 'X': i['X'], 'Y': i['Y'], 'Y_TEXT': i['Y_TEXT']} for i in ctxs_candidates]

        return C, X, Y, Y_TEXT, example_list
