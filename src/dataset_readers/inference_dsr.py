import more_itertools
import numpy as np
import torch
from transformers import AutoTokenizer

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


class InferenceDatasetReader(torch.utils.data.Dataset):

    def __init__(self, model_name, task_name, index_split, dataset_path, n_tokens=1600):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.dataset_wrapper = get_dataset_wrapper(task_name)(dataset_path=dataset_path)
        self.index_dataset = get_dataset_wrapper(task_name)(dataset_split=index_split)

        self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.map(
            set_length,
            fn_kwargs={'tokenizer': self.tokenizer,
                       'set_field': 'sentence',
                       'field_getter': self.dataset_wrapper.field_getter}
        )
        self.index_dataset.dataset = self.index_dataset.dataset.map(
            set_length,
            fn_kwargs={'tokenizer': self.tokenizer,
                       'set_field': 'sentence_label',
                       'field_getter': self.index_dataset.field_getter}
        )

        self.n_tokens_in_prompt = n_tokens
        self.num_processes = 1
        self.process_index = 0

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        question, answer, lengths_list, prompts_list = self.get_ctxs_inputs(entry)

        trunc_prompts_list = self.truncate(question, lengths_list, prompts_list)
        prompt_enc_text = "\n".join(trunc_prompts_list)

        enc_text = f"{prompt_enc_text}\n{question}\t{self.dataset_wrapper.postfix}"
        tokenized_example = self.tokenizer.encode_plus(enc_text, truncation=False, return_tensors='pt',
                                                       add_special_tokens=False)

        entry['id'] = self.num_processes * self.process_index + index
        entry['prompt_list'] = trunc_prompts_list
        entry['enc_text'] = enc_text

        return {
            'input_ids': tokenized_example.input_ids.squeeze(),
            'attention_mask': tokenized_example.attention_mask.squeeze(),
            "metadata": entry
        }

    def __len__(self):
        return len(self.dataset_wrapper)

    def get_ctxs_inputs(self, entry):
        question = self.dataset_wrapper.get_field(entry, 'q')
        answer = self.dataset_wrapper.get_field(entry, 'a')
        ctx = [self.index_dataset.dataset[i] for i in entry['ctxs']]
        # ctx = [self.index_dataset.dataset[i['id']] for i in entry['ctxs']]
        prompts_list = [i['qa'] for i in ctx]
        lengths_list = [i['qa_len'] for i in ctx]
        return question, answer, lengths_list, prompts_list

    def shard(self, accelerator):
        self.num_processes = accelerator.num_processes
        self.process_index = accelerator.process_index
        self.dataset_wrapper.dataset = list(
            more_itertools.distribute(accelerator.num_processes, self.dataset_wrapper.dataset)[
                accelerator.process_index])

    def truncate(self, question, lengths_list, prompts_list):
        q_length = get_length(self.tokenizer, question)
        max_prompts = np.searchsorted(np.cumsum(lengths_list), self.n_tokens_in_prompt - q_length)
        # logger.info(self.n_tokens_in_prompt, max_prompts)
        trunc_prompts_list = prompts_list[:max_prompts][::-1]  # more similar more close
        return trunc_prompts_list
