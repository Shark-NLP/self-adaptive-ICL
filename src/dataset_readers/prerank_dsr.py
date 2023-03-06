import torch
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper


class PrerankDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, field, dataset_path=None, dataset_split=None,
                 ds_size=None, tokenizer=None) -> None:
        self.tokenizer = tokenizer
        self.dataset_wrapper = get_dataset_wrapper(task_name)(dataset_path=dataset_path,
                                                              dataset_split=dataset_split,
                                                              ds_size=ds_size)

        self.field = field

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        enc_text = self.dataset_wrapper.get_field(entry, self.field)

        tokenized_inputs = self.tokenizer.encode_plus(enc_text, truncation=True, return_tensors='pt', padding='longest')

        return {

            'input_ids': tokenized_inputs.input_ids.squeeze(),
            'attention_mask': tokenized_inputs.attention_mask.squeeze(),
            "metadata": {"id": index}
        }

    def __len__(self):
        return len(self.dataset_wrapper)