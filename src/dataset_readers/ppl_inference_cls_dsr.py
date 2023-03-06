from src.dataset_readers.retriever_dsr import RetrieverDatasetReader


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


class PPLCLSInferenceDatasetReader(RetrieverDatasetReader):

    def __init__(self, model_name, task_name, index_split, dataset_path, n_tokens=1600, tokenizer=None,
                 index_data_path=None):
        super().__init__(model_name, task_name, index_split, dataset_path, n_tokens, tokenizer, index_data_path)

    def get_ctxs_inputs(self, entry):
        C = self.dataset_wrapper.get_field(entry, 'C')
        X = self.dataset_wrapper.get_field(entry, 'X')
        Y = self.dataset_wrapper.get_field(entry, 'Y')
        Y_TEXT = self.dataset_wrapper.get_field(entry, 'Y_TEXT')
        ctx = [self.index_dataset.dataset[i] for i in entry['ctxs']]
        example_list = [{'C': i['C'], 'X': i['X'], 'Y': i['Y'], 'Y_TEXT': i['Y_TEXT']} for i in ctx]

        return C, X, Y, Y_TEXT, example_list
