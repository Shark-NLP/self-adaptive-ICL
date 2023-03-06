from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch


def load_tokenizer_model(checkpoint, pad_trunc_right=True, use_fast=True):
    if checkpoint == 'opt175b':
        model = None
        checkpoint = 'facebook/opt-30b'  # all opt tokenizers are same
        use_fast = False  # the fast tokenizer currently does not work correctly
    elif 'opt' in checkpoint:
        model = load_opt_model(checkpoint)
        use_fast = False  # the fast tokenizer currently does not work correctly
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        if torch.cuda.is_available():
            model.parallelize()

    if pad_trunc_right:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left',
                                                  truncation_side='left', use_fast=use_fast)

    tokenizer.pad_token = tokenizer.eos_token  # original pad token id is None, not in embedding matrix
    if model is not None:
        model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def load_opt_model(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint)

    # Initializes an empty shell with the model. This is instant and does not take any RAM.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    # Initialize the model under the previous context manager breaks the tied weights.
    model.tie_weights()

    # Infer device map automatically
    device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')
    print(device_map)

    load_checkpoint_and_dispatch(
        model.model,
        checkpoint,
        device_map=device_map,
        offload_folder=None,
        dtype='float16',
        offload_state_dict=True
    )
    model.tie_weights()
    return model