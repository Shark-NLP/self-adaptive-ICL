import glob
import json
import os
import warnings
import logging

import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForSeq2SeqLM

from src.metrics import eval_datasets
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg, accelerator) -> None:
        self.task_name = cfg.dataset_reader.task_name
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.output_file = cfg.output_file
        self.accelerator = accelerator
        self.model_name = cfg.model_name

        if cfg.model_name == 'opt-175b':
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            self.tokenizer.pad_token = "<|endoftext|>"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model, self.dataloader = self.init_model_dataloader(cfg)

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size)
        if cfg.model_name == 'opt-175b':
            model = None
        elif 't5' in cfg.model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
            model = self.accelerator.prepare(model)
            if hasattr(model, "module"):
                model = model.module
        else:
            model = hu.instantiate(cfg.model).eval()
            model = self.accelerator.prepare(model)
            if hasattr(model, "module"):
                model = model.module

        return model, dataloader

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        avg_ice_num = 0
        total_num = 0
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            for i, entry in enumerate(dataloader):
                metadata = entry.pop("metadata")
                with torch.no_grad():
                    res = self.model.generate(input_ids=entry.input_ids,
                                              attention_mask=entry.attention_mask,
                                              eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                              pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                              max_new_tokens=100,
                                              do_sample=False)
                    a = int(entry.attention_mask.shape[1])  # maxlength???
                    for mdata, res_el in zip(metadata, res.tolist()):
                        mdata['generated'] = self.dataset_reader.tokenizer.decode(res_el[a:],
                                                                                  skip_special_tokens=True)
                        buffer.write(mdata)
                        avg_ice_num += len(mdata['prompt_list'])
                        total_num += 1

        logging.info(f"Average number of in-context examples after truncating is {avg_ice_num / total_num}")

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                data.extend(f.read())
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)

        with open(self.output_file, "w") as f:
            json.dump(data, f)

        data, metric = eval_datasets.app[self.task_name](self.output_file)
        logger.info(f"metric: {str(metric)}")
        with open(self.output_file + '_metric', "w") as f:
            logger.info(f'{self.output_file}:{metric}')
            json.dump({'metric': metric}, f)
        with open(self.output_file, "w") as f:
            json.dump(data, f)

        return data


@hydra.main(config_path="configs", config_name="inferencer")
def main(cfg):
    logger.info(cfg)
    accelerator = Accelerator()
    inferencer = Inferencer(cfg, accelerator)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferencer.forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_results()


if __name__ == "__main__":
    main()
