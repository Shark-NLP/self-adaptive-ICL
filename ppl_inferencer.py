import os
import warnings
import logging
import hydra
import numpy as np
import tqdm
import random
from accelerate import Accelerator

from src.utils.cache_util import BufferedJsonWriter
from src.datasets.labels import  get_mapping_token

from src.utils.calculate import dict_list2list_dict, transform
from inferencer import Inferencer
from src.datasets.instructions import *
from src.models.model import evaluate, generate

logger = logging.getLogger(__name__)


# 改为继承inferencer
class PPLInferencer(Inferencer):

    def __init__(self, cfg, accelerator) -> None:
        super(PPLInferencer, self).__init__(cfg, accelerator)

        self.output_file = cfg.output_file
        self.instruction_template = cfg.instruction_template
        self.span = cfg.span
        self.task_type = get_task_type(self.task_name)
        self.n_tokens = cfg.n_tokens
        self.printonce = 0
        self.calibrate = cfg.calibrate
        self.prior_no = cfg.prior_no
        self.reverse_label = cfg.reverse_label

        instructions = get_template(self.task_name, self.instruction_template)
        if instructions is not None:
            self.labels = [y for y in instructions.keys()]  # int
            self.example_instruction = {label: instructions[label]['example_instruction'] for label in self.labels}
            self.prompting_instruction = {label: instructions[label]['prompting_instruction'] for label in self.labels}
        if self.task_type == "QA":
            self.tokenizer.padding_side = "left"


    def forward(self):

        def build_batch(_metadata, y, generated=None, prior=False):

            _metadata = transform(_metadata)

            if self.reverse_label:
                for m in _metadata:
                    for e in m['examples']:
                        e['Y'] = (e['Y'] + 1) % len(self.labels)

            choice = self.task_type == "CHOICE"
            choice_sep = get_choice_sep(self.task_name)
            if generated is None:
                return [build_instruction(x=m['X'], c=m['C'],
                                          e=m['examples'],
                                          y_text="",
                                          tokenizer=self.tokenizer,
                                          instruction=self.prompting_instruction[y],
                                          e_instruction=self.example_instruction,
                                          need_span_ids=self.span,
                                          max_len=self.n_tokens,
                                          prior=prior, prior_no=self.prior_no,
                                          choice=choice, choice_sep=choice_sep)[0] for m in _metadata]
            else:
                return [build_instruction(x=m['X'], c=m['C'],
                                          e=m['examples'],
                                          y_text=y_text,
                                          tokenizer=self.tokenizer,
                                          instruction=self.prompting_instruction[y],
                                          e_instruction=self.example_instruction, need_span_ids=self.span,
                                          max_len=self.n_tokens)[0] for m, y_text in zip(_metadata, generated)]

        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        mapping_token = get_mapping_token(self.task_name)
        tmpfile = f"{self.output_file}tmp_{self.accelerator.device}.bin"
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:

            for ii, entry in enumerate(dataloader):
                metadata = entry.pop("metadata")

                batch_labels = [build_batch(metadata, label) for label in self.labels]  # label:int
                if self.calibrate:
                    batch_labels_prior = [build_batch(metadata, label, prior=True) for label in self.labels]
                    prior_loss_list = []
                    for batch in batch_labels_prior:
                        with torch.no_grad():
                            prior_ce_loss, prior_lens = evaluate(self, batch, span=self.span)
                            avg_prior_loss = (prior_ce_loss / prior_lens).tolist()
                            prior_loss_list.append(avg_prior_loss)

                if self.printonce > 0:
                    self.printonce -= 1
                    logger.info('batchlabels', batch_labels)

                if self.task_type != "QA":
                    loss_list = []  # [labels,batch size]
                    lens_list = []

                    for i, batch in enumerate(batch_labels):
                        with torch.no_grad():
                            ce_loss, lens = evaluate(self, batch, span=self.span)
                            avg_loss = (ce_loss / lens).tolist()
                            loss_list.append(avg_loss)
                            lens_list.append(lens)
                    if self.calibrate:
                        preds_prior = np.array(prior_loss_list).argmin(axis=0)
                        preds_prior = [mapping_token[pred] for pred in preds_prior]

                        prior_probs = np.exp(-np.array(prior_loss_list))
                        prior_normalized_probs = prior_probs / prior_probs.sum(0, keepdims=True)
                        prior_probs = np.transpose(prior_normalized_probs).tolist()

                        loss_list = np.array(loss_list) - np.array(prior_loss_list)

                    preds = np.array(loss_list).argmin(axis=0)  # [batch size]
                    preds = [mapping_token[pred] for pred in preds]

                    probs = np.exp(-np.array(loss_list))
                    normalized_probs = probs / probs.sum(0, keepdims=True)
                    probs = np.transpose(normalized_probs).tolist()
                else:
                    batch = batch_labels[0]
                    with torch.no_grad():
                        preds = generate(self, batch, span=self.span)

                    batch_labels = [build_batch(metadata, label, preds) for label in self.labels]
                    loss_list = []
                    for batch in batch_labels:
                        with torch.no_grad():
                            ce_loss, lens = evaluate(self, batch, span=self.span)
                            avg_loss = (ce_loss / lens).tolist()
                            loss_list.append(avg_loss)
                    probs = np.exp(-np.array(loss_list))
                    probs = np.transpose(probs).tolist()

                metadata.pop("examples")

                metadata_tmp = dict_list2list_dict(metadata)

                if self.calibrate and self.task_type != "QA":
                    for mdata, pred_text, pred_text_prior, prob, prior_prob in zip(metadata_tmp, preds, preds_prior,
                                                                                   probs, prior_probs):

                        mdata['generated'] = pred_text
                        mdata['prior'] = pred_text_prior
                        mdata['probs'] = prob
                        mdata['prior_prob'] = prior_prob
                        for key in mdata.keys():
                            if torch.is_tensor(mdata[key]):
                                mdata[key] = mdata[key].tolist()
                        buffer.write(mdata)
                else:
                    for mdata, pred_text, prob in zip(metadata_tmp, preds, probs):

                        mdata['generated'] = pred_text
                        mdata['probs'] = prob
                        for key in mdata.keys():
                            if torch.is_tensor(mdata[key]):
                                mdata[key] = mdata[key].tolist()
                        buffer.write(mdata)


@hydra.main(config_path="configs", config_name="ppl_inferencer")
def main(cfg):
    logger.info(cfg)
    random.seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)
    if not cfg.overwrite:
        if os.path.exists(cfg.output_file):
            logger.info(f'{cfg.output_file} already exists,skip')
            return
    accelerator = Accelerator()
    inferencer = PPLInferencer(cfg, accelerator)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferencer.forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_results()


if __name__ == "__main__":
    main()
