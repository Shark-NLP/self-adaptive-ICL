from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from src.utils.calculate import entropy


def no_init(loading_code):
    '''
    no_init_weights is used in from_pretrained to speed up loading large models.
    However, torch-built-in modules like torch.nn.Linear are heavily used in models of transformers,
    while its weights initialization cannot be disabled by no_init_weights.
    '''

    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def generate(self, batch, span=False):
    if span:
        # 从input中取出字符级别的span，并把这部分去掉
        for i, input_text in enumerate(batch):
            id_begin = input_text.rfind('.')
            batch[i] = input_text[:id_begin]

    tokenized_inputs = \
        self.tokenizer.batch_encode_plus(batch, truncation=True, max_length=self.n_tokens, return_tensors='pt',
                                         add_special_tokens=False, padding='longest').to(
            self.model.device)  # truncation
    res = self.model.generate(input_ids=tokenized_inputs.input_ids,
                              attention_mask=tokenized_inputs.attention_mask,
                              # eos_token_id=self.tokenizer.encode("\n")[0],  # prompt中是以\n表示结束
                              pad_token_id=self.tokenizer.pad_token_id,
                              max_new_tokens=100,
                              do_sample=False)
    a = tokenized_inputs.attention_mask.shape[1]
    generated = self.tokenizer.batch_decode(res[:, a:], skip_special_tokens=True)
    return generated


def evaluate(self, input_texts: List[str], span=False) -> Tuple:

    def span_char2span_token(encoded, x_span_char0):
        x_span_token0 = []
        for ii in range(len(x_span_char0)):
            x_span_token_begin = -1
            x_span_token_end = -1
            for token_index in range(len(encoded.tokens(ii))):
                this_token = encoded.word_ids(ii)[token_index]
                if (not this_token == None):
                    # print('###########################')
                    # print(token_index)
                    cur_span = encoded.token_to_chars(ii, token_index)
                    # print(encoded.token_to_chars(token_index))
                    if x_span_token_begin == -1 and cur_span.start >= x_span_char0[ii][
                        0] - 1:
                        x_span_token_begin = token_index
                    if cur_span.end <= x_span_char0[ii][1] + 1:
                        x_span_token_end = token_index

            x_span_token0.append([x_span_token_begin, x_span_token_end])
        return x_span_token0

    def span_char2span_token2(encoded, x_span_char0):
        decode_lens = (encoded["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        x_span_token0 = []
        for ii in range(len(x_span_char0)):
            decode_len = decode_lens[ii]
            decode_list = [self.tokenizer.decode(encoded['input_ids'][ii][jj]) for jj in range(decode_len)]
            decode_list_len = [len(t) for t in decode_list]
            span_len = x_span_char0[ii][1] - x_span_char0[ii][0]
            sum_len = 0
            x_span_token_begin = len(decode_list_len) - 1
            x_span_token_end = len(decode_list_len)
            for k in range(len(decode_list_len) - 1, -1, -1):
                sum_len += decode_list_len[k]
                if sum_len > span_len:
                    x_span_token_begin = k + 1
                    break
            x_span_token0.append([x_span_token_begin, x_span_token_end])
        return x_span_token0

    if span:
        span_char = []
        for i, input_text in enumerate(input_texts):
            id_begin = input_text.rfind('.')
            tmp = input_text[id_begin:].split(' ')
            input_texts[i] = input_text[:id_begin]
            span_char.append([int(tmp[1]), int(tmp[2])])

    inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)

    if self.model_name == 'opt-175b':
        import requests
        import json
        URL = "http://10.140.0.230:6010/completions"
        headers = {
            "Content-Type": "application/json; charset=UTF-8"
        }
        pyload = {"prompt": input_texts, "max_tokens": 0, "echo": True}
        response = json.loads(
            requests.post(URL, data=json.dumps(pyload), headers=headers, proxies={"https": "", "http": ""}).text)

        lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
        loss_lens = np.array([len(r['logprobs']['token_logprobs']) for r in response['choices']])

        loss = [r['logprobs']['token_logprobs'] for r in response['choices']]

        max_len = loss_lens.max()
        loss_pad = list(map(lambda l: l + [0] * (max_len - len(l)), loss))

        loss = -np.array(loss_pad)

        loss = torch.tensor(loss)
        if span:
            span_token = span_char2span_token2(inputs, span_char)
        if span:
            mask = torch.zeros_like(loss)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if span_token[i][0] <= j <= span_token[i][1]:
                        mask[i][j] = 1

            loss = loss * mask
            lens = np.array([(x_span[1] - x_span[0]) for x_span in span_token])

        ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
        return ce_loss, lens

    if span:
        if 'opt' in self.model_name:
            span_token = span_char2span_token2(inputs, span_char)
        else:
            try:
                span_token = span_char2span_token(inputs, span_char)
            except:
                span_token = span_char2span_token2(inputs, span_char)

    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}


    outputs = self.model(**inputs)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    # note here we assume padding is performed on the right, left padding token will affect position_id in gpt2
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())

    lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()

    if span:
        mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if span_token[i][0] <= j <= span_token[i][1]:
                    mask[i][j] = 1
        loss = loss * mask
        lens = np.array([(x_span[1] - x_span[0]) for x_span in span_token])
    ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))

    return ce_loss, lens


def get_model(**kwargs):
    return no_init(lambda: AutoModelForCausalLM.from_pretrained(**kwargs))


def get_score(self, batch_labels, method='mdl', span=False, prior_loss_list=None):
    loss_list = []  # [labels,batch size]
    for i, batch in enumerate(batch_labels):
        with torch.no_grad():
            ce_loss, lens = evaluate(self, batch, span=span)
            avg_loss = (ce_loss / lens).tolist()
            loss_list.append(avg_loss)
    if prior_loss_list is not None:
        loss_list = np.array(loss_list) - np.array(prior_loss_list)
    probs = np.exp(-np.array(loss_list))  # [labels,dataset size]

    normalized_probs = probs / probs.sum(0, keepdims=True)

    if method == 'mdl':
        neg_entropy = -entropy(normalized_probs, label_dim=0)
        return neg_entropy
    elif method == "entropy":
        return entropy(normalized_probs, label_dim=0)
