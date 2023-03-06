import json
import logging
import re
from datasets import load_metric
from src.utils.app import App
from src.datasets.labels import get_mapping_token


def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ", ";", text)
    return text


app = App()


def add_squad_acc2(file_name, id_list=None):
    with open(file_name) as f:
        pred_data = json.load(f)
        predictions = [{'prediction_text': d['generated'].split('\n')[0], 'id': str(i)} for i, d in
                       enumerate(pred_data)]
        references = [{'answers': {'answer_start': [0], 'text': [d['Y_TEXT']]}, 'id': str(i)} for i, d in
                      enumerate(pred_data)]
        metric = load_metric('squad')
        score = metric.compute(predictions=predictions, references=references)
        logging.info(f'exact_match={score["exact_match"]}')
        logging.info(f'f1={score["f1"]}')
    return pred_data, score["f1"]


@app.add("sst2")
def add_sst2_acc(file_name, id_list=None, task_name="sst2"):
    cor = 0.0
    with open(file_name) as f:
        label2text = get_mapping_token(task_name)
        data = json.load(f)
        for line in data:
            if id_list is not None and line['idx'] not in id_list:
                continue
            label = label2text[line['Y']]
            pred = line['generated'].split(" ")[-1].strip()
            if label == pred:
                cor += 1
            else:
                continue
    return data, cor / len(data)


@app.add("web_questions")  # 用start with?
def add_wq_acc(file_name, id_list=None):
    def include(pred, gold):
        pred = pred.lower().strip()
        gold = gold.lower().strip()
        if gold in pred or pred in gold:
            return 1
        else:
            return 0

    with open(file_name) as f:
        data = json.load(f)
        cor = 0
        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            line['acc'] = include(line['generated'], line['Y_TEXT'])
            cor += line['acc']
        lenn = len(id_list) if id_list is not None else len(data)
        return data, cor / lenn


@app.add("rte")
def add_yelp_polarity_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="rte")


@app.add("qnli")
def add_qnli_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="qnli")


@app.add("boolq")
def add_boolq_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="booq")


@app.add("ag_news")
def add_ag_news_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="ag_news")


@app.add("trec")
def add_trec_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="trec")


@app.add("commonsense_qa")
def add_commonsense_qa_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="commonsense_qa")


@app.add("copa")
def add_copa_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="copa")


@app.add("piqa")
def add_copa_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="piqa")


@app.add("mrpc")
def add_mrpc_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="mrpc")


@app.add("yelp_polarity")
def add_yelp_polarity_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="yelp_polarity")


@app.add("imdb")
def add_imdb_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="imdb")


@app.add("sst5")
def add_sst5_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="sst5")


@app.add("mnli")
def add_mnli_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="mnli")


@app.add("snli")
def add_snli_acc(file_name, id_list=None):
    return add_sst2_acc(file_name=file_name, id_list=id_list, task_name="snli")
