
def get_mapping_set(task_name):
    if task_name == 'sst2' or task_name == 'yelp_polarity' or task_name == "imdb":
        return sst2_map_set


def get_mapping_token(task_name):
    r = get_mapping_token0(task_name)

    if r is None:
        return {i: str(i) for i in range(100)}  # 100分类应该够了
    return r


def get_mapping_token0(task_name):
    if task_name == 'sst2' or task_name == 'yelp_polarity' or task_name == "imdb":
        return sst2_map_token
    if task_name == 'mnli' or task_name == "snli":
        return mnli_map_token
    if task_name == 'rte' or task_name == 'qnli' or task_name == 'mrpc':
        return binary_nli_map_token
    if task_name == 'sst5':
        return sst5_map_token
    if task_name == 'ag_news':
        return ag_news_map_token
    if task_name == 'trec':
        return trec_map_token
    return None


def get_mapping_idx(task_name):
    idx2token = get_mapping_token(task_name)
    return {value: key for key, value in idx2token.items()}


data_path = {'sst2': ["gpt3mix/sst2", None],
             'mrpc': ["glue", "mrpc"],
             'snli': ['snli', None],
             'mnli': ['LysandreJik/glue-mnli-train', None],
             'iwslt17zh': ["iwslt2017", 'iwslt2017-zh-en'],
             "mtop": ["iohadrubin/mtop", 'mtop'],
             "qnli": ["glue", "qnli"],
             "rte": ["glue", "rte"],
             "squad": ["squad", None],
             "sst5": ["SetFit/sst5", None],
             "ag_news": ["ag_news", None],
             "trec": ["trec", None],
             "commonsense_qa": ["commonsense_qa",None],
             "copa":['super_glue', 'copa'],
             "boolq": ['super_glue', 'boolq']
             }


def get_datapath(task_name):
    return data_path[task_name]


sst2_map_set = {0: [" negative", " bad", " terrible", "negative", "bad", "terrible"],
                1: [" positive", " good", " great", "positive", "good", "great"]}
sst2_map_token = {0: "terrible", 1: "great"}
mnli_map_token = {0: "No", 1: "Maybe", 2: 'Yes'}
binary_nli_map_token = {0: 'Yes', 1: "No"}
sst5_map_token = {0: "terrible", 1: "bad", 2: "okay", 3: "good", 4: "great"}
ag_news_map_token = {0: "world", 1: "sports", 2: "business", 3: "technology"}
trec_map_token = {0: "abbreviation", 1: "entity", 2: "description and abstract concept", 3: "human being",
                  4: "location", 5: "numeric value"}