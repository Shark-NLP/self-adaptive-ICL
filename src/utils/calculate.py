import copy
import itertools

import numpy as np


def entropy(probs: np.array, label_dim: int = 0, mask=None):
    if mask is None:
        return - (probs * np.log(probs)).sum(label_dim)
    return - (mask * probs * np.log(probs)).sum(label_dim)


def mdl(probs: np.array, label_dim: int = 0, mask=None):
    if mask is None:
        return - (np.log(probs)).sum(label_dim)
    return - (mask * np.log(probs)).sum(label_dim)


# dict_list2list_dict
def dict_list2list_dict(m):
    return [{key: m[key][j] for key in m.keys()}
            for j in range(len(m[list(m.keys())[0]]))]


def reshape_examples(examples):  # 8*{X:12,Y:12...}->12*{X:8,Y:8...}->12*[8*{X,Y}]
    len1 = len(examples)  # example num
    if (len1 > 0):
        len2 = len(examples[0]['X'])  # batchsize
    else:
        len2 = 0
    _examples = []
    for j in range(len2):
        example = {'C': [examples[k]['C'][j] for k in range(len1)],
                   'X': [examples[k]['X'][j] for k in range(len1)],
                   'Y': [examples[k]['Y'][j] for k in range(len1)],
                   'Y_TEXT': [examples[k]['Y_TEXT'][j] for k in range(len1)]}
        _examples.append(dict_list2list_dict(example))

    return _examples


def transform(_metadata):
    metadata_tmp = copy.deepcopy(_metadata)
    metadata_tmp["examples"] = reshape_examples(_metadata["examples"])
    if (len(metadata_tmp["examples"]) == 0):
        metadata_tmp["examples"] = [[] for i in range(len(metadata_tmp['X']))]
    return dict_list2list_dict(metadata_tmp)


def get_global_entropy(preds, label_num):
    count = [1 for i in range(label_num)]  # 防止0导致的NAN
    for pred in preds:
        count[pred] += 1
    probs = np.array(count) / len(preds)
    return entropy(probs)


def get_local_entropy(probs):
    return np.vectorize(lambda x: entropy(x))(np.array(probs)).sum() / len(probs)


def get_mi2(probs):
    conditional_e = get_local_entropy(probs)
    preds = np.array(probs).argmax(axis=0)
    e = get_global_entropy(preds, len(probs[0]))
    mi = e - conditional_e
    return mi


# def get_mi(probs):
#     return get_mi2(probs)

def get_mi(probs):  # 这种不适用与ppl inferencer，效果很差
    conditional_e = get_local_entropy(probs)
    global_probs = np.array(probs).sum(axis=0)
    global_probs = global_probs / global_probs.sum()
    e = entropy(global_probs)
    mi = e - conditional_e
    return mi


def get_permutations(num):
    array = [i for i in range(num)]
    permutation = list(itertools.permutations(array))
    permutation = [list(t) for t in permutation]
    return permutation  # list int
