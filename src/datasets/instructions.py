import logging

import torch

PLACEHOLDER_C = "<C>"
PLACEHOLDER_X = "<X>"
PLACEHOLDER_Y = "<Y>"
PLACEHOLDER_EXAMPLE = "<E>"

SINGLE_SENT_TASKS = ['imdb', 'sst2', 'yelp', 'rotten', 'elec', 'sst5']
SENT_PAIR_TASKS = ['rte', 'qnli', 'snli', 'mnli']
QA_TASKS = ['squad', 'adversarial_qa', 'mtop', 'iwslt17zh', 'xsum', 'iwslt17de', 'gsm8k', 'web_questions']
CHOICE_TASKS = ['commonsense_qa', 'copa']


def get_task_type(task_name):
    if task_name in SINGLE_SENT_TASKS:
        return "SINGLE_CLS"
    elif task_name in SENT_PAIR_TASKS:
        return "PAIR_CLS"
    elif task_name in QA_TASKS:
        return "QA"
    elif task_name in CHOICE_TASKS:
        return "CHOICE"
    else:
        return "OTHER"


def get_choice_sep(task_name):
    if task_name == 'commonsense_qa':
        return ', '
    elif task_name == 'copa':
        return '||||'
    return None


# 选用不同的template
def get_template(task_name, t=1):
    if t == 1:
        if task_name == "sst2" or task_name == "imdb":
            return {
                0: {
                    "example_instruction": "Negative Movie Review: \"<X>\"",
                    "prompting_instruction": "<E>Negative Movie Review: \"<X>\"",
                },
                1: {
                    "example_instruction": "Positive Movie Review: \"<X>\"",
                    "prompting_instruction": "<E>Positive Movie Review: \"<X>\"",
                }
            }
        elif task_name == 'boolq':
            return {
                0: {  # False
                    "example_instruction": "<C>\n Can we know <X> based on context above? No.",
                    "prompting_instruction": "<E><C>\n Can we know <X> based on context above? No.",
                },
                1: {  # True
                    "example_instruction": "<C>\n Can we know <X> based on context above? Yes.",
                    "prompting_instruction": "<E><C>\n Can we know <X> based on context above? Yes.",
                }
            }
        elif task_name == "commonsense_qa":
            return {
                0: {
                    "example_instruction": "<X> <C> The answer is <Y>",
                    "prompting_instruction": "<E><X> <C> The answer is <Y0>",
                },
                1: {
                    "example_instruction": "<X> <C> The answer is <Y>",
                    "prompting_instruction": "<E><X> <C> The answer is <Y1>",
                },
                2: {
                    "example_instruction": "<X> <C> The answer is <Y>",
                    "prompting_instruction": "<E><X> <C> The answer is <Y2>",
                },
                3: {
                    "example_instruction": "<X> <C> The answer is <Y>",
                    "prompting_instruction": "<E><X> <C> The answer is <Y3>",
                },
                4: {
                    "example_instruction": "<X> <C> The answer is <Y>",
                    "prompting_instruction": "<E><X> <C> The answer is <Y4>",
                }
            }
        elif task_name == "copa" or task_name == "piqa":
            return {
                0: {
                    "example_instruction": "<X> <Y>",
                    "prompting_instruction": "<E><X> <Y0>",
                },
                1: {
                    "example_instruction": "<X> <Y>",
                    "prompting_instruction": "<E><X> <Y1>",
                }
            }
        elif task_name == "sst5":
            return {
                0: {
                    "example_instruction": "\"<X>\" It is terrible.",
                    "prompting_instruction": "<E>\"<X>\" It is terrible.",
                },
                1: {
                    "example_instruction": "\"<X>\" It is bad.",
                    "prompting_instruction": "<E>\"<X>\" It is bad.",
                },
                2: {
                    "example_instruction": "\"<X>\" It is okey.",
                    "prompting_instruction": "<E>\"<X>\" It is okey.",
                },
                3: {
                    "example_instruction": "\"<X>\" It is good.",
                    "prompting_instruction": "<E>\"<X>\" It is good.",
                },
                4: {
                    "example_instruction": "\"<X>\" It is great.",
                    "prompting_instruction": "<E>\"<X>\" It is great.",
                }
            }
        elif task_name == "ag_news":
            return {
                0: {
                    "example_instruction": "\"<X>\" It is about world.",
                    "prompting_instruction": "<E>\"<X>\" It is about world.",
                },
                1: {
                    "example_instruction": "\"<X>\" It is about sports.",
                    "prompting_instruction": "<E>\"<X>\" It is about sports.",
                },
                2: {
                    "example_instruction": "\"<X>\" It is about business.",
                    "prompting_instruction": "<E>\"<X>\" It is about business.",
                },
                3: {
                    "example_instruction": "\"<X>\" It is about science and technology.",
                    "prompting_instruction": "<E>\"<X>\" It is about science and technology.",
                }
            }
        elif task_name == "trec":
            return {
                0: {
                    "example_instruction": "\"<X>\" It is about abbreviation.",
                    "prompting_instruction": "<E>\"<X>\" It is about abbreviation.",
                },
                1: {
                    "example_instruction": "\"<X>\" It is about entity.",
                    "prompting_instruction": "<E>\"<X>\" It is about entity.",
                },
                2: {
                    "example_instruction": "\"<X>\" It is about description and abstract concept.",
                    "prompting_instruction": "<E>\"<X>\" It is about description and abstract concept.",
                },
                3: {
                    "example_instruction": "\"<X>\" It is about human being.",
                    "prompting_instruction": "<E>\"<X>\" It is about human being.",
                },
                4: {
                    "example_instruction": "\"<X>\" It is about location.",
                    "prompting_instruction": "<E>\"<X>\" It is about location.",
                },
                5: {
                    "example_instruction": "\"<X>\" It is about numeric value.",
                    "prompting_instruction": "<E>\"<X>\" It is about numeric value.",
                }
            }
        elif task_name == "yelp_polarity":
            return {
                0: {
                    "example_instruction": "Negative Restaurant Review: \"<X>\"",
                    "prompting_instruction": "<E>Negative Restaurant Review: \"<X>\"",
                },
                1: {
                    "example_instruction": "Positive Restaurant Review: \"<X>\"",
                    "prompting_instruction": "<E>Positive Restaurant Review: \"<X>\"",
                }
            }
        elif task_name == "mtop":
            return {
                0: {
                    "example_instruction": "<X>\t<Y>",
                    "prompting_instruction": "<E><X>\t<Y>",
                }
            }
        elif task_name == "squad" or task_name == "web_questions":
            return {
                0: {
                    "example_instruction": "<C>\t<X>\t<Y>",
                    "prompting_instruction": "<E><C>\t<X>\t<Y>"
                }
            }
        elif task_name == 'gsm8k':
            return {
                0: {
                    "example_instruction": "Solve the follow math problem: <X> \n Answer: <Y>",
                    "prompting_instruction": "<E>Solve the follow math problem: <X> \n Answer: <Y>"
                }
            }
        elif task_name == 'iwslt17zh':
            return {
                0: {
                    "example_instruction": "What is the Chinese translation of <X> : <Y>",
                    "prompting_instruction": "<E>What is the Chinese translation of <X> : <Y>"
                }
            }
        elif task_name == 'iwslt17de':
            return {
                0: {
                    "example_instruction": "What is the German translation of <X> : <Y>",
                    "prompting_instruction": "<E>What is the German translation of <X> : <Y>"
                }
            }
        elif task_name == "mnli" or task_name == "snli":
            return {
                0: {  # entailment
                    "example_instruction": "<C>? Yes, <X>",
                    "prompting_instruction": "<E><C>? Yes, <X>",
                },
                1: {  # neutral
                    "example_instruction": "<C>? Maybe, <X>",
                    "prompting_instruction": "<E><C>? Maybe, <X>",
                },
                2: {  # contradiction
                    "example_instruction": "<C>? No, <X>",
                    "prompting_instruction": "<E><C>? No, <X>",
                }
            }
        elif task_name == 'qnli':
            return {
                0: {  # entailment
                    "example_instruction": "<C> Can we know <X>? Yes.",
                    "prompting_instruction": "<E><C> Can we know <X>? Yes.",
                },
                1: {  # contradiction
                    "example_instruction": "<C> Can we know <X>? No.",
                    "prompting_instruction": "<E><C> Can we know <X>? No.",
                }
            }
        elif task_name == 'rte':
            return {
                0: {  # entailment
                    "example_instruction": "<X>? Yes, <C>",
                    "prompting_instruction": "<E><X>? Yes, <C>",
                },
                1: {  # contradiction
                    "example_instruction": "<X>? No, <C>",
                    "prompting_instruction": "<E><X>? No, <C>",
                }
            }
        elif task_name == 'mrpc':
            return {
                0: {  # entailment
                    "example_instruction": "<C> Yes , <X>",
                    "prompting_instruction": "<E><C> Yes , <X>",
                },
                1: {  # contradiction
                    "example_instruction": "<C> No , <X>",
                    "prompting_instruction": "<E><C> No , <X>",
                }
            }
        elif task_name == 'xsum':
            return {
                0: {
                    "example_instruction": "Document: <X> Summary: <Y>",
                    "prompting_instruction": "<E>Document: <X> Summary: <Y>"
                }
            }

    elif t == 2:
        if task_name == "sst2" or task_name == "imdb":
            return {
                0: {
                    "example_instruction": "\"<X>\" It is terrible.",
                    "prompting_instruction": "<E>\"<X>\" It is terrible.",
                },
                1: {
                    "example_instruction": "\"<X>\" It is great.",
                    "prompting_instruction": "<E>\"<X>\" It is great.",
                }
            }
        elif task_name == "piqa":
            return {
                0: {
                    "example_instruction": "<X>\nWhich is the correct ending? <C>\n <Y>",
                    "prompting_instruction": "<E>\nWhich is the correct ending? <C>\n <Y0>",
                },
                1: {
                    "example_instruction": "<X>\nWhich is the correct ending? <C>\n <Y>",
                    "prompting_instruction": "<E>\nWhich is the correct ending? <C>\n <Y1>",
                }
            }
        elif task_name == "squad":
            return {
                0: {
                    "example_instruction": "The context is: \"<C>\"\nThe answer to the question \"<X>\" is: <Y>\"",
                    "prompting_instruction": "<E>The context is: \"<C>\"\nThe answer to the question \"<X>\" is: <Y>\""
                }
            }
        elif task_name == "trec":
            return {
                0: {
                    "example_instruction": "\"<X>\" This is about abbreviation.",
                    "prompting_instruction": "<E>\"<X>\" It is about abbreviation.",
                },
                1: {
                    "example_instruction": "\"<X>\" It is about entity.",
                    "prompting_instruction": "<E>\"<X>\" It is about entity.",
                },
                2: {
                    "example_instruction": "\"<X>\" It is about description and abstract concept.",
                    "prompting_instruction": "<E>\"<X>\" It is about description and abstract concept.",
                },
                3: {
                    "example_instruction": "\"<X>\" It is about human being.",
                    "prompting_instruction": "<E>\"<X>\" It is about human being.",
                },
                4: {
                    "example_instruction": "\"<X>\" It is about location.",
                    "prompting_instruction": "<E>\"<X>\" It is about location.",
                },
                5: {
                    "example_instruction": "\"<X>\" It is about numeric value.",
                    "prompting_instruction": "<E>\"<X>\" It is about numeric value.",
                }
            }
        elif task_name == "mnli" or task_name == "snli":
            return {
                0: {  # entailment
                    "example_instruction": "Input: \"<C>\" implies \"<X>\"\n Answer: true",
                    "prompting_instruction": "<E> Input: \"<C>\" implies \"<X>\"\n Answer: true",
                },
                1: {  # neutral
                    "example_instruction": "Input: \"<C>\" implies \"<X>\"\n Answer: inconclusive",
                    "prompting_instruction": "<E> Input: \"<C>\" implies \"<X>\"\n Answer: inconclusive",
                },
                2: {  # contradiction
                    "example_instruction": "Input: \"<C>\" implies \"<X>\"\n Answer: false",
                    "prompting_instruction": "<E> Input: \"<C>\" implies \"<X>\"\n Answer: false",
                }
            }
        elif task_name == 'rte' or task_name == 'mrpc':
            return {
                0: {  # entailment
                    "example_instruction": "Can <C> implies <X>? Yes",
                    "prompting_instruction": "<E>Can <C> implies <X>? Yes",
                },
                1: {  # contradiction
                    "example_instruction": "Can <C> implies <X>? No",
                    "prompting_instruction": "<E>Can <C> implies <X>? No",
                }
            }
        elif task_name == 'iwslt17zh':
            return {
                0: {
                    "example_instruction": "English: <X> \n Chinese translation: <Y>",
                    "prompting_instruction": "<E>English: <X> \n Chinese translation: <Y>"
                }
            }
        elif task_name == 'iwslt17de':
            return {
                0: {
                    "example_instruction": "English: <X> \n German translation: <Y>",
                    "prompting_instruction": "<E>English: <X> \n German translation: <Y>"
                }
            }
        elif task_name == "sst5":
            return {
                0: {
                    "example_instruction": "Review: <X>\nSentiment: terrible",
                    "prompting_instruction": "<E>Review: <X>\nSentiment: terrible",
                },
                1: {
                    "example_instruction": "Review: <X>\nSentiment: bad",
                    "prompting_instruction": "<E>Review: <X>\nSentiment: bad",
                },
                2: {
                    "example_instruction": "Review: <X>\nSentiment: okay",
                    "prompting_instruction": "<E>Review: <X>\nSentiment: okay",
                },
                3: {
                    "example_instruction": "Review: <X>\nSentiment: good",
                    "prompting_instruction": "<E>Review: <X>\nSentiment: good",
                },
                4: {
                    "example_instruction": "Review: <X>\nSentiment: great",
                    "prompting_instruction": "<E>Review: <X>\nSentiment: great",
                }
            }
        elif task_name == "trec":
            return {
                0: {
                    "example_instruction": "Input: <X> \n Topic: abbreviation.",
                    "prompting_instruction": "<E>Input: <X> \n Topic: abbreviation.",
                },
                1: {
                    "example_instruction": "Input: <X> \n Topic: entity.",
                    "prompting_instruction": "<E>Input: <X> \n Topic: entity.",
                },
                2: {
                    "example_instruction": "Input: <X> \n Topic: description and abstract concept.",
                    "prompting_instruction": "<E>Input: <X> \n Topic: description and abstract concept.",
                },
                3: {
                    "example_instruction": "Input: <X> \n Topic: human being.",
                    "prompting_instruction": "<E>Input: <X> \n Topic: human being.",
                },
                4: {
                    "example_instruction": "Input: <X> \n Topic: location.",
                    "prompting_instruction": "<E>Input: <X> \n Topic: location.",
                },
                5: {
                    "example_instruction": "Input: <X> \n Topic: numeric value.",
                    "prompting_instruction": "<E>Input: <X> \n Topic: numeric value.",
                }
            }
        elif task_name == "commonsense_qa":
            return {
                0: {
                    "example_instruction": "Answer the following question:\n<X> \n Answer: <Y>",
                    "prompting_instruction": "<E>Answer the following question:\n<X> \n Answer: <Y0>",
                },
                1: {
                    "example_instruction": "Answer the following question:\n<X> \n Answer: <Y>",
                    "prompting_instruction": "<E>Answer the following question:\n<X> \n Answer: <Y1>",
                },
                2: {
                    "example_instruction": "Answer the following question:\n<X> \n Answer: <Y>",
                    "prompting_instruction": "<E>Answer the following question:\n<X> \n Answer: <Y2>",
                },
                3: {
                    "example_instruction": "Answer the following question:\n<X> \n Answer: <Y>",
                    "prompting_instruction": "<E>Answer the following question:\n<X> \n Answer: <Y3>",
                },
                4: {
                    "example_instruction": "Answer the following question:\n<X> \n Answer: <Y>",
                    "prompting_instruction": "<E>Answer the following question:\n<X> \n Answer: <Y4>",
                }
            }
    elif t == 3:
        if task_name == "squad":
            return {
                0: {
                    "example_instruction": "The answer to the question \"<X>\" is: <Y>\"",
                    "prompting_instruction": "<E>The context is: \"<C>\"\nThe answer to the question \"<X>\" is: <Y>\""
                }
            }

    return None



def build_instruction(instruction, c=None, x=None, y_text=None, e=None, e_instruction=None, tokenizer=None, max_len=700,
                      C_KEY='C', X_KEY='X', Y_KEY='Y', Y_TEXT_KEY='Y_TEXT', reverse=True, need_span_ids=False,
                      prior=False, prior_no=1, choice=False, choice_sep=', '):
    """

    Args:
        choice_sep:
        choice:
        prior_no:
        prior: 将c x 替换为等量的mask
        Y_TEXT_KEY:
        need_span_ids: 如果为True，会在instruction后面拼上需要计算prob对应的字符位置. 格式为". 12 24"，直接rfind(‘.’)再split(' ')即可
        一般找除了example之外的所有部分，包括X和Y，暂且命名为X
        reverse:
        Y_KEY:
        X_KEY:
        C_KEY:
        instruction: prompting_instruction
        c: sentence1
        x: sentence2
        y_text: label text
        e: example list [{'C': str,'X':str,'Y':int},...]
        e_instruction: {label1:example_instruction1,...}
        tokenizer:
        max_len:
    Returns:

    """
    output = instruction

    if choice:
        choices = c.replace(' or ', choice_sep).replace('?', '').split(choice_sep)
        for i in range(len(choices)):
            output = output.replace(f'<Y{i}>', choices[i])

    if prior:
        if prior_no == 1:
            if c is not None:
                c_len = len(tokenizer.tokenize(c))
                c = ' '.join(['x' for i in range(c_len)])

            if x is not None:
                x_len = len(tokenizer.tokenize(x))
                x = ' '.join(['x' for i in range(x_len)])

            if y_text is not None:  # not used
                y_len = len(tokenizer.tokenize(y_text))
                y_text = ' '.join(['x' for i in range(y_len)])
        elif prior_no == 2:
            c = 'N/A'
            x = 'N/A'
            y_text = 'N/A'
        elif prior_no == 3:
            c = '[MASK]'
            x = '[MASK]'
            y_text = '[MASK]'

    if c is not None:
        output = output.replace(PLACEHOLDER_C, c)

    if x is not None:
        output = output.replace(PLACEHOLDER_X, x)

    if y_text is not None:  # not used
        output = output.replace(PLACEHOLDER_Y, y_text)

    if e is not None:

        cur_len = len(tokenizer.tokenize(output))
        if cur_len > max_len:
            logging.info(f'x is too long {cur_len}')
            t = ' '.join(tokenizer.tokenize(output)[-cur_len + 30:])
            return f'{t}. 0 1', []

        # print(e)
        keep_exs = []
        keep_ex_strs = []
        if len(e) == 0:
            output = output.replace(PLACEHOLDER_EXAMPLE, "")
            if need_span_ids:
                span_begin = 0
                span_end = len(output)
                output += f'. {span_begin} {span_end}'

        else:
            total_len = len(tokenizer.tokenize(output))
            for _ex in e:
                if torch.is_tensor(_ex[Y_KEY]):
                    _ex_str = build_instruction(instruction=e_instruction[_ex[Y_KEY].item()],
                                                c=_ex[C_KEY] if C_KEY in _ex else None,
                                                x=_ex[X_KEY],
                                                y_text=_ex[Y_TEXT_KEY] if Y_TEXT_KEY in _ex else None)
                else:
                    _ex_str = build_instruction(instruction=e_instruction[_ex[Y_KEY]],
                                                c=_ex[C_KEY] if C_KEY in _ex else None,
                                                x=_ex[X_KEY],
                                                y_text=_ex[Y_TEXT_KEY] if Y_TEXT_KEY in _ex else None)
                _len = len(tokenizer.tokenize(_ex_str))
                if _len + total_len <= max_len:
                    keep_exs.append(_ex)
                    keep_ex_strs.append(_ex_str)
                    total_len += _len
                else:
                    break
            if reverse:
                keep_ex_strs.reverse()

            if need_span_ids:  # concat the begin and end positions e.g.". 12 24"，
                span_begin = output.rfind(PLACEHOLDER_EXAMPLE)
                span_end = len(output)
                if span_begin == -1:
                    span_begin = 0
                span_len = span_end - span_begin

            output = output.replace(PLACEHOLDER_EXAMPLE, '\n\n'.join(keep_ex_strs) + '\n\n')

            if need_span_ids:
                span_end = len(output)
                span_begin = span_end - span_len
                output += f'. {span_begin} {span_end}'  # [ )

        return output, keep_exs  # return the in-context examples with  (possibly) reduced number

    # if any placeholder is not set yet, reset to ""
    output = output.replace(PLACEHOLDER_C, "").replace(PLACEHOLDER_X, ""). \
        replace(PLACEHOLDER_Y, "").replace(PLACEHOLDER_EXAMPLE, "")
    return output
