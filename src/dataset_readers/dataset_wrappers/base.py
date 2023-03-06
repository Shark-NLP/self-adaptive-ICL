#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random


class DatasetWrapper:
    name = "base"

    def __init__(self):
        self.dataset = None
        self.field_getter = None

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_field(self, entry, field):
        return self.field_getter.functions[field](entry)

    def get_corpus(self, field):
        return [self.get_field(entry, field) for entry in self.dataset]


def load_partial_dataset(dataset, size=1):
    if size == 1 or size >= len(dataset):
        return dataset

    total_size = len(dataset)
    size = int(size * total_size) if size < 1 else size

    rand = random.Random(x=size)
    index_list = list(range(total_size))
    rand.shuffle(index_list)
    dataset = dataset.select(index_list[:size])
    return dataset