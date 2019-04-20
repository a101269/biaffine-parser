# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, words, vectors):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word])

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        words, vectors = map(list, zip(*reprs))
        embedding = cls(words, vectors)

        return embedding
