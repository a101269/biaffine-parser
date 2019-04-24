# -*- coding: utf-8 -*-

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # initialize k centroids randomly
    c, old = x[random.sample(range(len(x)), k)], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)

    while old is None or not c.equal(old):
        # handle the empty clusters
        for i in range(k):
            # choose the farthest datapoint from the biggest cluster
            # and move that the empty cluster
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        # update the centroids
        c, old = torch.tensor([x[y.eq(i)].mean() for i in range(k)]), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(dim=-1)
    clusters = [y.eq(i) for i in range(k)]
    clusters = [i.nonzero().view(-1).tolist() for i in clusters if i.any()]

    return clusters


def collate_fn(data):
    reprs = (pad_sequence(i, True) for i in zip(*data))
    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs


class TextSampler(Sampler):

    def __init__(self, buckets, shuffle=False):
        self.buckets = buckets
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return (self.buckets[i][j]
                    for i in torch.randperm(len(self.buckets)).tolist()
                    for j in torch.randperm(len(self.buckets[i])).tolist())
        else:
            return (self.buckets[i][j]
                    for i in range(len(self.buckets))
                    for j in range(len(self.buckets[i])))

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets)


class TextDataset(Dataset):

    def __init__(self, items, n_buckets=1):
        super(TextDataset, self).__init__()

        self.items = items
        # NOTE: the final number of buckets should be less or equal to n_buckets
        self.buckets = kmeans(x=[len(i) for i in self.items[0]], k=n_buckets)

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"  num of sentences: {len(self)}\n"
        info += f"  num of buckets: {len(self.buckets)}\n"
        info += f")"

        return info

    def __getitem__(self, index):
        return tuple(item[index] for item in self.items)

    def __len__(self):
        return len(self.items[0])
