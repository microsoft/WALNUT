# this class wraps a torch.utils.data.DataLoader into an iterator for batch by batch fetching
import torch
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split as tts
from datasets import DatasetDict
from collections import Counter

class DataIterator(object):
    def __init__(self, dataloader, nonstop=True):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)
        self.nonstop = nonstop

    def __next__(self):
        try:
            tup = next(self.iterator)
        except StopIteration:
            if not self.nonstop:
                raise StopIteration()
            
            self.iterator = iter(self.loader)
            tup = next(self.iterator)

        return tup

def load_precomputed_split(dataset, seed=42, precomputed_indices_file=None):
    """
    Load pre-computed clean/weak splits for token-level classification
    The splits have been computed via stratified sampling

    # Input:
    :param dataset: a huggingface dataset
    :param seed: random seed
    :param precomputed_indices_file: the path of the file with pre-computed indices
    :return: a huggingface DatasetDict with a "train" (clean) and "test" (weak) Dataset Object
    """

    assert precomputed_indices_file is not None, "need to provide the path for the pre-computed indices file"
    # Load pre-computed indices from a file. If indices do not exist, then create them and update the file
    assert os.path.exists(precomputed_indices_file), "file with precomputed indices does not exist"
    inds_dict = joblib.load(precomputed_indices_file)
    index_str = "seed{}".format(seed)
    assert index_str in inds_dict, "splits have not been computed for seed={}".format(seed)
    clean_inds, weak_inds = inds_dict[index_str]["clean"], inds_dict[index_str]["weak"]

    split = DatasetDict()
    split['train'] = dataset.select(clean_inds)
    split['test'] = dataset.select(weak_inds)
    assert len(set(split['train']['id']) & set(split['test']['id'])) == 0, "issue with splitting data"
    return split