import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split as tts
from datasets import DatasetDict
from collections import Counter

# stratified split
def token_level_stratified_sampling(dataset, label_type='ner_tags', train_size=10, seed=42, label_names=None, shuffle=True,
                                    precomputed_indices_file=None):
    """
    Stratified sampling for token-level classification
    We compute a single label for each sequence based on the distinct tags appearing in the sequence
    Note: this approach ignores counts of tags in a sequence
    # Input:
    :param dataset: a huggingface dataset
    :param label_type:
    :param train_size:
    :param seed:
    :param label_names:
    :param shuffle:
    :param precomputed_indices_file:
    :return: a huggingface DatasetDict with a "train" (clean) and "test" (weak) Dataset Object
    """

    inds = np.arange(len(dataset))  # indices: order of data, not their 'id'
    # inds = dataset['id']
    labels = dataset[label_type]
    distinct_labels = [tuple(sorted(set([label_names[y] for y in ys]))) for ys in labels]
    distinct_label_strings = ['_'.join(y) for y in distinct_labels]
    c = Counter(distinct_label_strings)
    distinct_label_strings = [l if c[l] > 1 else 'lowfreq' for l in distinct_label_strings]  # ignore combinations with freq = 1 to avoid errors in train_test_split
    if precomputed_indices_file:
        # Load pre-computed indices from a file. If indices do not exist, then create them and update the file
        # if not os.path.exists(precomputed_indices_file):
        #    test = {}
        #    joblib.dump(test, precomputed_indices_file)
        assert os.path.exists(precomputed_indices_file), "file with precomputed indices does not exist"
        inds_dict = joblib.load(precomputed_indices_file)
        index_str = "seed{}".format(seed)
        if not index_str in inds_dict:
            print("Creating new entry for seed={} in {}".format(seed, precomputed_indices_file))
            clean_inds, weak_inds = tts(inds, stratify=distinct_label_strings, train_size=train_size, random_state=seed, shuffle=shuffle)
            inds_dict[index_str] = {"clean": clean_inds, "weak": weak_inds}
            joblib.dump(inds_dict, precomputed_indices_file)
        else:
            clean_inds, weak_inds = inds_dict[index_str]["clean"], inds_dict[index_str]["weak"]
    else:
        # Compute indices on the fly
        clean_inds, weak_inds = tts(inds, stratify=distinct_label_strings, train_size=train_size, random_state=seed, shuffle=shuffle)

    split = DatasetDict()
    split['train'] = dataset.select(clean_inds)
    split['test'] = dataset.select(weak_inds)
    assert len(set(split['train']['id']) & set(split['test']['id'])) == 0, "issue with splitting data"
    return split