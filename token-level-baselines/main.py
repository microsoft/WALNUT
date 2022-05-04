import argparse
import os
import numpy as np
from numpy.random import seed
import random
import pandas as pd
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
from datasets import ClassLabel, Sequence, DatasetDict
from datetime import datetime
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from Logger import get_logger, close
import joblib
from utils import load_label_names, token_level_stratified_sampling
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def split_clean_weak(args, logger):
    logger.info('creating a clean/weak split')
    dataset_folder = os.path.join(args.datapath, args.dname)
    datasets = load_dataset('json', data_files={
        'train': '{}/analysis/train.json'.format(dataset_folder),
        'validation': '{}/analysis/validation.json'.format(dataset_folder),
        'test': '{}/analysis/test.json'.format(dataset_folder)
    })
    datasets = datasets.rename_column('labels', 'ner_tags')
    datasets = datasets.rename_column('agg_label_majority', 'mv')
    datasets = datasets.rename_column('agg_label_snorkel', 'nb')

    class_name_fpath = os.path.join(dataset_folder, 'analysis/class_names.txt')

    name2ind = {n:i for i,n in enumerate(label_names)}
    ind2name = {i:n for n,i in name2ind.items()}
    precomputed_indices_file = os.path.join(args.datapath, "{}/{}_indices.pkl".format(args.dname, args.dname))
    num_labels = len(label_names)
    logger.info("train_frac = {} * {} = {}".format(args.train_frac, num_labels, args.train_frac * num_labels))
    args.train_frac *= num_labels

    logger.info("Creating pre-computed indices for seed={} in {}".format(args.seed, precomputed_indices_file))

    # Run this for the first time to generate indices and save them into files
    for seed in [0, 20, 7, 1993, 42]:
        split = token_level_stratified_sampling(datasets['train'], train_size=args.train_frac, shuffle=True, seed=seed,
                                        label_names=label_names, precomputed_indices_file=precomputed_indices_file,
                                        create_clean_weak_splits=True)
        print(split)

    return {
        'clean': len(split['train']),
        'weak': len(split['test'])
    }


def train_base_model(args, logger):
    # Train the base model
    logger.info('training base model on labels: {}'.format(args.label_name))
    label_all_tokens = True

    # Load dataset
    dataset_folder = os.path.join(args.datapath, args.dname)
    datasets = load_dataset('json', data_files={
        'train': '{}/analysis/train.json'.format(dataset_folder),
        'validation': '{}/analysis/validation.json'.format(dataset_folder),
        'test': '{}/analysis/test.json'.format(dataset_folder)
    })
    datasets = datasets.rename_column('labels', 'ner_tags')
    datasets = datasets.rename_column('agg_label_majority', 'mv')
    datasets = datasets.rename_column('agg_label_snorkel', 'nb')
    print(datasets)

    class_name_fpath = os.path.join(dataset_folder, 'analysis/class_names.txt')
    label_names = load_label_names(class_name_fpath)
    label_list = label_names

    def print_stats(datasets):
        for label_type in ['labels']:
            for method in ['train', 'validation', 'test']:
                labels = datasets[method][label_type]
                labels = [y for ys in labels for y in ys]
                df = pd.DataFrame()
                df['tags'] = labels
                df['tags'] = df['tags'].map(lambda x: label_names[x])
                logger.info("{}-{} ({})\n{}".format(label_type, method, df['tags'].shape[0], df['tags'].value_counts()))

    logger.info("Original datasets:\n{}".format(datasets))

    if args.train_frac > 1:
        # converting #examples/class to #examples (only for token-level classification)
        num_labels = len(label_names)
        logger.info("train_frac = {} * {} = {}".format(args.train_frac, num_labels, args.train_frac * num_labels))
        args.train_frac *= num_labels

        if args.walnut_mode:
            # This was done only for the training size experiment
            if not args.clean_plus_weak and args.label_name != 'ner_tags':
                # this is the weak approach: we remove the clean labeled data from the weak set.
                logger.info("WEAK approach: keeping {} - {} = {} train examples".format( len(datasets['train']), args.train_frac, len(datasets['train']) - args.train_frac))
                args.train_frac = len(datasets['train']) - args.train_frac

        if args.train_frac <= 0:
            logger.info("train_frac = {} < 0 skipping...".format(args.train_frac))
            return {}

        if args.train_frac > len(datasets['train']):
            logger.info("train_frac = {} > {} skipping...".format(args.train_frac, len(datasets['train'])))
            return {}

    if args.clean_plus_weak:
        # %clean = train_frac, %weak = (1 - train_frac)
        weak_perc = 1 - args.train_frac if args.train_frac < 1 else len(datasets['train']) - args.train_frac
        logger.info("Combining Clean ({}) + Weak ({})".format(args.train_frac, weak_perc))

        if args.walnut_mode:
            # Use custom token-level stratified sampling function and save indices
            precomputed_indices_file = os.path.join(args.datapath, "{}/{}_indices.pkl".format(args.dname, args.dname))
            logger.info("Loading pre-computed indices for seed={} in {}".format(args.seed, precomputed_indices_file))
            split = token_level_stratified_sampling(datasets['train'], train_size=args.train_frac, shuffle=True, seed=args.seed,
                                                    label_names=label_names, precomputed_indices_file=precomputed_indices_file,
                                                    create_clean_weak_splits=False)
        else:
            split = datasets['train'].train_test_split(train_size=args.train_frac, shuffle=True, seed=args.seed)

        logger.info("CLEAN={}\tWEAK={}".format(len(split['train']),len(split['test'])))
        split['train'] = split['train'].rename_column("ner_tags", "labels")
        split['test'] = split['test'].rename_column(args.label_name, "labels")

        # bring datasets into same schema
        split['train'] = split['train'].remove_columns(args.label_name)
        split['test'] = split['test'].remove_columns("ner_tags")
        split['train'] = split['train'].cast(split['test'].features)
        
        datasets['train'] = concatenate_datasets([split['train'], split['test']])
        datasets['train'] = datasets['train'].shuffle(seed=args.seed)

    else:
        # Using just clean / weak labels
        if args.walnut_mode:
            # Use custom token-level stratified sampling function and save indices
            precomputed_indices_file = os.path.join(args.datapath, "{}/{}_indices.pkl".format(args.dname, args.dname))
            logger.info("Loading pre-computed indices for seed={} in {}".format(args.seed, precomputed_indices_file))
            split = token_level_stratified_sampling(datasets['train'], train_size=args.train_frac, shuffle=True,
                                                    seed=args.seed, label_names=label_names,
                                                    precomputed_indices_file=precomputed_indices_file,
                                                    create_clean_weak_splits=False)
            datasets['train'] = split['train'] if args.label_name == 'ner_tags' else split['test']  # clean (train) vs. weak (test)
        elif args.train_frac != 1:
            split = datasets['train'].train_test_split(train_size=args.train_frac, shuffle=True, seed=args.seed)
            datasets['train'] = split['train'] if args.label_name == 'ner_tags' else split['test']  # clean (train) vs. weak (test)
        else:
            # clean/weak approach: in either case we take datasets['train'] and use clean/weak labels
            # no need to change anything int he training side
            pass

        datasets['train'] = datasets['train'].rename_column(args.label_name, "labels")

    # For the validation and test sets, set "labels" column to clean labels
    datasets['validation'] = datasets['validation'].rename_column("ner_tags", "labels")
    datasets['test'] = datasets['test'].rename_column("ner_tags", "labels")

    logger.info("Pre-processed datasets for training base model:\n{}".format(datasets))
    print_stats(datasets)

    #label_list = datasets['test'].features["labels"].feature.names
    logger.info(label_list)

    # Pre-process data
    if "roberta" in args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)  # for RoBERTa.
    elif "lstm" in args.base_model:
        glovefile = "glove.6B.{}d.txt.pkl".format(getattr(args, "lstm_embed_dim", 50))
        args.glove_data_path = os.path.join(args.datapath, "glove_embeds/{}".format(glovefile))
        args.random_seed = args.seed
        import pickle
        glove_data = pickle.load(open(args.glove_data_path, 'rb'))
        glove_map = {i[0]: index + 1 for index, i in enumerate(glove_data)}
        glove_weight = np.stack([np.zeros((glove_data[0][1].size)), *[i[1] for i in glove_data]], axis=0)
        from utils import bilstmTokenizer
        tokenizer = bilstmTokenizer(glove_map=glove_map)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_labels_lstm(examples):
        tokenized_inputs = tokenizer.glove_tokenize(examples["tokens"], labels=examples['labels'])
        tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'][0]
        tokenized_inputs["labels"] = tokenized_inputs['labels'][0]
        return tokenized_inputs

    if 'lstm' in args.base_model:
        tokenized_datasets = datasets.map(tokenize_and_align_labels_lstm)
        from utils import LSTM_text
        hx_dim = 128
        model = LSTM_text(len(label_list), h_dim=hx_dim, embed_weight=glove_weight, is_ner=True)
    else:
        tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
        model = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=len(label_list))

    # Define trainer
    trainer_args = TrainingArguments(
        output_dir = args.logdir + 'checkpoints',
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        seed=args.seed,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Train model
    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()

    # Test model
    trainer.evaluate()
    predictions, labels, _ = trainer.predict(tokenized_datasets[args.test_mode])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


def train_size_experiment(args, logger):
    all_res = []
    train_samples_per_class = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

    for frac in train_samples_per_class:
        try:
            logger.info("\n\n\t\t TRAIN FRAC = {}".format(frac))
            args.train_frac = frac
            res = train_base_model(args, logger)
            res['frac'] = frac
            all_res.append(res)
            logger.info("Results:\n{}".format(res))
        except:
            logger.info("skipping {}".format(frac))
            pass
    return all_res


def teacher(args, logger):
    # Directly evaluates the rules applied on the test set and aggregated via majority voting
    # No base model is trained

    # Load dataset
    if args.dname == 'CoNLL':
        savepath = os.path.join(args.datapath, "CoNLL/conll2003_weak_labels")
        datasets = load_from_disk(savepath)
    else:
        dpath = os.path.join(args.datapath, "{}/{}.py".format(args.dname, args.dname))
        datasets = load_dataset(dpath)

    logger.info(datasets)
    label_list = datasets['train'].features["ner_tags"].feature.names
    logger.info(label_list)

    metric = load_metric("seqeval")

    def get_labels(x):
        return [[label_list[i] for i in labels] for labels in x]

    if args.label_name != 'all_weak_labels':
        for test_mode in ['train', 'validation', 'test']:
            logger.info("\n\t **** {} results **** ".format(test_mode))
            teacher_predictions = get_labels(datasets[test_mode][args.label_name])
            true_labels = get_labels(datasets[test_mode]["ner_tags"])
            results = metric.compute(predictions=teacher_predictions, references=true_labels)
            for m in ['overall_precision', 'overall_recall', 'overall_f1']:
                logger.info("{}={:.3f}".format(m, results[m]))
    else:
        # Evaluating each rule separately
        #weak_label_name = 'all_weak_labels' if args.dname != 'CoNLL' else 'weak_labels'
        weak_label_name='weak_labels'
        num_rules = len(datasets['train'][0][weak_label_name][0])

        all_res = {}
        for test_mode in ['train', 'validation', 'test']:
            all_res[test_mode] = []
            logger.info("\n\t **** {} results ({} rules) **** ".format(test_mode, num_rules))
            all_latex_str = []
            rule_df = pd.DataFrame()
            for rule_ind in range(num_rules):
                teacher_predictions = get_labels(
                    [[y[rule_ind] for y in x] for x in datasets[test_mode][weak_label_name]])
                true_labels = get_labels(datasets[test_mode]["ner_tags"])
                results = metric.compute(predictions=teacher_predictions, references=true_labels)
                all_res[test_mode].append(results)
                logger.info(
                    "rule {}: precision={:.3f}\trecall={:.3f}\tF1={:.3f}".format(rule_ind, results['overall_precision'],
                                                                                 results['overall_recall'],
                                                                                 results['overall_f1']))
                latex_str = "rule {} & {:.3f} & {:.3f} & {:.3f} \\\\".format(rule_ind, results['overall_precision'], results['overall_recall'], results['overall_f1'])
                all_latex_str.append(latex_str)
                rule_df = rule_df.append(pd.Series({
                    'precision': results['overall_precision'],
                    'recall': results['overall_recall'],
                    'f1': results['overall_f1']
                }), ignore_index=True)
            import seaborn as sns
            sns.set()
            sns.set_style("whitegrid")
            import matplotlib.pyplot as plt
            rule_df.plot.scatter(x='recall', y='precision')  # , c='f1') #, c='good_prec')
            plt.savefig('rule_scatterplot_{}_{}.pdf'.format(args.dname, test_mode))

            logger.info("latex report: \n{}".format("\n".join(all_latex_str)))
        savefile = os.path.join(args.logdir, 'rule_performance_results.pkl')
        joblib.dump(all_res, savefile)
    return results

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dname", help="Dataset name", type=str, default='NCBI')
    parser.add_argument("--base_model", help="Huggingface (base) model", type=str, default='distilbert-base-uncased')
    parser.add_argument("--label_name", help="Dataset name", type=str, default='ner_tags')  # mv
    parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default='../data/Token-level')
    parser.add_argument("--experimentsdir", help="Directory to save experiments", type=str, default='./experiments')
    parser.add_argument("--savefolder", help="name of savefolder", type=str, default='./')
    parser.add_argument('--seed', help="Random seed (default: 42)", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Run debugging experiment")
    parser.add_argument('--batch_size', help="Batch size (for base model)", type=int, default=16)
    parser.add_argument('--epochs', help="Number of train epochs (default: 10)", type=int, default=10)
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--train_frac", default=1.0, type=float, help="Fraction of the training data to keep")
    parser.add_argument("--train_size_experiment", action="store_true", help="Training size experiment")
    parser.add_argument("--teacher", action="store_true", help="Evaluate the teacher")
    parser.add_argument("--clean_plus_weak", action="store_true", help="Clean plus weak mode: concatenating clean and weak data (note: if train_frac=1 or label_name='ner_tags' then this method reduces to clean)")
    parser.add_argument("--test_mode", help="Testing mode (validation / test set)", type=str, default='test')
    parser.add_argument("--walnut_mode", action="store_true", help="Use pre-specified hyperparameters for WALNUT benchmark")
    parser.add_argument("--create_clean_weak_splits", action="store_true", help="Create clean/weak splits. (This is needed just once for a dataset)")

    args = parser.parse_args()
    if args.train_frac > 1:
        args.train_frac = int(args.train_frac)

    set_seed(args.seed)
    args.random_seed = args.seed

    if args.walnut_mode:
        # specify parameters
        train_fracs = {
            "CoNLL": 20,
            "NCBI": 20,
            "BC5CDR": 10,
            "LaptopReview": 50,
            "wikigold": 40,
            "ontonotes": 40,
            "mit-restaurants": 20,
            "mit-movies": 20
        }
        args.train_frac = int(train_fracs[args.dname])
        assert args.dname in train_fracs, "{} is not supported in WALNUT experiments or no clean/weak splits available are. \
            If you are running experiments for the first time, then you need to decide the number of labeled data per class and then add the --create_clean_weak_splits parameter"
    if args.clean_plus_weak and ((args.train_frac == 1) or (args.label_name == 'ner_tags')):
        raise(BaseException('argument clean_plus_weak makes no sense for train_frac=1 or using "ner_tags" (clean labels) for weak label '))
    #if args.dname == 'CoNLL' and args.label_name == 'mv':
    #    args.label_name = 'majority_label'

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H_%M")
    if args.debug:
        args.logdir = './debug/'
    else:
        args.logdir = os.path.join(os.path.abspath(args.experimentsdir), args.dname)
        args.logdir = os.path.abspath(os.path.join(args.logdir, args.savefolder))
        args.logdir = os.path.join(args.logdir,"{}_{}_{}_epoch{}_seed{}_frac{}_lr{}".format(date_time, args.base_model, args.label_name, args.epochs, args.seed, args.train_frac, args.lr))
        if args.teacher:
            args.logdir += '_teacher'
        if args.clean_plus_weak:
            args.logdir += '_clean_plus_weak'
        if args.walnut_mode:
            args.logdir += '_WALNUT'
        if args.base_model != 'distilbert-base-uncased':
            args.logdir += '_{}'.format(args.base_model)
    os.makedirs(args.logdir, exist_ok=True)

    logger = get_logger(logfile=os.path.join(args.logdir, 'log.log'))

    if args.create_clean_weak_splits:
        logger.info("\n\n\t\t *** SPLITTING DATASET TO TRAIN AND TEST ***\nargs={}".format(args))
        res = split_clean_weak(args, logger)
    elif args.train_size_experiment:
        logger.info("\n\n\t\t *** TRAINING SIZE EXPERIMENT ***\nargs={}".format(args))
        res = train_size_experiment(args, logger)
    elif args.teacher:
        logger.info("\n\n\t\t *** EVALUATING THE TEACHER ***\nargs={}".format(args))
        res = teacher(args, logger)
    else:
        logger.info("\n\n\t\t *** NEW EXPERIMENT ***\nargs={}".format(args))
        res = train_base_model(args, logger)

    logger.info("All Results:\n{}".format(res))
    respath = os.path.join(args.logdir, 'res.pkl')
    logger.info("results stored at {}".format(respath))
    joblib.dump(res, respath)
    close(logger)


if __name__ == "__main__":
    main()
