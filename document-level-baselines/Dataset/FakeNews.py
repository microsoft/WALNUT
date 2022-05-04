class FakeNewsDataset(torch.utils.data.Dataset):
    # only for fake news dataset
    def __init__(self, hparams, train_status, is_only_clean=None, elmo_map=None):
        file_path = hparams.file_path
        clean_ratio = getattr(hparams, "clean_ratio", -1)
        # dummy code to redirect the file path
        if hparams.is_transformer:
            read_file = file_path + "/" + train_status + "_{}.torch".format(hparams.transformer_model_name)
            tokenizer = AutoTokenizer.from_pretrained(hparams.transformer_model_name)
        else:
            read_file = file_path + "/" + train_status + "_glove.torch"
            tokenizer = partial(glove_tokenize_text, elmo_map)
        if is_only_clean is None:
            is_only_clean = getattr(hparams, "is_only_clean", False)
        self.is_training = train_status == "train"
        if hparams.is_overwrite_file or os.path.exists(read_file) is False:
            # weak data
            # data = torch.load(file_path)
            if train_status == "train":
                weak_data = load_fake_news(file_path+"/noise_0.1.csv")
                clean_data = load_fake_news(file_path+"/gold_0.1.csv")
                # TODO: support the fake news dataset.
                # if clean_ratio > 0:
                #     clean_data
                weak_data = {**tokenizer_text(weak_data['news'], tokenizer, max_length=256), **weak_data}
                clean_data = {**tokenizer_text(clean_data['news'], tokenizer, max_length=256), **clean_data}
                data_interest = {"clean": clean_data, "weak": weak_data}
            elif train_status == "test":
                data_interest = load_fake_news(file_path+"/test.csv")
                data_interest = {**tokenizer_text(data_interest['news'], tokenizer, max_length=256), **data_interest}
            else:
                data_interest = load_fake_news(file_path + "/val.csv")
                data_interest = {**tokenizer_text(data_interest['news'], tokenizer, max_length=256), **data_interest}

            # data_interest = {**tokenizer_text(data_interest['text'], tokenizer), **data_interest}
            torch.save(data_interest, read_file)
        else:
            data_interest = torch.load(read_file)

        is_agg_weak = getattr(hparams, "is_agg_weak", False)
        is_flat = getattr(hparams, "is_flat", False)

        assert is_agg_weak + is_flat != 2, "Don't flat and aggregate weak labels at the same time."

        if train_status == "train":
            clean_data = data_interest['clean']
            # placeholder
            clean_data['lf'] = [[-1, -1, -1]] * len(clean_data['label'])
            weak_data = data_interest['weak']

            if is_agg_weak:
                # replace the true label with aggergate weak labels
                weak_labels = np.array([list(map(int, weak_data[key])) for key in weak_data if "_label" in key]).T
                agg_label = label_aggregation(weak_labels, hparams.agg_fn_str,
                                                            class_count=2,
                                                            snorkel_ckpt_file=hparams.snorkel_ckpt_file)
                # replace the clean label with the aggregated weak label
                weak_data['lf'] = weak_labels.tolist()
                weak_data['label'] = agg_label

            # check whether only clean or only weak
            if getattr(hparams, "is_concat_weak_clean", True):
                if len(clean_data) == 0:
                    # no clean data
                    self.data_interest = weak_data
                else:
                    all_data = {}
                    for key in weak_data:
                        try:
                            all_data[key] = weak_data[key] + clean_data[key]
                        except:
                            th = 1
                    self.data_interest = all_data
            else:
                if is_only_clean:
                    self.data_interest = clean_data
                else:
                    self.data_interest = weak_data
        else:
            # placeholder
            data_interest['lf'] = [[-1, -1, -1]] * len(data_interest['label'])
            self.data_interest = data_interest

    def __len__(self):
        return len(self.data_interest['input_ids'])

    def __getitem__(self, item):
        # ATTENTION: 256 tokens take much longer time than expectation.
        input_ids = torch.tensor(self.data_interest['input_ids'][item][:256], dtype=torch.long)
        attention_mask = torch.tensor(self.data_interest['attention_mask'][item][:256])
        label = torch.tensor(self.data_interest['label'][item], dtype=torch.long)
        weak_labels = torch.tensor(self.data_interest['lf'][item], dtype=torch.long)
        return input_ids, attention_mask, label, weak_labels

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        data_parser.add_argument("--file_path", type=str, required=True)
        data_parser.add_argument("--is_agg_weak", action="store_true")
        data_parser.add_argument("--is_overwrite_file", action="store_true")
        data_parser.add_argument("--agg_fn_str", choices=['most_vote', 'snorkel'], default="most_vote")
        data_parser.add_argument("--snorkel_ckpt_file", type=str, default="")
        data_parser.add_argument("--is_concat_weak_clean", action="store_true")
        data_parser.add_argument("--is_only_clean", action="store_true")
        return data_parser
