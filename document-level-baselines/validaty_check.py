import torch

random_seed = 123
clean_count=5
for seed in [123, 456, 79, 10, 49]:
    for clean_count in [5, 10, 20, 25, 50]:
        task="agnews"
        base_path = "/home/kshu/PycharmProjects/MetaReduce-main/data/{}".format(task)
        file_path = "{}/{}_organized_nb_train_{}_{}.index".format(base_path, task, clean_count, random_seed)
        data_file_path = "{}/{}_organized_nb_train_bert-base-uncased.torch".format(base_path, task)



        weak_index = torch.load(file_path)['weak_index']
        clean_index = torch.load(file_path)['clean_index']
        new_weak_index = torch.load(file_path+"1")['weak_index']
        new_clean_index = torch.load(file_path)['clean_index']
        assert len(set(clean_index+new_clean_index).difference(set(clean_index))) + len(set(clean_index+new_clean_index).difference(set(new_clean_index))) == 0
        print("Before")
        print(len(weak_index))
        print(len(new_weak_index))
        print(len(set(weak_index).difference(set(new_weak_index))))

        data = torch.load(data_file_path)
        lf_labels = data['lf']
        valid_lf_count = torch.sum(torch.where(lf_labels == -1, torch.zeros_like(lf_labels),
                                                           torch.ones_like(lf_labels)), dim=1)
        # at least two valiable index file
        lf_index = torch.nonzero(valid_lf_count > 1).squeeze(1).tolist()
        weak_index = list(set(weak_index).intersection(set(lf_index)))

        print("After")
        # print(len(weak_index))
        # print(len(new_weak_index))
        # print(set(weak_index).difference(set(new_weak_index)))
        assert len(set(weak_index+new_weak_index).difference(set(new_weak_index))) + len(set(weak_index+new_weak_index).difference(set(weak_index))) == 0
        print("*"*100)