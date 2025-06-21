import os
import torch
import random
import pandas as pd
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)

        # 加载.pt文件中的数据
        data = torch.load(data_path)

        # 获取特征和标签
        features = data['features']
        labels = data['labels']

        return features, labels


class CustomDatasetFromList(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TCGA_Survival(data.Dataset):
    def __init__(self, excel_file, val_ratio=0.33, test_ratio=0.33):
        #print('[dataset] loading dataset from %s' % (excel_file))
        rows = pd.read_csv(excel_file)
        self.rows = self.disc_label(rows)
        label_dist = self.rows['Label'].value_counts().sort_index()
        #print('[dataset] discrete label distribution: ')
        #print(label_dist)
        #print('[dataset] dataset from %s, number of cases=%d' % (excel_file, len(self.rows)))
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

    def get_split(self, fold=0):
        random.seed(1)
        ratio = self.test_ratio
        assert 0 <= fold <= 4, 'fold should be in 0 ~ 4'
        sample_index = random.sample(range(len(self.rows)), len(self.rows))
        num_split = round((len(self.rows) - 1) * ratio)
        if fold < 1 / ratio - 1:
            val_split = sample_index[fold * num_split: (fold + 1) * num_split]
        else:
            val_split = sample_index[fold * num_split:]
        train_split = [i for i in sample_index if i not in val_split]
        split_point = int(len(train_split) * (1-self.val_ratio))
        train_part = train_split[:split_point]
        val_part = train_split[split_point:]
        #print("[dataset] training split: {}, validation split: {}, test split: {}".format(len(train_part), len(val_part), len(val_split)))
        return train_part, val_part, val_split

    def read_WSI(self, path):
        wsi = [torch.load(x) for x in path.split(';')]
        wsi = torch.cat(wsi, dim=0)
        return wsi

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI = case[:5]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        WSI = self.read_WSI(WSI)
        return (ID, WSI, Event, Censorship, Label)

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['Status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['Event'].max() + eps
        q_bins[0] = rows['Event'].min() - eps
        disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'Label', disc_labels)
        return rows


class SequentialSubsetSampler(Sampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):

        return iter(self.indices)

    def __len__(self):

        return len(self.indices)


class BalancedCustomDataset(Dataset):
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = num_samples

        class_0_indices = [i for i, label in enumerate(self.dataset.labels) if label == 0]
        class_1_indices = [i for i, label in enumerate(self.dataset.labels) if label == 1]

        if len(class_0_indices) < num_samples or len(class_1_indices) < num_samples:
            raise ValueError("Not enough samples to create the balanced dataset with the specified number of samples")

        selected_class_0_indices = random.sample(class_0_indices, num_samples)
        selected_class_1_indices = random.sample(class_1_indices, num_samples)

        self.selected_indices = selected_class_0_indices + selected_class_1_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        return self.dataset[original_idx]


def camelyon16_train_and_val(train_and_val_save_path, seed, fewshot_samples, val_ratio, num_workers):
    train_and_val_dataset = CustomDataset(train_and_val_save_path)
    all_features = []
    all_labels = []
    for i in range(len(train_and_val_dataset)):
        features, labels = train_and_val_dataset[i]
        all_features.append(features)
        all_labels.append(labels)

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    for train_index, val_index in sss.split(all_features, all_labels):
        train_data = [all_features[i] for i in train_index]
        train_labels = [all_labels[i] for i in train_index]
        val_data = [all_features[i] for i in val_index]
        val_labels = [all_labels[i] for i in val_index]
    train_dataset = CustomDatasetFromList(train_data, train_labels)
    if fewshot_samples > 0:
        train_dataset = BalancedCustomDataset(train_dataset, fewshot_samples)  # fewshot_samples should less than the number of samples in one class
    val_dataset = CustomDatasetFromList(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def camelyon16_test(test_save_path, seed):
    test_dataset = CustomDataset(test_save_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    return test_loader

def tcga_brca_dataloader(data_save_path, seed, fewshot_samples, val_ratio, num_workers, test_ratio):
    ini_dataset = CustomDataset(data_save_path)
    all_features = []
    all_labels = []
    for i in range(len(ini_dataset)):
        features, labels = ini_dataset[i]
        all_features.append(features)
        all_labels.append(labels)

    if fewshot_samples > 0:
        train_val_test_data = []
        train_val_test_labels = []
        ratio = 1 - (fewshot_samples / len(ini_dataset))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=1048576239)
        for train_index, test_index in sss.split(all_features, all_labels):
            train_val_test_data = [all_features[i] for i in train_index]
            train_val_test_labels = [all_labels[i] for i in train_index]
    else:
        train_val_test_data = all_features
        train_val_test_labels = all_labels

    train_and_val_data = []
    train_and_val_labels = []
    test_data = []
    test_labels = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=1048576239)
    for train_index, test_index in sss.split(train_val_test_data, train_val_test_labels):
        train_and_val_data = [all_features[i] for i in train_index]
        train_and_val_labels = [all_labels[i] for i in train_index]
        test_data = [all_features[i] for i in test_index]
        test_labels = [all_labels[i] for i in test_index]

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    for train_index, val_index in sss.split(train_and_val_data, train_and_val_labels):
        train_data = [train_and_val_data[i] for i in train_index]
        train_labels = [train_and_val_labels[i] for i in train_index]
        val_data = [train_and_val_data[i] for i in val_index]
        val_labels = [train_and_val_labels[i] for i in val_index]

    del train_and_val_data, train_and_val_labels
    train_dataset = CustomDatasetFromList(train_data, train_labels)
    val_dataset = CustomDatasetFromList(val_data, val_labels)
    test_dataset = CustomDatasetFromList(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def tcga_blca_dataloader(csv_path, fewshot_samples, val_ratio, num_workers, test_ratio):
    dataset = TCGA_Survival(excel_file=csv_path, val_ratio=val_ratio, test_ratio=test_ratio)
    train_split, val_split, test_split = dataset.get_split()
    if fewshot_samples > 0:
        train_split = random.sample(train_split, fewshot_samples)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True,
                              sampler=SubsetRandomSampler(train_split))
    val_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True,
                            sampler=SubsetRandomSampler(val_split))
    test_split = test_split[:119]  # the memory is not enough to load the last file
    test_loader = DataLoader(dataset, batch_size=1, pin_memory=True, sampler=SequentialSubsetSampler(test_split))

    return train_loader, val_loader, test_loader

