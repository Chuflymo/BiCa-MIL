import os
import random
from rrt_encoder import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

class PlipAdapter(nn.Module):
    def __init__(self, input_dim, load_length):
        super(PlipAdapter, self).__init__()
        self.L = 512 
        self.D = 128  
        self.K = 1 

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.rrt = RRTEncoder(mlp_dim=512,epeg_k=15,crmsa_k=3)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        # instance_feature -> bag_feature
        feature = self.feature(x)
        feature = self.rrt(feature)
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        bag_feature = torch.mm(A, feature)  # KxL
        norm = bag_feature.norm(dim=-1, keepdim=True)
        norm_bag_feature = bag_feature / norm.expand_as(bag_feature)

        return norm_bag_feature


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)

        data = torch.load(data_path)

        features = data['feature']
        labels = data['label']

        return features, labels, data_path


class PartiallyFilteredCustomDataset(Dataset):
    def __init__(self, original_dataset, remove_zero_label_count=638):
        self.original_dataset = original_dataset
        self.filtered_data_files = []
        removed_count = 0

        for idx in range(len(original_dataset)):
            features, labels, data_path = original_dataset[idx]
            if labels == 0 and removed_count < remove_zero_label_count:
                removed_count += 1
                continue
            self.filtered_data_files.append(data_path)

        assert removed_count == remove_zero_label_count, f"Removed {removed_count} items, but expected to remove {remove_zero_label_count}"

    def __len__(self):
        return len(self.filtered_data_files)

    def __getitem__(self, idx):
        data_path = self.filtered_data_files[idx]

        data = torch.load(data_path)

        features = data['feature']
        labels = data['label']

        return features, labels


class CustomDatasetFromList(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def cache_model_prepare(loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_list = []
    for feature, label in loader:
        labels = label.to(device)
        label_list.extend(labels.to(torch.float))
    train_labels = torch.tensor(label_list).to(device)
    replacement_dict = {
        0: torch.tensor([1, 0]),
        1: torch.tensor([0, 1])
    }
    cache_values = torch.zeros(len(train_labels), 2).to(device)
    for i, value in enumerate(train_labels):
        # noinspection PyUnresolvedReferences
        cache_values[i] = replacement_dict[value.item()]

    return cache_values

def plip_zero_shot(image_features, texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = r"D:\WORK\Projects\VLMs&fine_tune\PLIP_test\plip_finish"
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    processed_texts = processor(text=texts, padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**processed_texts)
    norm = text_features.norm(dim=-1, keepdim=True)
    text_features = text_features / norm.expand_as(text_features)
    logit_scale_init_value = torch.tensor(2.6592).exp()
    logits = 100. * logit_scale_init_value * image_features @ text_features.T

    return logits

def count_matching_elements(list1, list2):
    return sum(1 for x, y in zip(list1, list2) if x == y)

def main(seed):

    prompts = ["This is an H&E image of invasive ductal carcinoma tissue.",
               "This is an H&E image of invasive lobular carcinoma tissue."]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_torch(seed)

    data_save_path = r"D:\DATA\PLIP_MIL\TCGA_features"
    ini_dataset = CustomDataset(data_save_path)
    filtered_dataset = PartiallyFilteredCustomDataset(ini_dataset, remove_zero_label_count=0)
    print(f"Partially filtered dataset length: {len(filtered_dataset)}")
    all_features = []
    all_labels = []
    for i in range(len(filtered_dataset)):
        features, labels = filtered_dataset[i]
        all_features.append(features)
        all_labels.append(labels)

    train_and_val_data = []
    train_and_val_labels = []
    test_data = []
    test_labels = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=1048576239)
    for train_index, test_index in sss.split(all_features, all_labels):
        train_and_val_data = [all_features[i] for i in train_index]
        train_and_val_labels = [all_labels[i] for i in train_index]
        test_data = [all_features[i] for i in test_index]
        test_labels = [all_labels[i] for i in test_index]

    train_data = []
    train_labels = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=seed)
    for train_index, val_index in sss.split(train_and_val_data, train_and_val_labels):
        train_data = [train_and_val_data[i] for i in train_index]
        train_labels = [train_and_val_labels[i] for i in train_index]

    del train_and_val_data, train_and_val_labels
    train_dataset = CustomDatasetFromList(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    test_dataset = CustomDatasetFromList(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    input_dim = 512
    num_classes = 2
    beta = 0.1
    alpha = 0.9
    trainload_length = len(train_loader)
    cache_values = cache_model_prepare(loader=train_loader)
    PlipAdapter_model = torch.load(fr"D:\DATA\PLIP_MIL\TCGA_adaptermil\l1_adapter_mil_model_{seed}.pt").to(device)
    best_auc_score = 0.0
    best_loss = float('inf')
    patient_loss = float('inf')
    patient_count = 0
    PlipAdapter_model.eval()
    predictions = []
    true_labels = []
    predicted_probs = []
    with torch.no_grad():
        cache_keys_lst = []
        for feature, label in train_loader:
            train_feature, train_label = feature.to(device), label.to(device)
            train_feature = train_feature.squeeze(0)
            bag_train_feature = PlipAdapter_model(train_feature)
            cache_keys_lst.append(bag_train_feature.squeeze().tolist())
        cache_keys = torch.tensor(cache_keys_lst).to(device)
        for batch_features, batch_labels in test_loader:
            feature, label = batch_features.to(device), batch_labels.to(device)
            feature = feature.squeeze(0)
            bag_test_feature = PlipAdapter_model(feature)
            similarity = bag_test_feature @ cache_keys.t()
            affinity = (beta * (similarity - 1)).exp()
            temp_logits = affinity @ cache_values
            cache_model_logits = alpha * temp_logits
            test_zero_shot_logits = plip_zero_shot(image_features=bag_test_feature,
                                                   texts=prompts)
            total_logits =  cache_model_logits + test_zero_shot_logits
            pred = total_logits.topk(1, 1, True, True)[1].t().squeeze(0)
            true_labels.extend(label.cpu())
            predictions.extend(pred.detach().cpu())
            predicted_probs.extend(torch.softmax(total_logits, dim=1)[:, 1].cpu().numpy())

        matching_count = count_matching_elements(predictions, true_labels)
        pre_prob = matching_count / len(true_labels)
        auc_score = roc_auc_score(true_labels, predicted_probs)
        f1 = f1_score(true_labels, predictions)
        precision_sc = precision_score(true_labels, predictions)
        recall_sc = recall_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)

    print(f"{seed} Accuracy: {pre_prob:.4f}")
    print(f"{seed} AUC: {auc_score:.4f}")
    print(f"{seed} F1 Score: {f1:.4f}")
    print(f"{seed} Precision: {precision_sc:.4f}")
    print(f"{seed} Recall: {recall_sc:.4f}")
    print(f"{seed} Specificity: {specificity:.4f}")
    print(f"{seed} NPV: {npv:.4f}\n")


if __name__ == '__main__':
    main(seed=2021)
