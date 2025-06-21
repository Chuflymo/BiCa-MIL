import os
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # survival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    max_index = S.size(1) - 1
    Y = Y.clamp(min=0, max=max_index)
    Y = Y.long()
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    for feature, label in tqdm(loader):
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

def cache_model_prepare_survival(loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_list = []
    dataloader = tqdm(loader)
    for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
        labels = data_Label.to(device)
        label_list.extend(labels.to(torch.float))
    train_labels = torch.tensor(label_list).to(device)
    replacement_dict = {
        0: torch.tensor([0, 0, 0, 1]),
        1: torch.tensor([0, 0, 1, 0]),
        2: torch.tensor([0, 1, 0, 0]),
        3: torch.tensor([1, 0, 0, 0])
    }
    cache_values = torch.zeros(len(train_labels), 4).to(device)
    for i, value in enumerate(train_labels):
        # noinspection PyUnresolvedReferences
        cache_values[i] = replacement_dict[value.item()]

    return cache_values

def plip_zero_shot(image_features, texts, plip_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plip_model_path = plip_model_path
    model = CLIPModel.from_pretrained(plip_model_path).to(device)
    processor = CLIPProcessor.from_pretrained(plip_model_path)
    processed_texts = processor(text=texts, padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**processed_texts)
    norm = text_features.norm(dim=-1, keepdim=True)
    text_features = text_features / norm.expand_as(text_features)
    logit_scale_init_value = torch.tensor(2.6592).exp()
    logits = 100. * logit_scale_init_value * image_features @ text_features.T

    return logits

def count_matching_elements(list1, list2):
    return sum(1 for x, y in zip(list1, list2) if x == y)

def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions

def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def classifier_test(true_labels, predicted_probs, predictions, seed):
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

def Mean_pooling(feature):
    bag_feature = torch.mean(feature, dim=0, keepdim=True)
    norm = bag_feature.norm(dim=-1, keepdim=True)
    norm_bag_feature = bag_feature / norm.expand_as(bag_feature)

    return norm_bag_feature