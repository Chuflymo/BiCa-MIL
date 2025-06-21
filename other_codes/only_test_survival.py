import random
from tqdm import tqdm
from rrt_encoder import *
import torch.nn.functional as F
from survival_data_process import *
from torch.utils.data.sampler import Sampler
from transformers import CLIPModel, CLIPProcessor
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, SubsetRandomSampler


class PlipAdapter(nn.Module):
    def __init__(self, input_dim, load_length):
        super(PlipAdapter, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        # 特征提取层
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.rrt = RRTEncoder(mlp_dim=512,epeg_k=15,crmsa_k=3)

        # 注意力层
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


class DAttention(nn.Module):
    def __init__(self, input_dim, n_classes, dropout, act):
        super(DAttention, self).__init__()
        self.L = 512  # 512
        self.D = 128  # 128
        self.K = 1
        self.feature = [nn.Linear(input_dim, 512)]
        #self.rrt = RRTEncoder(mlp_dim=512,epeg_k=15,crmsa_k=3)

        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

        self.apply(initialize_weights)

    def forward(self, x, return_attn=False, no_norm=False):
        feature = self.feature(x)

        feature = feature.squeeze(0)
       # feature = self.rrt(feature)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        logits = self.classifier(M)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S


class SequentialSubsetSampler(Sampler):
    def __init__(self, indices):

        super().__init__()
        self.indices = indices

    def __iter__(self):

        return iter(self.indices)

    def __len__(self):

        return len(self.indices)


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

def count_matching_elements(list1, list2):
    return sum(1 for x, y in zip(list1, list2) if x == y)

def cache_model_prepare(loader):
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

def plip_zero_shot(image_features, texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = r"D:\WORK\Projects\PLIP_test\plip_finish"
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    processed_texts = processor(text=texts, padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**processed_texts)
    norm = text_features.norm(dim=-1, keepdim=True)
    text_features = text_features / norm.expand_as(text_features)
    logit_scale_init_value = torch.tensor(2.6592).exp()
    logits = 100. * logit_scale_init_value * image_features @ text_features.T

    return logits

def ab_main(seed, lr):
    seed_torch(seed)

    csv_path = r"D:\DATA\PLIP_MIL\Survival_data\BLCA_Splits.csv"
    dataset = TCGA_Survival(csv_path)
    train_split, val_split, test_split = dataset.get_split()
    test_split = test_split[:119]
    test_loader = DataLoader(dataset, batch_size=1, pin_memory=True,sampler=SequentialSubsetSampler(test_split))

    input_dim = 512
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    abmil_model = torch.load(fr'D:\DATA\PLIP_MIL\Survival_abmil\{lr}_ab_mil_{seed}.pt').to(device)

    abmil_model.eval()
    total_loss = 0.0
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    with torch.no_grad():
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(test_loader):
            data_WSI = data_WSI.cuda()
            data_Label = data_Label.type(torch.LongTensor).cuda()
            data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()

            hazards, S = abmil_model(data_WSI)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event

        test_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print(f"{seed} Test C-Index:", test_c_index)

def adapter_main(seed, lr):
    seed_torch(seed)
    prompts = [
        "This image is from a patient with low-grade bladder cancer, where the tumor cells have minimal atypia and closely resemble normal tissue. The prognosis is excellent.",
        "This image is from a patient with moderate-grade bladder cancer, where the tumor cells have some degree of atypia but still retain some normal tissue characteristics. The prognosis is moderate.",
        "This image is from a patient with high-grade bladder cancer, where the tumor cells exhibit notable heterogeneity and aggressive characteristics. The prognosis is poor.",
        "This image is from a patient with extremely high-grade bladder cancer, where the tumor cells show significant heterogeneity and aggressive behavior. The prognosis is very poor."]

    csv_path = r"D:\DATA\PLIP_MIL\Survival_data\BLCA_Splits.csv"
    dataset = TCGA_Survival(csv_path)
    train_split, val_split, test_split = dataset.get_split()
    test_split = test_split[:119]
    train_loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True,
                              sampler=SubsetRandomSampler(train_split))
    cache_values = cache_model_prepare(loader=train_loader)
    test_loader = DataLoader(dataset, batch_size=1, pin_memory=True, sampler=SequentialSubsetSampler(test_split))

    input_dim = 512
    beta = 1
    alpha = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter_model = torch.load(fr'D:\DATA\PLIP_MIL\Survival_adaptermil\{lr}_adapter_mil_{seed}.pt').to(device)

    adapter_model.eval()
    total_loss = 0.0
    total_loss = 0.0
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    with torch.no_grad():
        dataloader = tqdm(train_loader)
        cache_keys_lst = []
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.cuda()
            train_feature = data_WSI.squeeze(0)
            bag_train_feature = adapter_model(train_feature)
            cache_keys_lst.append(bag_train_feature.squeeze().tolist())
        cache_keys = torch.tensor(cache_keys_lst).to(device)
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(test_loader):
            data_WSI = data_WSI.cuda()
            data_Label = data_Label.type(torch.LongTensor).cuda()
            data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            test_feature = data_WSI.squeeze(0)
            bag_test_feature = adapter_model(test_feature)
            similarity = bag_test_feature @ cache_keys.t()
            affinity = (beta * (similarity - 1)).exp()
            temp_logits = affinity @ cache_values
            cache_model_logits = alpha * temp_logits
            test_zero_shot_logits = plip_zero_shot(image_features=bag_test_feature,
                                                  texts=prompts)
            total_logits = (cache_model_logits + test_zero_shot_logits) / 1000
            hazards = torch.sigmoid(total_logits)
            S = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event

        test_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                                  tied_tol=1e-08)[0]
        print(f"{seed} Test C-Index:", test_c_index)



if __name__ == '__main__':
    seeds = [2023,2024,2025]
    for seed in seeds:
        adapter_main(seed, 'l2')
