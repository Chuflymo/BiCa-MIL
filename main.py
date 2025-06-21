import copy
import argparse
from utils import *
from modules import *
from dataloader import *
import torch.optim as optim
from sksurv.metrics import concordance_index_censored
from torch.optim.lr_scheduler import CosineAnnealingLR


def main(args):
    seed_torch(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #set dataset
    if args.datasets.lower() == 'camelyon16':
        train_and_val_path = os.path.join(args.dataset_root, 'train_and_val')
        test_path = os.path.join(args.dataset_root, 'test')
        train_loader, val_loader = camelyon16_train_and_val(train_and_val_path, seed=args.seed, fewshot_samples=args.fewshot_samples, val_ratio=args.val_ratio, num_workers=args.num_workers)
        test_loader = camelyon16_test(test_path, seed=args.seed)

    elif args.datasets.lower() == 'tcga-brca':
        data_path = args.dataset_root
        train_loader, val_loader, test_loader = tcga_brca_dataloader(data_path, seed=args.seed, fewshot_samples=args.fewshot_samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, num_workers=args.num_workers)

    elif args.datasets.lower() == 'tcga-blca':
        # tcga-blca is used for survival analysis
        data_path = args.dataset_root
        csv_path = os.path.join(data_path, 'BLCA_Splits.csv') # you need a csv file for this dataset
        train_loader, val_loader, test_loader = tcga_blca_dataloader(csv_path, fewshot_samples=args.fewshot_samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, num_workers=args.num_workers)

    #set model
    if args.model.lower() == 'ab_mil':
        model = ABMIL(input_dim=args.input_dim, n_classes=args.n_classes).to(device)

    elif args.model.lower() == 'rrt_mil':
        model = RRTMIL(input_dim=args.input_dim, n_classes=args.n_classes, epeg_k=args.epeg_k, crmsa_k=args.crmsa_k).to(device)

    elif args.model.lower() == 'adapter_mil' or args.model.lower() == 'text_mil':
        model = Generator(input_dim=args.input_dim, epeg_k=args.epeg_k, crmsa_k=args.crmsa_k).to(device)

    #set optimizer,scheduler,criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    if args.datasets.lower() == 'tcga-blca':
        criterion = NLLSurvLoss(alpha=0.0) # for survival analysis
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=0)

    if args.datasets.lower() == 'tcga-blca':
        if args.model.lower() == 'adapter_mil':
            survival_adapter(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device, switch=True)
        elif args.model.lower() == 'text_mil':
            survival_adapter(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device, switch=False)
        elif args.model.lower() == 'ab_mil' or args.model.lower() == 'rrt_mil':
            survival_ab(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device)
    else:
        if args.model.lower() == 'adapter_mil':
            classifier_adapter(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device, switch=True)
        elif args.model.lower() == 'text_mil':
            classifier_adapter(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device, switch=False)
        elif args.model.lower() == 'ab_mil' or args.model.lower() == 'rrt_mil':
            classifier_ab(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device)

def survival_adapter(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device, switch=True):
    prompts = [
        "This image is from a patient with low-grade bladder cancer, where the tumor cells have minimal atypia and closely resemble normal tissue. The prognosis is excellent.",
        "This image is from a patient with moderate-grade bladder cancer, where the tumor cells have some degree of atypia but still retain some normal tissue characteristics. The prognosis is moderate.",
        "This image is from a patient with high-grade bladder cancer, where the tumor cells exhibit notable heterogeneity and aggressive characteristics. The prognosis is poor.",
        "This image is from a patient with extremely high-grade bladder cancer, where the tumor cells show significant heterogeneity and aggressive behavior. The prognosis is very poor."]

    init_alpha = args.init_alpha
    init_beta = args.init_beta
    num_epochs = args.epochs
    patient_epochs = args.patients
    cache_values = cache_model_prepare_survival(loader=train_loader)
    best_c_index = 0.0
    best_loss = float('inf')
    patient_loss = float('inf')
    patient_count = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        model.train()
        if switch:
            dataloader = tqdm(train_loader, desc="Train Epoch {}".format(epoch))
            cache_keys_lst = []
            for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
                data_WSI = data_WSI.cuda()
                train_feature = data_WSI.squeeze(0)
                bag_train_feature = model(train_feature)
                cache_keys_lst.append(bag_train_feature.squeeze().tolist())
            cache_keys = torch.tensor(cache_keys_lst).to(device)
        total_loss = 0.0
        all_risk_scores = np.zeros((len(train_loader)))
        all_censorships = np.zeros((len(train_loader)))
        all_event_times = np.zeros((len(train_loader)))
        dataloader = tqdm(train_loader, desc="Train Epoch {}".format(epoch))
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            train_feature = data_WSI.squeeze(0)
            bag_train_feature = model(train_feature)
            if switch:
                similarity = bag_train_feature @ cache_keys.t()
                affinity = (init_beta * (similarity - 1)).exp()
                temp_logits = affinity @ cache_values
                cache_model_logits = init_alpha * temp_logits
            train_zero_shot_logits = plip_zero_shot(image_features=bag_train_feature,
                                                    texts=prompts,
                                                    plip_model_path=args.plip_model_path)
            if switch:
                total_logits = (cache_model_logits + train_zero_shot_logits) / 100
            else:
                total_logits = train_zero_shot_logits / 100
            hazards = torch.sigmoid(total_logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = total_loss / len(dataloader)
        train_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print("Train loss: {:.4f}, Train c_index: {:.4f}".format(train_loss, train_c_index))
        scheduler.step()

        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(val_loader)))
        all_censorships = np.zeros((len(val_loader)))
        all_event_times = np.zeros((len(val_loader)))
        dataloader = tqdm(val_loader, desc="Train Epoch {}".format(epoch + 1))
        with torch.no_grad():
            for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
                val_feature = data_WSI.squeeze(0)
                bag_val_feature = model(val_feature)
                if switch:
                    similarity = bag_val_feature @ cache_keys.t()
                    affinity = (init_beta * (similarity - 1)).exp()
                    temp_logits = affinity @ cache_values
                    cache_model_logits = init_alpha * temp_logits
                val_zero_shot_logits = plip_zero_shot(image_features=bag_val_feature,
                                                        texts=prompts,
                                                        plip_model_path=args.plip_model_path)
                if switch:
                    total_logits = (cache_model_logits + val_zero_shot_logits) / 100
                else:
                    total_logits = val_zero_shot_logits / 100
                hazards = torch.sigmoid(total_logits)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = data_Censorship.item()
                all_event_times[batch_idx] = data_Event
                total_loss += loss.item()
            val_loss = total_loss / len(dataloader)
            val_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
            if val_c_index > best_c_index or val_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                model_dir = args.save_path
                if switch:
                    model_name = f'survival_{args.lr}_adapter_mil_{args.seed}.pt'
                else:
                    model_name = f'survival_{args.lr}_text_mil_{args.seed}.pt'
                model_filepath = os.path.join(model_dir, model_name)
                torch.save(model, model_filepath)
            if val_c_index > best_c_index:
                best_c_index = val_c_index
            if val_loss < best_loss:
                best_loss = val_loss
        print("Valid loss: {:.4f}, Valid c_index: {:.4f}".format(val_loss, val_c_index))
        if val_loss < patient_loss:
            patient_loss = val_loss
        else:
            patient_count += 1
        if patient_count > patient_epochs:
            break

    # alpha and beta for test, which are the best in validation
    beta = 1
    alpha = 1
    model.eval()
    total_loss = 0.0
    total_loss = 0.0
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    with torch.no_grad():
        if switch:
            dataloader = tqdm(train_loader)
            cache_keys_lst = []
            for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
                data_WSI = data_WSI.cuda()
                train_feature = data_WSI.squeeze(0)
                bag_train_feature =model(train_feature)
                cache_keys_lst.append(bag_train_feature.squeeze().tolist())
            cache_keys = torch.tensor(cache_keys_lst).to(device)
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(test_loader):
            data_WSI = data_WSI.cuda()
            data_Label = data_Label.type(torch.LongTensor).cuda()
            data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            test_feature = data_WSI.squeeze(0)
            bag_test_feature = model(test_feature)
            if switch:
                similarity = bag_test_feature @ cache_keys.t()
                affinity = (beta * (similarity - 1)).exp()
                temp_logits = affinity @ cache_values
                cache_model_logits = alpha * temp_logits
            test_zero_shot_logits = plip_zero_shot(image_features=bag_test_feature,
                                                   texts=prompts,
                                                   plip_model_path=args.plip_model_path)
            if switch:
                total_logits = (cache_model_logits + test_zero_shot_logits) / 100
            else:
                total_logits = test_zero_shot_logits / 100
            hazards = torch.sigmoid(total_logits)
            S = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event

        test_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print(f"{args.seed} Test C-Index:", test_c_index)

def survival_ab(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device):
    num_epochs = args.epochs
    patient_epochs = args.patients
    best_c_index = 0.0
    best_loss = float('inf')
    patient_loss = float('inf')
    patient_count = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(train_loader)))
        all_censorships = np.zeros((len(train_loader)))
        all_event_times = np.zeros((len(train_loader)))
        dataloader = tqdm(train_loader, desc="Train Epoch {}".format(epoch))
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            logits = model(data_WSI)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss = total_loss / len(dataloader)
        train_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print("Train loss: {:.4f}, Train c_index: {:.4f}".format(train_loss, train_c_index))
        scheduler.step()

        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(val_loader)))
        all_censorships = np.zeros((len(val_loader)))
        all_event_times = np.zeros((len(val_loader)))
        dataloader = tqdm(val_loader, desc="Train Epoch {}".format(epoch + 1))
        with torch.no_grad():
            for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
                logits = model(data_WSI)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = data_Censorship.item()
                all_event_times[batch_idx] = data_Event
                total_loss += loss.item()
            val_loss = total_loss / len(dataloader)
            val_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
            if val_c_index > best_c_index or val_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                model_dir = args.save_path
                if args.model.lower() == 'ab_mil':
                    model_name = f'survival_{args.lr}_ab_mil_{args.seed}.pt'
                else:
                    model_name = f'survival_{args.lr}_rrt_mil_{args.seed}.pt'
                model_filepath = os.path.join(model_dir, model_name)
                torch.save(model, model_filepath)
            if val_c_index > best_c_index:
                best_c_index = val_c_index
            if val_loss < best_loss:
                best_loss = val_loss
        print("Valid loss: {:.4f}, Valid c_index: {:.4f}".format(val_loss, val_c_index))
        if val_loss < patient_loss:
            patient_loss = val_loss
        else:
            patient_count += 1
        if patient_count > patient_epochs:
            break

    model.eval()
    total_loss = 0.0
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    with torch.no_grad():
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(test_loader):
            data_WSI = data_WSI.cuda()
            data_Label = data_Label.type(torch.LongTensor).cuda()
            data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            logits = model(data_WSI)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
        test_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print(f"{args.seed} Test C-Index:", test_c_index)

def classifier_adapter(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device, switch=True):
    if args.datasets.lower() == 'camelyon16':
        prompts = ["This is an H&E image of normal axillary lymph node tissue.",
                   "This is an H&E image of axillary lymph node tissue containing metastases."]
    elif args.datasets.lower() == 'tcga_brca':
        prompts = ["This is an H&E image of invasive ductal carcinoma tissue.",
                   "This is an H&E image of invasive lobular carcinoma tissue."]
    init_alpha = args.init_alpha
    init_beta = args.init_beta
    num_epochs = args.epochs
    patient_epochs = args.patients
    cache_values = cache_model_prepare(loader=train_loader)
    best_auc_score = 0.0
    best_loss = float('inf')
    patient_loss = float('inf')
    patient_count = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        model.train()
        if switch:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            cache_keys_lst = []
            for feature, label in pbar:
                train_feature, train_label = feature.to(device), label.to(device)
                train_feature = train_feature.squeeze(0)
                bag_train_feature = model(train_feature)
                cache_keys_lst.append(bag_train_feature.squeeze().tolist())
            cache_keys = torch.tensor(cache_keys_lst).to(device)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for feature, label in pbar:
            train_feature, train_label = feature.to(device), label.to(device)
            train_feature = train_feature.squeeze(0)
            bag_train_feature = model(train_feature)
            if switch:
                similarity = bag_train_feature @ cache_keys.t()
                affinity = (init_beta * (similarity - 1)).exp()
                temp_logits = affinity @ cache_values
                cache_model_logits = init_alpha * temp_logits
            train_zero_shot_logits = plip_zero_shot(image_features=bag_train_feature,
                                                    texts=prompts,
                                                    plip_model_path=args.plip_model_path)
            if switch:
                total_logits = cache_model_logits + train_zero_shot_logits
                total_logits = train_zero_shot_logits
            else:
                total_logits = train_zero_shot_logits
            loss = criterion(total_logits, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {loss.item():.4f}")

        model.eval()
        true_labels = []
        predicted_probs = []
        val_loss = 0.0
        with torch.no_grad():
            for feature, label in val_loader:
                feature, label = feature.to(device), label.to(device)
                feature = feature.squeeze(0)
                bag_val_feature = model(feature)
                if switch:
                    similarity = bag_val_feature @ cache_keys.t()
                    affinity = (init_beta * (similarity - 1)).exp()
                    temp_logits = affinity @ cache_values
                    cache_model_logits = init_alpha * temp_logits
                val_zero_shot_logits = plip_zero_shot(image_features=bag_val_feature,
                                                      texts=prompts,
                                                      plip_model_path=args.plip_model_path)
                if switch:
                    total_logits = cache_model_logits + val_zero_shot_logits
                else:
                    total_logits = val_zero_shot_logits
                loss = criterion(total_logits, label)
                val_loss += loss.item()
                true_labels.extend(label.cpu())
                predicted_probs.extend(torch.softmax(total_logits, dim=1)[:, 1].cpu().numpy())
            epoch_val_loss = val_loss / len(val_loader)
            auc_score = roc_auc_score(true_labels, predicted_probs)
            if auc_score > best_auc_score or epoch_val_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                model_dir = args.save_path
                if switch:
                    model_name = f'{args.lr}_adapter_mil_{args.seed}.pt'
                else:
                    model_name = f'{args.lr}_text_mil_{args.seed}.pt'
                model_filepath = os.path.join(model_dir, model_name)
                torch.save(model, model_filepath)
            if auc_score > best_auc_score:
                best_auc_score = auc_score
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}")
        if epoch_val_loss < patient_loss:
            patient_loss = epoch_val_loss
        else:
            patient_count += 1
        if patient_count > patient_epochs:
            break

    # alpha and beta for test, which are the best in validation
    beta = 1
    alpha = 1
    model.eval()
    predictions = []
    true_labels = []
    predicted_probs = []
    with torch.no_grad():
        if switch:
            cache_keys_lst = []
            for feature, label in train_loader:
                train_feature, train_label = feature.to(device), label.to(device)
                train_feature = train_feature.squeeze(0)
                bag_train_feature = model(train_feature)
                cache_keys_lst.append(bag_train_feature.squeeze().tolist())
            cache_keys = torch.tensor(cache_keys_lst).to(device)
        for batch_features, batch_labels in test_loader:
            feature, label = batch_features.to(device), batch_labels.to(device)
            feature = feature.squeeze(0)
            bag_test_feature = model(feature)
            if switch:
                similarity = bag_test_feature @ cache_keys.t()
                affinity = (beta * (similarity - 1)).exp()
                temp_logits = affinity @ cache_values
                cache_model_logits = alpha * temp_logits
            test_zero_shot_logits = plip_zero_shot(image_features=bag_test_feature,
                                                   texts=prompts,
                                                   plip_model_path=args.plip_model_path)
            if switch:
                total_logits = cache_model_logits + test_zero_shot_logits
            else:
                total_logits = test_zero_shot_logits
            pred = total_logits.topk(1, 1, True, True)[1].t().squeeze(0)
            true_labels.extend(label.cpu())
            predictions.extend(pred.detach().cpu())
            predicted_probs.extend(torch.softmax(total_logits, dim=1)[:, 1].cpu().numpy())
        classifier_test(true_labels, predicted_probs, predictions, args.seed)

def classifier_ab(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, device):
    num_epochs = args.epochs
    patient_epochs = args.patients
    best_auc_score = 0.0
    best_loss = float('inf')
    patient_loss = float('inf')
    patient_count = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            batch_features, batch_labels = batch
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {loss.item():.4f}")
        model.eval()
        true_labels = []
        predicted_probs = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                _, preds = torch.max(outputs.data, 1)
                val_loss += loss.item() * batch_features.size(0)
                true_labels.extend(batch_labels.cpu().numpy())
                predicted_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            epoch_val_loss = val_loss / len(val_loader)
            auc_score = roc_auc_score(true_labels, predicted_probs)
            if auc_score > best_auc_score or epoch_val_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                model_dir = args.save_path
                if args.model.lower() == 'ab_mil':
                    model_name = f'{args.lr}_ab_mil_{args.seed}.pt'
                else:
                    model_name = f'{args.lr}_rrt_mil_{args.seed}.pt'
                model_filepath = os.path.join(model_dir, model_name)
                torch.save(model, model_filepath)
            if auc_score > best_auc_score:
                best_auc_score = auc_score
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}")
        if epoch_val_loss < patient_loss:
            patient_loss = epoch_val_loss
        else:
            patient_count += 1
        if patient_count > patient_epochs:
            break

    model.eval()
    predictions = []
    true_labels = []
    predicted_probs = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            _, preds = torch.max(outputs.data, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
            predicted_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        classifier_test(true_labels, predicted_probs, predictions, args.seed)

def training_free_model(args):
    # for camelyon16
    prompts = ["This is an H&E image of normal axillary lymph node tissue.",
               "This is an H&E image of axillary lymph node tissue containing metastases."]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_torch(args.seed)
    train_and_val_path = os.path.join(args.dataset_root, 'train_and_val')
    test_path = os.path.join(args.dataset_root, 'test')
    train_loader, val_loader = camelyon16_train_and_val(train_and_val_path, seed=args.seed,
                                                        fewshot_samples=args.fewshot_samples, val_ratio=args.val_ratio,
                                                        num_workers=args.num_workers)
    test_loader = camelyon16_test(test_path, seed=args.seed)
    with torch.no_grad():
        cache_values = cache_model_prepare(loader=train_loader)
        cache_keys_lst = []
        pbar_1 = tqdm(train_loader)
        for feature, label in pbar_1:
            train_feature, train_label = feature.to(device), label.to(device)
            train_feature = train_feature.squeeze(0)
            bag_train_feature = Mean_pooling(train_feature)
            cache_keys_lst.append(bag_train_feature.squeeze().tolist())
        cache_keys = torch.tensor(cache_keys_lst).to(device)
        best_alpha = 0.5
        best_beta = 2
        predictions = []
        true_labels = []
        predicted_probs = []
        pbar_2 = tqdm(test_loader)
        for batch_features, batch_labels in pbar_2:
            feature, label = batch_features.to(device), batch_labels.to(device)
            feature = feature.squeeze(0)
            bag_test_feature = Mean_pooling(feature)
            similarity = bag_test_feature @ cache_keys.t()
            affinity = (best_beta * (similarity - 1)).exp()
            temp_logits = affinity @ cache_values
            cache_model_logits = best_alpha * temp_logits
            test_zero_shot_logits = plip_zero_shot(image_features=bag_test_feature,
                                                   texts=prompts,
                                                   plip_model_path=args.plip_model_path)
            total_logits = test_zero_shot_logits + cache_model_logits
            pred = total_logits.topk(1, 1, True, True)[1].t().squeeze(0)
            true_labels.extend(label.cpu())
            predictions.extend(pred.detach().cpu())
            predicted_probs.extend(torch.softmax(total_logits, dim=1)[:, 1].cpu().numpy())
        classifier_test(true_labels, predicted_probs, predictions, args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga-brca, tcga-blca]')
    parser.add_argument('--dataset_root', default=r"D:\DATA\PLIP_MIL\RAW_PLIP_features", type=str, help='Dataset root path')
    parser.add_argument('--fewshot_samples', default=-1, type=int, help='Few-shot samples, default all samples')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--val_ratio', default=0.33, type=float, help='Validation ratio')
    parser.add_argument('--test_ratio', default=0.33, type=float, help='Test ratio(not for camelyon16)')

    # Train
    parser.add_argument('--seed', default=2024, type=int, help='random seed [2024]')
    parser.add_argument('--input_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes, 4 for survival analysis')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--patients', default=30, type=int, help='early stopping patients, 20 for TCGA datasets')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--model', default='ab_mil', type=str, help='[ab_mil, rrt_mil, adapter_mil, text_mil]')
    parser.add_argument('--epeg_k', default=15, type=int, help='for RRT_Encoder')
    parser.add_argument('--crmsa_k', default=3, type=int, help='for RRT_Encoder')
    parser.add_argument('--init_alpha', default=1.0, type=float, help='initial alpha for adapter')
    parser.add_argument('--init_beta', default=1.0, type=float, help='initial beta for adapter')
    parser.add_argument('--plip_model_path', default=r"D:\WORK\Projects\PLIP_test\plip_finish", type=str, help='pretrained model path')
    parser.add_argument('--save_path', default=r"D:\DATA\PLIP_MIL\ab_mil_model", type=str, help='model save path')

    args = parser.parse_args()
    print(args)
    main(args)
    # training_free_model(args)