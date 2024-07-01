import os
import librosa
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    SR = 32000
    N_MFCC = 13
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 5
    LR = 3e-4
    SEED = 42

CONFIG = Config()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

df = pd.read_csv('./train.csv')
train, val = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

def get_mfcc_feature(df, train_mode=True):
    features, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC).T, axis=0)
        features.append(mfcc)
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
    return (features, labels) if train_mode else features

train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        return (self.mfcc[index], self.label[index]) if self.label is not None else self.mfcc[index]

train_loader = DataLoader(CustomDataset(train_mfcc, train_labels), batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CustomDataset(val_mfcc, val_labels), batch_size=CONFIG.BATCH_SIZE, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    best_val_score, best_model = 0, None

    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader):
            features, labels = features.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        print(f'Epoch [{epoch}], Train Loss: [{np.mean(train_loss):.5f}], Val Loss: [{_val_loss:.5f}], Val AUC: [{_val_score:.5f}]')

        if best_val_score < _val_score:
            best_val_score, best_model = _val_score, model

    return best_model

def multiLabel_AUC(y_true, y_scores):
    return np.mean([roc_auc_score(y_true[:, i], y_scores[:, i]) for i in range(y_true.shape[1])])

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for features, labels in tqdm(val_loader):
            features, labels = features.float().to(device), labels.float().to(device)
            probs = model(features)
            val_loss.append(criterion(probs, labels).item())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels, all_probs = np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0)
    return np.mean(val_loss), multiLabel_AUC(all_labels, all_probs)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
infer_model = train(model, optimizer, train_loader, val_loader, device)

test = pd.read_csv('./test.csv')
test_mfcc = get_mfcc_feature(test, False)
test_loader = DataLoader(CustomDataset(test_mfcc, None), batch_size=CONFIG.BATCH_SIZE, shuffle=False)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(test_loader):
            features = features.float().to(device)
            predictions += model(features).cpu().numpy().tolist()
    return predictions

preds = inference(infer_model, test_loader, device)
submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.to_csv('./baseline_submit.csv', index=False)
