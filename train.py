import numpy as np
import torch
from torch import nn
import torchaudio
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math
import timm
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, f1_score, classification_report, ConfusionMatrixDisplay

train_audio_files_path = 'LA/LA/ASVspoof2019_LA_train/flac'
train_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
len(os.listdir(train_audio_files_path))

def readtxtfile(path):
    with open(path, 'r') as file:
        text = file.read().splitlines()
        return text
    
def getlabels(path):
    text = readtxtfile(path)
    filename2label = {}
    for item in tqdm(text):
        key = item.split(' ')[1]
        value = item.split(' ')[-1]
        filename2label[key] = value
        
    return filename2label

filename2label = getlabels(train_labels_path)

val_audio_files_path = 'LA/LA/ASVspoof2019_LA_dev/flac'
val_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
val_filename2label = getlabels(val_labels_path)

test_audio_files_path = 'LA/LA/ASVspoof2019_LA_eval/flac'
test_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
test_filename2label = getlabels(test_labels_path)

l = list(filename2label.values())
print(l.count('bonafide'), l.count('spoof'))
plt.hist(l)

class ASVSpoof(torch.utils.data.Dataset):
    def __init__(self, audio_dir_path, num_samples, filename2label, transforms):
        super().__init__()
        self.audio_dir_path = audio_dir_path
        self.num_samples = num_samples
        self.audio_file_names = self.get_audio_file_names(filename2label)
        self.labels, self.label2id, self.id2label = self.get_labels(filename2label)
        self.transforms = transforms
        
    def __getitem__(self, index):
        signal, sr = torchaudio.load(os.path.join(self.audio_dir_path, self.audio_file_names[index]))
#         print(signal.shape)
        signal = self.mix_down_if_necessary(signal)
        signal = self.cut_if_necessary(signal)
#         print(signal.shape)
        signal = self.right_pad_if_necessary(signal)
#         print(signal.shape)
        signal = self.transforms(signal)
#         print(signal.shape)
        label = (self.labels[index])
        return signal, label
    
    def __len__(self):
        return len(self.labels)
    
    def get_audio_file_names(self, filename2label):
        audio_file_names = list(filename2label.keys())
        audio_file_names = [name + '.flac' for name in audio_file_names] # adding extension
        return audio_file_names
    
    def get_labels(self, filename2label):
        labels = list(filename2label.values())
        id2label = {idx : label for idx, label in  enumerate(list(set(labels)))}
        label2id = {label : idx for idx, label in  enumerate(list(set(labels)))}
        labels = [label2id[label] for label in labels]
        return labels, label2id, id2label
    
    def mix_down_if_necessary(self, signal): #converting from stereo to mono
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim = 0, keepdims = True)
        return signal
    
    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :num_samples]
        return signal
    
    def right_pad_if_necessary(self, signal):
        length = signal.shape[1]
        if self.num_samples > length:
            pad_last_dim = (0, num_samples - length)
            signal = torch.nn.functional.pad(signal, pad_last_dim)
        return signal
    
mel_spectogram = torchaudio.transforms.MelSpectrogram(
    sample_rate = 16000,
    n_fft = 1024,
    hop_length = 512,
    n_mels = 64
)
num_samples = 6 * 16000 # IMPORTANT!!
train_dataset = ASVSpoof(train_audio_files_path, num_samples, filename2label, mel_spectogram)
val_dataset = ASVSpoof(val_audio_files_path, num_samples, val_filename2label, mel_spectogram)
test_dataset = ASVSpoof(test_audio_files_path, num_samples, test_filename2label, mel_spectogram)

# train_dataset[0][0].shape

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet200d.ra2_in1k', pretrained = True, in_chans = 1)
        for i,(name, param) in enumerate(list(self.model.named_parameters())\
                                             [0:39]):
            param.requires_grad = False
            
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        x = self.features(inputs)
        x = self.custom_layers(x)
        return x
    
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_epochs = 12
criterion = nn.BCELoss()
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = 32)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle = True, batch_size = 1024)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = True, batch_size = 1024)
t_steps = len(train_loader)
v_steps = len(val_loader)
ts_steps = len(test_loader)


import gc
x = 100
while(x != 0):
    x = gc.collect()
    torch.cuda.empty_cache()
x


def EER(labels, outputs):
    fpr, tpr, threshold = roc_curve(labels, outputs, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_threshold
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

# Training loop
num_epochs = 15
train_losses = []
val_losses = []
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    y_true = []
    y_pred = []
    train_loss = 0.0
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (images, labels) in loop:
        loop.set_description(f'Epoch {epoch + 1} / {num_epochs}')
#         forward pass
        model.train()
        torch.cuda.empty_cache()
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(device).reshape(-1, 1)
        labels = labels.type(torch.cuda.FloatTensor)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(outputs.detach().cpu().numpy())
        
        loss = criterion(outputs, labels)
        train_loss += loss.item()
#         backward pass
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(Training_loss = loss.item())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    train_eer = EER(y_true, y_pred)
        
#   validation every epoch
    y_true = []
    y_pred = []
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_loop = tqdm(enumerate(val_loader), total = len(val_loader))
        for val_batch_idx, (val_images, val_labels) in val_loop:
            torch.cuda.empty_cache()
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_labels = val_labels.to(device).reshape(-1, 1)
            val_labels = val_labels.type(torch.cuda.FloatTensor) #use torch.FloatTensor if on cpu
        
        
            val_outputs = model(val_images)
            y_true.append(val_labels.detach().cpu().numpy())
            y_pred.append(val_outputs.detach().cpu().numpy())
            curr_val_loss = criterion(val_outputs, val_labels)
            val_loss += curr_val_loss.item()
            val_loop.set_postfix(validation_loss = curr_val_loss.item())
            
    train_loss_after_epoch = train_loss / t_steps
    val_loss_after_epoch = val_loss / v_steps
    train_losses.append(train_loss_after_epoch)
    val_losses.append(val_loss_after_epoch)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    val_eer = EER(y_true, y_pred)
    print(f'Epoch : {epoch + 1} Training loss : {train_loss_after_epoch} Train EER : {train_eer} Validation loss : {val_loss_after_epoch}  Val EER : {val_eer}')


plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'resnet200d.pt')

new_outputs = []
new_labels = []
model.eval()
test_loss = 0.0
with torch.no_grad():
    test_loop = tqdm(enumerate(test_loader), total = len(test_loader))
    for test_batch_idx, (test_images, test_labels) in test_loop:
        torch.cuda.empty_cache()
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_labels = test_labels.to(device).reshape(-1, 1)
        test_labels = test_labels.type(torch.cuda.FloatTensor) #use torch.FloatTensor if on cpu


        test_outputs = model(test_images)
        new_outputs.append(test_outputs.cpu().numpy())
        new_labels.append(test_labels.cpu().numpy())
        curr_test_loss = criterion(test_outputs, test_labels)
        test_loss += curr_test_loss.item()
        test_loop.set_postfix(test_loss = curr_test_loss.item())

labels = np.concatenate(new_labels)
outputs = np.concatenate(new_outputs)
print(labels.shape, outputs.shape)

score = roc_auc_score(labels, outputs)
score

RocCurveDisplay.from_predictions(labels, outputs)

def convert_into_whole(outputs):
    new_output = []
    for o in outputs:
        if o > 0.5:
            new_output.append(1)
        else:
            new_output.append(0)
    return new_output

new_outputs = convert_into_whole(outputs)
new_outputs = np.array(new_outputs)

print(classification_report(labels, new_outputs))

ConfusionMatrixDisplay.from_predictions(labels, new_outputs)

def EER(labels, outputs):
    fpr, tpr, threshold = roc_curve(labels, outputs, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_threshold
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

EER(labels, new_outputs)