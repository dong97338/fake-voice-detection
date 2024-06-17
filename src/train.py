import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from common import ASVSpoofDataset, Model, get_labels, EER
import wandb

wandb.init(project="fake-voice-detection")

train_audio_files_path = 'LA/LA/ASVspoof2019_LA_train/flac'
train_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
val_audio_files_path = 'LA/LA/ASVspoof2019_LA_dev/flac'
val_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

filename2label = get_labels(train_labels_path)
val_filename2label = get_labels(val_labels_path)

mel_spectrogram = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
num_samples = 6 * 16000
train_loader = DataLoader(ASVSpoofDataset(train_audio_files_path, num_samples, filename2label, mel_spectrogram), shuffle=True, batch_size=64)
val_loader = DataLoader(ASVSpoofDataset(val_audio_files_path, num_samples, val_filename2label, mel_spectrogram), shuffle=True, batch_size=1024)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

wandb.config = {"learning_rate": 0.001, "epochs": 15, "batch_size": 64}

num_epochs = 15
train_losses, val_losses = [], []
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    y_true, y_pred, train_loss = [], [], 0.0
    loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
    for images, labels in loop:
        model.train()
        images, labels = images.to(device), labels.to(device).reshape(-1, 1).type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        outputs = model(images)
        y_true.append(labels.cpu().numpy())
        y_pred.append(outputs.detach().cpu().numpy())  # Use detach() here
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        loop.set_postfix(Training_loss=loss.item())
    train_eer = EER(np.concatenate(y_true), np.concatenate(y_pred))
    
    y_true, y_pred, val_loss = [], [], 0.0
    model.eval()
    with torch.no_grad():
        for val_images, val_labels in tqdm(val_loader, desc='Validation'):
            val_images, val_labels = val_images.to(device), val_labels.to(device).reshape(-1, 1).type(torch.cuda.FloatTensor)
            val_outputs = model(val_images)
            y_true.append(val_labels.cpu().numpy())
            y_pred.append(val_outputs.detach().cpu().numpy())  # Use detach() here
            val_loss += criterion(val_outputs, val_labels).item()
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_eer = EER(np.concatenate(y_true), np.concatenate(y_pred))
    
    wandb.log({"epoch": epoch + 1, "train_loss": train_losses[-1], "val_loss": val_losses[-1], "train_eer": train_eer, "val_eer": val_eer})
    print(f'Epoch: {epoch+1}, Training loss: {train_losses[-1]}, Train EER: {train_eer}, Validation loss: {val_losses[-1]}, Val EER: {val_eer}')

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
torch.save(model.state_dict(), 'resnet200d.pt')
