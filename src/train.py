import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio.transforms as transforms
from tqdm.auto import tqdm
from common import ASVSpoofDataset, ResNetModel, ASTModel, get_labels, EER
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for ASV Spoof Detection")
    parser.add_argument('--model', choices=['resnet', 'ast'], default='resnet', help='Model type to use (resnet or ast)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    return parser.parse_args()

args = parse_args()
wandb.init(project="fake-voice-detection")

train_audio_files_path = 'LA/LA/ASVspoof2019_LA_train/flac'
train_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
val_audio_files_path = 'LA/LA/ASVspoof2019_LA_dev/flac'
val_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

filename2label = get_labels(train_labels_path)
val_filename2label = get_labels(val_labels_path)
num_samples = 6 * 16000
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
bit_length = torch.cuda.get_device_properties(args.gpu).total_memory.bit_length()

if args.model == 'resnet':
    transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
    model = ResNetModel().to(device)
    batch_size = 1 << bit_length - 27
else:
    transform = None
    model = ASTModel().to(device)
    batch_size = 1 << bit_length - 31

print(f'Using batch size: {batch_size}')

train_loader = DataLoader(ASVSpoofDataset(train_audio_files_path, num_samples, filename2label, transform), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(ASVSpoofDataset(val_audio_files_path, num_samples, val_filename2label, transform), shuffle=True, batch_size=1024)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

wandb.config.update({"learning_rate": 0.001, "epochs": 15, "batch_size": batch_size})

for epoch in range(wandb.config.epochs):
    model.train()
    train_loss, y_true, y_pred = 0.0, [], []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for _, (images, labels) in loop:
        images, labels = images.to(device), labels.to(device).reshape(-1, 1).float().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        y_true.append(labels.cpu().numpy())
        y_pred.append(outputs.detach().cpu().numpy())
        loop.set_postfix(Training_loss=loss.item())

    train_eer = EER(np.concatenate(y_true), np.concatenate(y_pred))

    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []
    val_loop = tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():
        for val_images, val_labels in val_loop:
            val_images, val_labels = val_images.to(device), val_labels.to(device).reshape(-1, 1).float().to(device)
            val_outputs = model(val_images)
            y_true.append(val_labels.cpu().numpy())
            y_pred.append(val_outputs.cpu().numpy())
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()
            val_loop.set_postfix(validation_loss=loss.item())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_eer = EER(np.concatenate(y_true), np.concatenate(y_pred))
    
    # Log metrics to wandb
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_eer": train_eer, "val_loss": val_loss, "val_eer": val_eer})

    print(f'Epoch {epoch + 1}: Train Loss: {train_loss}, Train EER: {train_eer}, Val Loss: {val_loss}, Val EER: {val_eer}')

torch.save(model.state_dict(), f'model_{args.model}.pt')
