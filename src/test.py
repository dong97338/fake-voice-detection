# test.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
from tqdm.auto import tqdm
from common import ASVSpoofDataset, Model, get_labels, EER
import torchaudio.transforms as transforms

# Paths
test_audio_files_path = 'LA/LA/ASVspoof2019_LA_eval/flac'
test_labels_path = 'LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

# Load labels
test_filename2label = get_labels(test_labels_path)

# Data loader
mel_spectrogram = transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
num_samples = 6 * 16000
test_dataset = ASVSpoofDataset(test_audio_files_path, num_samples, test_filename2label, mel_spectrogram)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1024)

# Load the saved model
device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
model.load_state_dict(torch.load('resnet200d.pt'))
model.eval()

# Criterion
criterion = torch.nn.BCELoss()

# Testing
new_outputs, new_labels, test_loss = [], [], 0.0
with torch.no_grad():
    for _, (test_images, test_labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
        test_images, test_labels = test_images.to(device), test_labels.to(device).reshape(-1, 1).type(torch.cuda.FloatTensor)
        test_outputs = model(test_images)
        new_outputs.append(test_outputs.cpu().numpy())
        new_labels.append(test_labels.cpu().numpy())
        test_loss += criterion(test_outputs, test_labels).item()

labels = np.concatenate(new_labels)
outputs = np.concatenate(new_outputs)

# Compute metrics
roc_auc = roc_auc_score(labels, outputs)
eer = EER(labels, outputs)
report = classification_report(labels, (outputs > 0.5).astype(int))
ConfusionMatrixDisplay.from_predictions(labels, (outputs > 0.5).astype(int))

print(f'ROC AUC Score: {roc_auc}')
print(report)
print(f'EER: {eer}')
