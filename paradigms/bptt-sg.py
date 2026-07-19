import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import wfdb
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import WeightedRandomSampler
import time
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay, average_precision_score, PrecisionRecallDisplay
import os
import matplotlib.pyplot as plt

### DEVICE ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### DATA ACQUISITION ###
record_names = ['I01', 'I02', 'I03', 'I04', 'I05']
all_beats = []
all_labels = []

for rec_name in record_names:
    record = wfdb.rdrecord(record_name=rec_name, pn_dir='incartdb')
    ann = wfdb.rdann(record_name=rec_name, pn_dir='incartdb', extension='atr')
    fs = record.fs

    for i in range(len(ann.sample)):
        if ann.sample[i] < fs//2 or ann.sample[i] > len(record.p_signal) - fs//2:
            continue
        single_beat = ann.sample[i]
        window = record.p_signal[(single_beat - fs//4):(single_beat + fs//4)]
        label = ann.symbol[i]
        if label not in ('N', 'V'):
            continue
        all_beats.append(window)
        all_labels.append(label)

### PREPROCESSING - EDGES ###
record_ref = wfdb.rdrecord(record_name='I01', pn_dir='incartdb')
data = np.transpose(record_ref.p_signal)
cor = np.corrcoef(data)
np.fill_diagonal(cor, -1)
cor_bool = cor > 0.5
cor_indices = np.argwhere(cor_bool)
cor_indices = np.transpose(cor_indices)
edge_index = torch.tensor(cor_indices, dtype=torch.long).to(device)

### PREPROCESSING - X/Y TENSORS AND SPLITTING ###
X = np.array(all_beats).astype(np.float32) # (N, 128, 12)
mapping = {'N': 0, 'V': 1}
Y = np.array(pd.Series(all_labels).map(mapping)).astype(np.int64)

unique, counts = np.unique(Y, return_counts=True)
rs = 42
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs, stratify=Y)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.125, random_state=rs, stratify=Y_temp)

lead_mean = X_train.mean(axis=(0, 1), keepdims=True) # (1, 1, 12)
lead_std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8 # (1, 1, 12)
X_train = (X_train - lead_mean) / lead_std
X_val   = (X_val   - lead_mean) / lead_std
X_test  = (X_test  - lead_mean) / lead_std

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val   = torch.tensor(X_val,   dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
Y_val   = torch.tensor(Y_val,   dtype=torch.long).to(device)
Y_test  = torch.tensor(Y_test,  dtype=torch.long).to(device)

### DATASET AND DATALOADER INITIALIZATION ###
class ArrhythmiaDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.X = features
        self.Y = labels
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

training_data = ArrhythmiaDataset(X_train, Y_train)
validation_data = ArrhythmiaDataset(X_val, Y_val)
testing_data = ArrhythmiaDataset(X_test,  Y_test)

cls_count = np.bincount(Y_train.cpu().numpy(), minlength=2)
per_class_w = 1.0 / cls_count
sample_w = torch.as_tensor(per_class_w[Y_train.cpu().numpy()], dtype=torch.float)
train_sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
train_data_loader = DataLoader(training_data, batch_size=8, sampler=train_sampler)
val_data_loader = DataLoader(validation_data, batch_size=8, shuffle=False)
test_data_loader = DataLoader(testing_data, batch_size=8, shuffle=False)

### ARCHITECTURE ###
class SpatialGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = GCNConv(in_channels=1, out_channels=64)
        self.gnn2 = GCNConv(in_channels=64, out_channels=64)

    def forward(self, x, edge_index):
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        return x
    
class TemporalSNN(nn.Module):
    def __init__(self, gnn_out=768, hidden=256, beta=0.9):
        super().__init__()
        self.gnn = SpatialGNN()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc_in = nn.Linear(gnn_out, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.fc_h = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x, edge_index):
        # x: (batch, T, leads)
        T = x.shape[1]

        # zero the membrane potentials at the start of every beat
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for t in range(T):
            # spatial GNN on one time-step slice: (batch, leads, 1) -> (batch, leads, 64)
            gnn_out  = self.gnn(x[:, t, :].unsqueeze(2), edge_index)
            gnn_flat = torch.flatten(gnn_out, start_dim=1) # (batch, 768)

            cur1 = self.fc_in(gnn_flat) # synaptic current injection
            spk1, mem1 = self.lif1(cur1, mem1) # fire + update membrane

            cur2 = self.fc_h(spk1) # spikes -> layer 2
            spk2, mem2 = self.lif2(cur2, mem2) # fire + update membrane

        return mem2 # (batch, hidden) final membrane potential


class HybridModel(nn.Module):
    def __init__(self, hidden=256, beta=0.9):
        super().__init__()
        self.snn    = TemporalSNN(gnn_out=768, hidden=hidden, beta=beta)
        self.linear = nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        mem_final = self.snn(x, edge_index) # (batch, hidden)
        return self.linear(mem_final) # (batch, 2)

model = HybridModel().to(device)

### TRAINING LOOP ###
criterion = nn.CrossEntropyLoss()
lr = 1e-3
optimizer = Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
start = time.time()
epochs = 30

best_val_ap = -1.0
patience = 8
patience_counter = 0
os.makedirs('drive/gnn_snn_vahy', exist_ok=True)

for epoch in range(epochs):
    total_loss_train = 0
    total_acc_train = 0
    model.train()
    for index, (inputs, labels) in enumerate(train_data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        prediction = model(inputs, edge_index)
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()

        prediction_labels = torch.max(prediction, dim=1).indices
        total_acc_train += (prediction_labels == labels).sum().item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    train_loss = total_loss_train / len(train_data_loader)
    train_acc = total_acc_train / len(training_data) * 100

    # validation loop
    model.eval()
    total_loss_val = 0
    val_preds, val_trues, val_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in val_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            prediction = model(inputs, edge_index)
            total_loss_val += criterion(prediction, labels).item()

            prob_v = torch.softmax(prediction, dim=1)[:, 1]
            prediction_labels = torch.max(prediction, dim=1).indices
            val_preds.extend(prediction_labels.cpu().tolist())
            val_trues.extend(labels.cpu().tolist())
            val_probs.extend(prob_v.cpu().tolist())

    val_loss = total_loss_val / len(val_data_loader)
    val_f1 = f1_score(val_trues, val_preds, zero_division=0) # info only
    val_ap = average_precision_score(val_trues, val_probs) # PR-AUC
    try:
        val_auc = roc_auc_score(val_trues, val_probs)
    except ValueError:
        val_auc = float('nan')

    # step the scheduler on the ranking metric
    scheduler.step(val_ap)
    cur_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch {epoch+1:2d} | Train Loss {train_loss:.4f} Acc {train_acc:.2f} '
          f'| Val Loss {val_loss:.4f} PR-AUC {val_ap:.4f} ROC-AUC {val_auc:.4f} '
          f'F1@0.5 {val_f1:.4f} | lr {cur_lr:.1e}')

    # early stopping / best-model on PR-AUC
    if val_ap > best_val_ap:
        best_val_ap = val_ap
        patience_counter = 0
        torch.save(model.state_dict(), 'drive/gnn_snn_vahy/SG_best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

end = time.time()
print(f"Elapsed time: {end-start:.1f} seconds | best val PR-AUC: {best_val_ap:.4f}")
model.load_state_dict(torch.load('drive/MyDrive/gnn_snn_vahy/SG_best_model.pth'))
model.eval()

### THRESHOLD SELECTION ###
val_probs, val_true = [], []
with torch.no_grad():
    for inputs, labels in val_data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        p = torch.softmax(model(inputs, edge_index), dim=1)[:, 1]
        val_probs.extend(p.cpu().tolist())
        val_true.extend(labels.cpu().tolist())
val_probs = np.array(val_probs)
val_true  = np.array(val_true)

def _score(y, preds):
    return matthews_corrcoef(y, preds) if len(np.unique(preds)) > 1 else -1.0

thresholds = np.linspace(0.05, 0.95, 181)
best_threshold, best_obj = 0.5, -2.0
for t in thresholds:
    preds = (val_probs >= t).astype(int)
    s = _score(val_true, preds)
    if s > best_obj:
        best_obj, best_threshold = s, t

_p = (val_probs >= best_threshold).astype(int)

### TESTING LOOP ###
true_labels = []
pred_labels = []
pred_probs = []

model.eval()
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for index, (inputs, labels) in enumerate(test_data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        true_labels.extend(labels.tolist())
        prediction = model(inputs, edge_index)
        batch_loss = criterion(prediction, labels)
        total_loss_test += batch_loss.item()

        probs = torch.softmax(prediction, dim=1)[:, 1] 
        prediction_labels = (probs >= best_threshold).long() 
        pred_labels.extend(prediction_labels.tolist())
        pred_probs.extend(probs.tolist())
        acc = (prediction_labels == labels).sum().item()
        total_acc_test += acc

        if index == 0:
            print(labels)
            print(prediction_labels)

print(f'Test Loss: {round(total_loss_test/len(test_data_loader), 4)} Test Accuracy: {round(total_acc_test/testing_data.__len__() * 100, 4)}')

### METRICS ###
cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
print(cm)
print(classification_report(y_true=true_labels, y_pred=pred_labels, target_names=['N', 'V']))

# ROC-AUC
roc_auc = roc_auc_score(true_labels, pred_probs)
print(f'ROC-AUC: {round(roc_auc, 4)}')
RocCurveDisplay.from_predictions(true_labels, pred_probs, name='GNN-SNN')
plt.title('ROC Curve')
plt.show()

# PR-AUC
pr_auc = average_precision_score(true_labels, pred_probs)
print(f'PR-AUC: {round(pr_auc, 4)}')
PrecisionRecallDisplay.from_predictions(true_labels, pred_probs, name='GNN-SNN')
plt.title('PR Curve')
plt.show()

# specificity (TNR)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)  # same as V recall, shown here for pairing
print(f'Sensitivity (V recall): {round(sensitivity, 4)}')
print(f'Specificity (N recall): {round(specificity, 4)}')

# expected calibration error (ECE)
n_bins = 10
bin_boundaries = np.linspace(0, 1, n_bins + 1)
ece = 0.0
pred_probs_np = np.array(pred_probs)
true_labels_np = np.array(true_labels)
for i in range(n_bins):
    mask = (pred_probs_np >= bin_boundaries[i]) & (pred_probs_np < bin_boundaries[i + 1])
    if mask.sum() == 0:
        continue
    bin_confidence = pred_probs_np[mask].mean()
    bin_accuracy = true_labels_np[mask].mean()
    ece += mask.sum() * abs(bin_confidence - bin_accuracy)
ece /= len(pred_probs_np)
print(f'ECE: {round(ece, 4)}')