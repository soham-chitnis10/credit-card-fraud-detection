from utils import load_data, preprocess_data, normalize_data, seed_everything
from sklearn.metrics import accuracy_score, f1_score

seed_everything(42)

df_train = load_data('data/fraudTrain.csv')
df_test = load_data('data/fraudTest.csv')

df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

X_train_scaled = normalize_data(df_train.drop(columns=['is_fraud']))
X_test_scaled = normalize_data(df_test.drop(columns=['is_fraud']))


from torch.utils.data import DataLoader, TensorDataset
import torch
train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(df_train['is_fraud'].values, dtype=torch.float32))
test_dataset = TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32),
    torch.tensor(df_test['is_fraud'].values, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

import torch.nn as nn
import torch.optim as optim
from model import CreditCardFraudDetector
loss = nn.CrossEntropyLoss()
model = CreditCardFraudDetector(input_size=X_train_scaled.shape[1], hidden_size=256)

devices = ['cpu']
device = torch.device(devices[0])
print(f"Using device: {device}")
model.to(device)
loss.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import trange, tqdm
for epoch in range(10):
    model.train()
    for batch in tqdm(train_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs, targets.long())
        loss_value.backward()
        optimizer.step()

    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (preds == targets.long()).sum().item()
            all_targets.append(targets)
            all_preds.append(preds)
    targets = torch.cat(all_targets)
    predicted = torch.cat(all_preds)

    print(f'Epoch {epoch+1}, Accuracy: {100 * correct / total:.2f}%')
    print(f'Epoch {epoch+1}, F1 Score: {f1_score(targets.detach().cpu().numpy(), predicted.detach().cpu().numpy(), average="binary"):.2f}')
