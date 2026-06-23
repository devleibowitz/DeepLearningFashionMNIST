import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import lightgbm as lgb

# Configuration
WINDOW_SIZE = 800  # 200 seconds at 4Hz
STRIDE = 400       # 50% overlap
BATCH_SIZE = 8
LSTM_HIDDEN = 64
EPOCHS = 30

# ------------------
# 1. Data Preparation
# ------------------
def preprocess_ctg(ctg_tensor):
    """Process CTG tensor into normalized windows"""
    # Separate channels [32, 2, 21620] -> [32, 21620] for each channel
    fhr = ctg_tensor[:, 0, :]  # Shape (32, 21620)
    uc = ctg_tensor[:, 1, :]   # Shape (32, 21620)
    
    # Create overlapping windows
    def window_tensor(t, window_size, stride):
        return t.unfold(1, window_size, stride).permute(0,2,1)  # (patients, windows, window_size)
    
    fhr_windows = window_tensor(fhr, WINDOW_SIZE, STRIDE)  # (32, 54, 800)
    uc_windows = window_tensor(uc, WINDOW_SIZE, STRIDE)    # (32, 54, 800)
    
    # Normalize per-patient
    fhr_normalized = (fhr_windows - fhr_windows.mean(dim=2, keepdim=True)) / (fhr_windows.std(dim=2, keepdim=True) + 1e-7)
    uc_normalized = (uc_windows - uc_windows.mean(dim=2, keepdim=True)) / (uc_windows.std(dim=2, keepdim=True) + 1e-7)
    
    return fhr_normalized, uc_normalized

def preprocess_ehr(ehr_df):
    """Process EHR dataframe"""
    # Separate features and labels
    X_ehr = ehr_df.drop('label', axis=1)
    y = ehr_df['label'].values
    
    # Preprocessing pipeline
    numeric_features = X_ehr.select_dtypes(include=np.number).columns
    categorical_features = X_ehr.select_dtypes(exclude=np.number).columns
    
    preprocessor = ColumnTransformer([
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    X_processed = preprocessor.fit_transform(X_ehr)
    return X_processed, y, preprocessor

# ------------------
# 2. Model Architecture
# ------------------
class CTGLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fhr_lstm = nn.LSTM(1, LSTM_HIDDEN, batch_first=True)
        self.uc_lstm = nn.LSTM(1, LSTM_HIDDEN, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2*LSTM_HIDDEN, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, fhr, uc):
        # Process FHR channel
        fhr_out, _ = self.fhr_lstm(fhr.unsqueeze(-1))
        fhr_last = fhr_out[:, -1, :]
        
        # Process UC channel
        uc_out, _ = self.uc_lstm(uc.unsqueeze(-1))
        uc_last = uc_out[:, -1, :]
        
        # Concatenate and classify
        combined = torch.cat([fhr_last, uc_last], dim=1)
        return self.classifier(combined)

# ------------------
# 3. Training Pipeline
# ------------------
# Prepare data
ctg_tensor = torch.randn(32, 2, 21620)  # Your CTG data
ehr_df = pd.DataFrame(np.random.randn(32, 10))  # Your EHR data
ehr_df['label'] = np.random.randint(0, 2, 32)

# Preprocess
fhr_windows, uc_windows = preprocess_ctg(ctg_tensor)
X_ehr, y_ehr, preprocessor = preprocess_ehr(ehr_df)

# Convert to window-level dataset
X_fhr = fhr_windows.reshape(-1, WINDOW_SIZE)  # (32*54, 800)
X_uc = uc_windows.reshape(-1, WINDOW_SIZE)    # (32*54, 800)
y_ctg = np.repeat(y_ehr, fhr_windows.shape[1])  # Create window-level labels

# Split indices
patient_ids = np.arange(32)
train_idx, test_idx = patient_ids[:25], patient_ids[25:]

# ------------------
# 4. LSTM Training
# ------------------
lstm_model = CTGLSTMModel()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Convert to tensors
train_mask = np.isin(np.arange(32), train_idx).repeat(fhr_windows.shape[1])
X_fhr_train = torch.FloatTensor(X_fhr[train_mask])
X_uc_train = torch.FloatTensor(X_uc[train_mask])
y_train = torch.FloatTensor(y_ctg[train_mask])

# Training loop
for epoch in range(EPOCHS):
    lstm_model.train()
    optimizer.zero_grad()
    
    outputs = lstm_model(X_fhr_train, X_uc_train).squeeze()
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

# ------------------
# 5. LightGBM Training
# ------------------
lgb_train = lgb.Dataset(X_ehr[train_idx], label=y_ehr[train_idx])
lgb_model = lgb.train({
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31
}, lgb_train, num_boost_round=100)

# ------------------
# 6. Fusion & Evaluation
# ------------------
with torch.no_grad():
    # Get CTG predictions
    ctg_probs = lstm_model(
        torch.FloatTensor(X_fhr),
        torch.FloatTensor(X_uc)
    ).squeeze().numpy()
    
    # Aggregate per-patient (max pooling)
    patient_probs = []
    for pid in range(32):
        mask = np.repeat(pid == np.arange(32), fhr_windows.shape[1])[train_mask|test_mask]
        patient_probs.append(ctg_probs[mask].max())
    ctg_probs = np.array(patient_probs)
    
    # Get EHR predictions
    ehr_probs = lgb_model.predict(X_ehr)
    
    # Combine predictions
    fusion_input = np.column_stack([ctg_probs, ehr_probs])
    
    # Train final classifier
    final_model = LogisticRegression().fit(fusion_input[train_idx], y_ehr[train_idx])
    
    # Evaluate
    test_probs = final_model.predict_proba(fusion_input[test_idx])[:,1]
    predictions = (test_probs > 0.5).astype(int)
    
    print(f"\nFinal Metrics:")
    print(f"AUC: {roc_auc_score(y_ehr[test_idx], test_probs):.3f}")
    print(f"Accuracy: {accuracy_score(y_ehr[test_idx], predictions):.3f}")
    print(f"F1 Score: {f1_score(y_ehr[test_idx], predictions):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_ehr[test_idx], predictions))
