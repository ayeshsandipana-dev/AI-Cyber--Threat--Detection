import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dataset paths
normal_file = "data/Monday-WorkingHours.pcap_ISCX.csv"
attack_file = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Loading datasets...")

# Load a manageable sample for 8GB RAM
normal = pd.read_csv(normal_file, nrows=30000)
attack = pd.read_csv(attack_file, nrows=100000)

# Clean column names
normal.columns = normal.columns.str.strip()
attack.columns = attack.columns.str.strip()

# Keep the required classes
normal = normal[normal["Label"].str.strip() == "BENIGN"]
attack = attack[attack["Label"].str.strip() == "DDoS"]

# Sample both classes
normal = normal.sample(
    n=min(10000, len(normal)),
    random_state=42
)

attack = attack.sample(
    n=min(10000, len(attack)),
    random_state=42
)

# Combine datasets
df = pd.concat([normal, attack], ignore_index=True)

print("Dataset shape:", df.shape)
print(df["Label"].value_counts())

if df["Label"].nunique() != 2:
    raise ValueError(
        "Both BENIGN and DDoS samples are required. "
        "Increase nrows or check the CSV files."
    )

# Prepare features and labels
X = df.drop(columns=["Label"])
y = df["Label"].str.strip()

# Keep numerical features
X = X.select_dtypes(include=[np.number])

# Clean invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training AI model...")

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=2
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nAI MODEL RESULTS")
print("-----------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model and feature names
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {"model": model, "features": X.columns.tolist()},
    "models/threat_detection_model.pkl"
)

print("\nModel saved successfully!")