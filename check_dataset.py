import pandas as pd

file = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file, nrows=1000)

df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nTraffic Labels:")
print(df["Label"].value_counts())