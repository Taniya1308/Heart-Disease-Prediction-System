import pandas as pd

df = pd.read_csv("dataset/heart.csv")

print("Columns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

print("\nTarget values:")
print(df["target"].value_counts())

print("\nFirst 5 rows:")
print(df.head())