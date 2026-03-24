import pandas as pd

df = pd.read_csv("data/processed/churn_clean.csv")

print("Tipos de datos:\n")
print(df.dtypes)

print("\nValores únicos en churn")
print(df["Churn"].unique())

print("\nFilas con Nan:")
print(df.isna().sum())