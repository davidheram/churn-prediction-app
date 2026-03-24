import pandas as pd 

def load_data(path):
    df = pd.read_csv(path)
    
    return df

def clean_data(df):
    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")    
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    input_path = "data/raw/churn.csv"
    output_path = "data/processed/churn_clean.csv"

    df = load_data(input_path)
    df = clean_data(df)

    save_data(df, output_path)

    print("dataset limpio guardando en:", output_path)

