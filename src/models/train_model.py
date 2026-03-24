import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib 

def load_data(path):
    return pd.read_csv(path)

def build_pipeline(num_features, cat_features): 
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model",model),
        ]
    )

    return pipeline

def main():
    df = load_data("data/processed/churn_clean.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(num_features, cat_features)

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/churn_model.pkl")

    print("Modelo entrenado y guardado en models/churn_model.pkl")


if __name__ == "__main__":
    main()    