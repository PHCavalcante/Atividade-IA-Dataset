import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from pathlib import Path

def preprocess(df: pd.DataFrame, target_col: str = "Conservation Status", test_size: float = 0.25, random_state: int = 42):
    df = df.copy()
    cols_to_drop = [target_col, "Scientific Name", "Common Name", "Genus", "Family"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ], remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor, numeric_cols, categorical_cols

if __name__ == "__main__":
    from data_load import load_dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, ncols, ccols = preprocess(df)
    print("X_train shape:", X_train.shape)
