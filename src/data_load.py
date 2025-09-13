import pandas as pd
from pathlib import Path

def load_dataset(csv_path: str = "data/crocodile_dataset.csv"):
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"{csv_path} não encontrado.")
        df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_dataset()
    print("Shape:", df.shape)
    print(df.head().to_string())
