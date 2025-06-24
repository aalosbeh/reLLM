import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path: str):
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    return df


def preprocess(df: pd.DataFrame):
    """Basic text cleaning and label encoding."""
    df = df.copy()
    df['text'] = df['text'].astype(str).str.strip()
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    return df, le


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label_id'])
