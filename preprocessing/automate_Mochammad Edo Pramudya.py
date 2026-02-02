import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


# Load Dataset
def load_data(path):
    df = pd.read_csv(path)
    return df


# Handle Missing Values + Drop Kolom
def handle_missing_and_drop(df):
    df = df.copy()

    # Missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Drop kolom tidak relevan
    df = df.drop(columns=["Cabin"])

    return df


# Cek & Hapus Duplikat
def handle_duplicates(df):
    df = df.drop_duplicates()
    return df


# Encoding Data Kategorikal
def encode_categorical(df):
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked"],
        drop_first=True
    )
    return df


# Binning Age
def binning_age(df):
    df = df.copy()

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    )

    df = pd.get_dummies(df, columns=["AgeGroup"], drop_first=True)
    return df


# Pisahkan Fitur & Target
def split_features_target(df):
    target = "Survived"

    X = df.drop(columns=[target, "Name", "Ticket"])
    y = df[target]

    return X, y


# Standarisasi Fitur
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Deteksi & Penanganan Outlier (IQR)
def handle_outliers(X_scaled, y):
    X_df = pd.DataFrame(X_scaled)

    Q1 = X_df.quantile(0.25)
    Q3 = X_df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~(
        (X_df < (Q1 - 3.0 * IQR)) |
        (X_df > (Q3 + 3.0 * IQR))
    ).any(axis=1)

    X_clean = X_scaled[mask]
    y_clean = y.iloc[mask.values]

    return X_clean, y_clean


# Simpan Output
def save_output(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y.values)

    print("✅ Data preprocessing berhasil disimpan di:", output_dir)


# Pipeline Utama
def run_preprocessing(
    data_path="Titanic-Dataset.csv",
    output_dir="preprocessing"
):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(BASE_DIR, data_path)
    output_dir = os.path.join(BASE_DIR, output_dir)

    df = pd.read_csv(data_path)
    df = handle_missing_and_drop(df)
    df = handle_duplicates(df)
    df = encode_categorical(df)
    df = binning_age(df)

    X, y = split_features_target(df)
    X_scaled = scale_features(X)
    X_final, y_final = handle_outliers(X_scaled, y)

    save_output(X_final, y_final, output_dir)



if __name__ == "__main__":
    run_preprocessing()
