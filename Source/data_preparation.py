import pandas as pd
import seaborn as sns
from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed"


def load_data():
    # Load the Titanic dataset from Seaborn.

    df = sns.load_dataset("titanic")

    return df


def inspect_data(df):
    # Display basic information about the dataset

    print("\nDataset Information")
    print("Shape:", df.shape)

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nStatistical Summary")
    print(df.describe(include="all"))


def clean_data(df):
    # Prepare the dataset for exploratory analysis and machine learning.


    df = df.copy()

    # Remove columns that are redundant orr directly reveal the target variable
    columns_to_drop = [
        "alive",
        "class",
        "embark_town",
        "who",
        "adult_male",
        "deck",
        "alone"
    ]

    df = df.drop(columns=columns_to_drop)

    # Standardise categorical text values
    df["sex"] = df["sex"].str.strip().str.lower()

    df["embarked"] = df["embarked"].str.strip().str.upper()

    # Check target variable
    df = df.dropna(subset=["survived"])

    # Check impossible values
    df.loc[df["age"] < 0, "age"] = pd.NA
    df.loc[df["fare"] < 0, "fare"] = pd.NA

    print("\nCleaning Summary")
    print("Cleaned shape:", df.shape)
    print("\nRemaining missing values:")
    print(df.isnull().sum())

    return df


def save_data(df):
    # Save the cleaned dataset as a CSV file.

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DATA_DIR / "titanic_cleaned.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")


def main():

    # Load
    df = load_data()

    # Inspect original data
    inspect_data(df)

    # Clean
    cleaned_df = clean_data(df)

    # Save
    save_data(cleaned_df)


if __name__ == "__main__":
    main()
