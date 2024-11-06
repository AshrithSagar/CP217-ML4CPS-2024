"""
dataset.py
"""

import os
import re
from typing import List, Union

import pandas as pd
from openpyxl import load_workbook


class DatasetLoader:
    """
    A class to load datasets from Excel files into DataFrames.
    Using pandas to load Excel files.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataframes = {}

    def load_all_datasets(self):
        """Load all Excel files from the dataset directory."""
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(self.dataset_dir, filename)
                self.dataframes[filename] = self.load_dataset(file_path)
        return self.dataframes

    def load_dataset(self, file_path):
        """Load a single Excel file into a DataFrame."""
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def get_dataframe(self, suburb_name):
        """Get the DataFrame for a specific suburb."""
        filename = f"{suburb_name}-Suburb - XLSX.xlsx"
        return self.dataframes.get(filename, None)


class DatasetLoaderXL:
    """
    A class to load datasets from Excel files into lists of lists.
    Using openpyxl to load Excel files.
    """

    def __init__(self, dataset_dir: Union[str, os.PathLike]) -> None:
        self.dataset_dir = dataset_dir
        self.dataframes = {}
        self.suburb = None

    def load_all_datasets(self) -> None:
        """Load all Excel files from the dataset directory using openpyxl."""
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(self.dataset_dir, filename)
                suburb_name = re.search(r"(.+)-Suburb - XLSX.xlsx", filename).group(1)
                self.dataframes[suburb_name] = self.load_dataset(file_path)

    def load_dataset(self, file_path) -> List[List]:
        """Load a single Excel file into a list of lists."""
        try:
            workbook = load_workbook(file_path)
            sheet = workbook.active  # Load the active sheet
            data = []
            for row in sheet.iter_rows(values_only=True):
                data.append(row)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def get_data(self, suburb_name: str) -> pd.DataFrame:
        """Get the data for a specific suburb as a DataFrame."""
        return self.dataframes.get(suburb_name, pd.DataFrame())

    def list_categories(self) -> pd.Series:
        """Get categories list"""

        return (
            self.suburb_df["Category"].unique()
            if not self.suburb_df.empty
            else pd.Series([])
        )

    def filter_by_category(self, suburb_name: str, category: str) -> pd.DataFrame:
        """Filter data for a specific category."""
        df = self.get_data(suburb_name)
        return df[df["Category"] == category] if not df.empty else pd.DataFrame()

    def summarize_data(self, suburb_name: str) -> pd.DataFrame:
        """Summarize data by category."""
        df = self.get_data(suburb_name)
        return (
            df.groupby("Category")["Value"].sum().reset_index()
            if not df.empty
            else pd.DataFrame()
        )

    def visualize_data(self, suburb_name: str):
        """Visualize the data for a specific suburb."""
        import matplotlib.pyplot as plt

        df = self.summarize_data(suburb_name)
        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(df["Category"], df["Value"], color="skyblue")
            plt.title(f"Summary of Values by Category for {suburb_name}")
            plt.xlabel("Category")
            plt.ylabel("Total Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data available to visualize for suburb: {suburb_name}")


if __name__ == "__main__":
    dfl = DatasetLoader(dataset_dir="../dataset")
    dataset = dfl.load_all_datasets()
    df = dfl.get_dataframe("Malvern")

    dsxl = DatasetLoaderXL(dataset_dir="../dataset")
    dataset = dsxl.load_all_datasets()
    data = dsxl.get_data("Malvern")
