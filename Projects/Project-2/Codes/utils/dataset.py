"""
dataset.py
"""

import os

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

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataframes = {}

    def load_all_datasets(self):
        """Load all Excel files from the dataset directory using openpyxl."""
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(self.dataset_dir, filename)
                self.dataframes[filename] = self.load_dataset(file_path)
        return self.dataframes

    def load_dataset(self, file_path):
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

    def get_data(self, suburb_name):
        """Get the data for a specific suburb."""
        filename = f"{suburb_name}-Suburb - XLSX.xlsx"
        return self.dataframes.get(filename, None)


if __name__ == "__main__":
    dfl = DatasetLoader(dataset_dir="../dataset")
    dataset = dfl.load_all_datasets()
    df = dfl.get_dataframe("Malvern")

    dsxl = DatasetLoaderXL(dataset_dir="../dataset")
    dataset = dsxl.load_all_datasets()
    data = dsxl.get_data("Malvern")
