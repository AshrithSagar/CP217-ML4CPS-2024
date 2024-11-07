"""
dataset.py
"""

import os
import re
from typing import List, Union

import pandas as pd
from openpyxl import load_workbook


class DatasetLoaderXL:
    """
    A class to load datasets from Excel files into lists of lists.
    Using openpyxl to load Excel files.
    """

    def __init__(
        self,
        dataset_dir: Union[str, os.PathLike],
        verbose: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.dataset = {}
        self.suburb_df = pd.DataFrame()
        self.verbose = verbose

    def load_all_datasets(self) -> None:
        """Load all Excel files from the dataset directory using openpyxl."""
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(self.dataset_dir, filename)
                suburb_name = re.search(r"(.+)-Suburb - XLSX.xlsx", filename)
                if suburb_name:
                    self.dataset[suburb_name.group(1)] = self.load_dataset(file_path)

    def load_dataset(self, file_path) -> List[List]:
        """Load a single Excel file into a list of lists."""
        try:
            workbook = load_workbook(file_path)
            sheet = workbook.active
            data = [row for row in sheet.iter_rows(values_only=True)]
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def list_suburbs(self) -> List[str]:
        """List all the suburbs in the dataset."""
        return list(self.dataset.keys())

    def get_data(self, suburb_name: str) -> pd.DataFrame:
        """Get the data for a specific suburb as a DataFrame."""
        if suburb_name not in self.dataset:
            print(f"No data found for suburb: {suburb_name}")
            return pd.DataFrame()

        data = self.dataset[suburb_name]
        df = pd.DataFrame(
            data,
            columns=["Category", "Subcategory", "Value", "Extra1", "Extra2"],
        )
        if df.empty:
            return

        df["Category"] = df["Category"].ffill()
        df.drop(columns=["Extra1", "Extra2"], inplace=True)
        self.suburb_df = df
        return self.suburb_df

    def list_categories(self) -> pd.Series:
        """Get categories list"""
        return (
            self.suburb_df["Category"].unique()
            if not self.suburb_df.empty
            else pd.Series([])
        )

    def get_category(self, category: str) -> pd.DataFrame:
        """Filter data for a specific category."""
        return (
            self.suburb_df[self.suburb_df["Category"] == category]
            if not self.suburb_df.empty
            else pd.DataFrame()
        )


if __name__ == "__main__":
    dsxl = DatasetLoaderXL(dataset_dir="../dataset")
    dsxl.load_all_datasets()
    print(dsxl.list_suburbs())
    print(dsxl.get_data("Malvern"))
    print(dsxl.list_categories())
    print(dsxl.get_category("Geography"))
