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
        self.suburb = None
        self.suburbs = []
        self.suburb_df = pd.DataFrame()
        self.verbose = verbose

    def load_all_datasets(self) -> None:
        """Load all Excel files from the dataset directory using openpyxl."""
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(self.dataset_dir, filename)
                suburb_name = re.search(
                    r"(.+)-Suburb - XLSX.xlsx",
                    filename,
                ).group(1)
                self.dataset[suburb_name] = self.load_dataset(file_path)

    def load_dataset(self, file_path) -> List[List]:
        """Load a single Excel file into a list of lists."""
        try:
            workbook = load_workbook(file_path)
            sheet = workbook.active
            data = []
            for row in sheet.iter_rows(values_only=True):
                data.append(row)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def list_suburbs(self) -> List[str]:
        """List all the suburbs in the dataset."""
        self.suburbs = list(self.dataset.keys())
        return self.suburbs

    def get_data(self, suburb_name: str) -> pd.DataFrame:
        """Get the data for a specific suburb as a DataFrame."""
        self.suburb_df = self.dataset.get(suburb_name, pd.DataFrame())
        self.clean_suburb_df()
        return self.suburb_df

    def clean_suburb_df(self) -> None:
        """Clean the suburb data."""
        df = pd.DataFrame(
            self.suburb_df,
            columns=["Category", "Subcategory", "Value", "Extra1", "Extra2"],
        )
        df["Category"] = df["Category"].ffill()
        df = df.drop(columns=["Extra1", "Extra2"])
        self.suburb_df = df

    def list_categories(self) -> pd.Series:
        """Get categories list"""
        df = self.suburb_df
        return df["Category"].unique() if not df.empty else pd.Series([])

    def get_category(self, category: str) -> pd.DataFrame:
        """Filter data for a specific category."""
        df = self.suburb_df
        df_category = df[df["Category"] == category]
        return df_category if not df.empty else pd.DataFrame()

    def summarize_data(self) -> pd.DataFrame:
        """Summarize data by category."""
        df = self.suburb_df
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
    dsxl = DatasetLoaderXL(dataset_dir="../dataset")
    dsxl.load_all_datasets()
    print(dsxl.list_suburbs())
    print(dsxl.get_data("Malvern"))
    print(dsxl.list_categories())
    print(dsxl.get_category("Geography"))
