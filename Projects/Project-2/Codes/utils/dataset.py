"""
dataset.py
"""

import os
import re
from typing import List, Union

import pandas as pd
from openpyxl import load_workbook
from rich.columns import Columns
from rich.console import Console
from rich.table import Table


class DatasetLoaderXL:
    """
    A class to load datasets from Excel files into lists of lists.
    Using openpyxl to load Excel files.
    """

    def __init__(
        self,
        dataset_dir: Union[str, os.PathLike],
        seed=42,
        console=Console(),
        verbose: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.seed = seed
        self.console = console
        self.verbose = verbose

        self.dataset = {}

    def get_verbose(self, verbose):
        if verbose is None:
            return self.verbose
        else:
            return verbose

    def load_dataset(self, file_path) -> List[List]:
        """Load a single Excel file"""
        try:
            workbook = load_workbook(file_path)
            sheet = workbook.active
            data = [row for row in sheet.iter_rows(values_only=True)]
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_all_datasets(self) -> None:
        """Load all the Excel files using openpyxl."""
        if self.dataset:
            return

        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(self.dataset_dir, filename)
                suburb_name = re.search(r"(.+)-Suburb - XLSX.xlsx", filename)
                if suburb_name:
                    self.dataset[suburb_name.group(1)] = self.load_dataset(file_path)

    def list_suburbs(self, verbose=None):
        """List all the suburbs in the dataset."""
        verbose = self.get_verbose(verbose)
        self.load_all_datasets()
        self.suburbs = sorted(self.dataset.keys())

        if verbose:
            self.console.print("Suburbs List:", style="bold black")
            columns = Columns(
                [
                    f"{idx}: {suburb}"
                    for idx, suburb in enumerate(self.suburbs, start=1)
                ],
                width=25,
            )
            self.console.print(columns, style="bold cyan")

    def get_data(self, suburb_name: str) -> pd.DataFrame:
        """Get the data for a specific suburb as a DataFrame."""
        if suburb_name not in self.dataset:
            self.console.print(f"No data found for suburb: {suburb_name}")
            return

        data = self.dataset[suburb_name]
        df = pd.DataFrame(
            data,
            columns=["Category", "Subcategory", "Value", "Extra1", "Extra2"],
        )
        if df.empty:
            return

        df["Category"] = df["Category"].ffill()
        df.drop(columns=["Extra1", "Extra2"], inplace=True)
        return df

    def list_categories(self, verbose=None) -> pd.Series:
        """Get categories list"""
        verbose = self.get_verbose(verbose)
        suburb_df = self.get_data(self.suburbs[0])
        self.categories = suburb_df["Category"].unique()

        if verbose:
            self.console.print("Categories List:", style="bold black")
            for idx, category in enumerate(self.categories, start=1):
                self.console.print(f"{idx}. {category}", style="bold cyan")

    def get_category(self, category: str, suburb=None) -> pd.DataFrame:
        """Filter data for a specific category."""
        if suburb is None:
            suburb = self.suburbs[0]
        suburb_df = self.get_data(suburb)
        category_df = suburb_df[suburb_df["Category"] == category]
        return category_df

    def list_subcategories(self, category=None, verbose=None) -> pd.Series:
        """Get subcategories list"""
        verbose = self.get_verbose(verbose)
        if not hasattr(self, "categories"):
            self.list_categories()

        def get(category):
            category_df = self.get_category(category)
            return sorted(category_df["Subcategory"].unique())

        if category is None:
            # List for all categories
            subcategories = []
            for category in self.categories:
                subcategories.extend(get(category))
        else:
            subcategories = get(category)

        if verbose:
            self.console.print("Subcategories List:", style="bold black")
            for idx, subcategory in enumerate(subcategories, start=1):
                self.console.print(f"{idx}. {subcategory}", style="bold cyan")

        return subcategories

    def get_category_across_all_suburbs(self, category: str) -> pd.DataFrame:
        """Get data for a specific category across all suburbs."""

        data = []
        for suburb in self.suburbs:
            category_df = self.get_category(category, suburb)
            df = category_df.drop(columns=["Category"])
            df["Suburb"] = suburb
            data.append(df)

        data = pd.concat(data, ignore_index=True)
        data.set_index("Suburb", inplace=True)
        data = data.groupby("Suburb").apply(
            lambda x: x.set_index("Subcategory")["Value"]
        )
        data = data.sort_index()
        return data
