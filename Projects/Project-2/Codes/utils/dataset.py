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
        self.suburb_df = pd.DataFrame()

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

    def list_categories(self, verbose=None) -> pd.Series:
        """Get categories list"""
        verbose = self.get_verbose(verbose)

        if self.suburb_df.empty:
            self.get_data(self.suburbs[0])

        self.categories = self.suburb_df["Category"].unique()

        if verbose:
            self.console.print("Categories List:", style="bold black")
            for idx, category in enumerate(self.categories, start=1):
                self.console.print(f"{idx}. {category}", style="bold cyan")

    def get_category(self, category: str) -> pd.DataFrame:
        """Filter data for a specific category."""
        return (
            self.suburb_df[self.suburb_df["Category"] == category]
            if not self.suburb_df.empty
            else pd.DataFrame()
        )

    def list_subcategories(self, category=None, verbose=None) -> pd.Series:
        """Get subcategories list"""
        verbose = self.get_verbose(verbose)

        if self.suburb_df.empty:
            self.get_data(self.suburbs[0])

        if category is None:
            category = self.suburb_df["Category"].unique()[0]

        self.subcategories = self.suburb_df[self.suburb_df["Category"] == category][
            "Subcategory"
        ].unique()

        if verbose:
            self.console.print("Subcategories List:", style="bold black")
            for idx, subcategory in enumerate(self.subcategories, start=1):
                self.console.print(f"{idx}. {subcategory}", style="bold cyan")

    def get_category_across_all_suburbs(self, category: str) -> pd.DataFrame:
        """Get data for a specific category across all suburbs."""

        all_dfs = []
        for suburb in self.suburbs:
            df = self.get_data(suburb)
            df = df[df["Category"] == category]
            df = df.drop(columns=["Category"])
            df["Suburb"] = suburb
            all_dfs.append(df)

        all_dfs = pd.concat(all_dfs, ignore_index=True)
        all_dfs.set_index("Suburb", inplace=True)
        all_dfs = all_dfs.sort_index()

        self.category = category
        self.category_df = all_dfs
        return self.category_df

    def get_values_for_subcategory_across_all_suburbs(self) -> pd.Series:
        pivot_df = self.category_df.pivot_table(
            index="Suburb", columns="Subcategory", aggfunc="first"
        )
        pivot_df.columns = pivot_df.columns.droplevel()
        return pivot_df
