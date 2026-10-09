import pandas as pd
import numpy as np


class DataTools:

    @staticmethod
    def remove_duplicates(df):
        before = len(df)

        cleaned = df.drop_duplicates().copy()

        removed = before - len(cleaned)

        return cleaned, removed

    @staticmethod
    def fill_missing_values(df):
        cleaned = df.copy()

        numeric_columns = cleaned.select_dtypes(
            include=np.number
        ).columns

        categorical_columns = cleaned.select_dtypes(
            exclude=np.number
        ).columns

        changes = []

        for column in numeric_columns:

            count = int(cleaned[column].isnull().sum())

            if count > 0:

                median = cleaned[column].median()

                cleaned[column] = cleaned[column].fillna(median)

                changes.append(
                    f"Filled {count} missing value(s) in {column} "
                    f"using the column median."
                )

        for column in categorical_columns:

            count = int(cleaned[column].isnull().sum())

            if count > 0:

                mode = cleaned[column].mode()

                if not mode.empty:

                    value = mode.iloc[0]

                else:

                    value = "Unknown"

                cleaned[column] = cleaned[column].fillna(value)

                changes.append(
                    f"Filled {count} missing value(s) in {column}."
                )

        return cleaned, changes

    @staticmethod
    def remove_outlier_rows(df, indices):

        if not indices:
            return df.copy(), 0

        valid_indices = [
            index for index in indices
            if index in df.index
        ]

        cleaned = df.drop(
            index=valid_indices
        ).copy()

        return cleaned, len(valid_indices)

    @staticmethod
    def calculate_quality_score(df):

        if df.empty:
            return 0

        total_cells = df.shape[0] * df.shape[1]

        if total_cells == 0:
            return 0

        missing_cells = int(
            df.isnull().sum().sum()
        )

        completeness = (
            1 -
            missing_cells / total_cells
        )

        duplicate_ratio = (
            df.duplicated().sum()
            /
            max(len(df), 1)
        )

        score = (
            completeness * 100
            -
            duplicate_ratio * 20
        )

        return round(
            max(0, min(100, score)),
            2
        )

    @staticmethod
    def dataset_summary(df):

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_columns = list(
            df.select_dtypes(
                exclude=np.number
            ).columns
        )

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_cells": int(
                df.isnull().sum().sum()
            ),
            "duplicate_rows": int(
                df.duplicated().sum()
            )
        }