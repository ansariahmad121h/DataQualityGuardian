import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class MLQualityDetector:

    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = None

    def detect_missing_values(self, df):

        missing = df.isnull().sum()

        return missing[
            missing > 0
        ].to_dict()

    def detect_duplicates(self, df):

        return int(
            df.duplicated().sum()
        )

    def detect_outliers(self, df):

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) == 0:
            return []

        data = df[numeric_columns].copy()

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.fillna(
            data.median()
        )

        if len(data) < 5:
            return []

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )

        predictions = self.model.fit_predict(data)

        anomaly_indices = np.where(
            predictions == -1
        )[0]

        return anomaly_indices.tolist()

    def analyze(self, df):

        missing = self.detect_missing_values(df)

        duplicates = self.detect_duplicates(df)

        outliers = self.detect_outliers(df)

        return {
            "missing_values": missing,
            "duplicate_rows": duplicates,
            "outlier_rows": outliers,
            "total_outliers": len(outliers)
        }