import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from agent import DataQualityAgent
from data_tools import DataTools


class BenchmarkComparator:
    """
    Runs a reproducible scientific benchmark comparing DataQualityGuardian
    against existing baseline approaches on real-world tabular datasets.
    """

    def __init__(self, random_state: int = 42, contamination: float = 0.05):
        self.random_state = random_state
        self.contamination = contamination
        self.tools = DataTools()

    def load_real_dataset(self, name: str = "Breast Cancer Wisconsin"):
        if name == "Wine Quality (UCI)":
            bunch = load_wine(as_frame=True)
        else:
            bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        target_col = "target"
        feature_cols = [c for c in df.columns if c != target_col]
        return df, feature_cols, target_col

    def inject_controlled_corruption(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        missing_rate: float = 0.05,
        duplicate_rate: float = 0.05,
        anomaly_rate: float = 0.05,
    ):
        rng = np.random.default_rng(self.random_state)
        corrupted = df.copy().reset_index(drop=True)
        n = len(corrupted)
        is_anomaly = np.zeros(n, dtype=int)

        n_anom = max(1, int(anomaly_rate * n))
        anom_idx = rng.choice(n, size=n_anom, replace=False)
        f1, f2 = feature_cols[0], feature_cols[min(2, len(feature_cols) - 1)]
        corrupted.loc[anom_idx, f1] = corrupted.loc[anom_idx, f1] * 2.5 + 15.0
        corrupted.loc[anom_idx, f2] = corrupted.loc[anom_idx, f2] * 2.5 + 40.0
        is_anomaly[anom_idx] = 1

        for col in feature_cols[: min(3, len(feature_cols))]:
            miss_idx = rng.choice(n, size=max(1, int(missing_rate * n)), replace=False)
            corrupted.loc[miss_idx, col] = np.nan

        n_dups = max(1, int(duplicate_rate * n))
        dup_idx = rng.choice(n, size=n_dups, replace=False)
        dup_rows = corrupted.iloc[dup_idx].copy()
        dup_labels = is_anomaly[dup_idx]

        corrupted = pd.concat([corrupted, dup_rows], ignore_index=True)
        is_anomaly = np.concatenate([is_anomaly, dup_labels])

        return corrupted, is_anomaly

    def run_comparison(
        self,
        dataset_name: str = "Breast Cancer Wisconsin",
        missing_rate: float = 0.05,
        duplicate_rate: float = 0.05,
        anomaly_rate: float = 0.05,
    ) -> pd.DataFrame:
        df, feature_cols, target_col = self.load_real_dataset(dataset_name)
        train_df, test_df = train_test_split(
            df, test_size=0.25, random_state=self.random_state, stratify=df[target_col]
        )

        dirty_train, y_anom_true = self.inject_controlled_corruption(
            train_df, feature_cols, missing_rate, duplicate_rate, anomaly_rate
        )
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        raw_score = self.tools.calculate_quality_score(dirty_train)

        results = []

        # 1. Raw Unprocessed Data
        t0 = time.perf_counter()
        X_raw_imp = dirty_train[feature_cols].fillna(dirty_train[feature_cols].median())
        clf_raw = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf_raw.fit(X_raw_imp, dirty_train[target_col])
        acc_raw = accuracy_score(y_test, clf_raw.predict(X_test)) * 100
        t_raw = (time.perf_counter() - t0) * 1000

        results.append({
            "Framework / Baseline": "1. Raw Unprocessed Data",
            "Anomaly Precision (%)": 0.0,
            "Anomaly Recall (%)": 0.0,
            "Anomaly F1-Score (%)": 0.0,
            "Data Health Score (%)": raw_score,
            "Downstream ML Acc. (%)": round(acc_raw, 2),
            "Latency (ms)": round(t_raw, 2),
            "Self-Healing & Rollback": "None",
        })

        # 2. Univariate IQR / Static Rules
        t0 = time.perf_counter()
        X_num = dirty_train[feature_cols].fillna(dirty_train[feature_cols].median())
        iqr_inlier_mask = np.ones(len(dirty_train), dtype=bool)
        for col in feature_cols:
            q1, q3 = X_num[col].quantile(0.25), X_num[col].quantile(0.75)
            iqr = q3 - q1
            iqr_inlier_mask &= X_num[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        pred_iqr = (~iqr_inlier_mask).astype(int)
        df_iqr_clean = dirty_train[iqr_inlier_mask].copy()
        score_iqr = self.tools.calculate_quality_score(df_iqr_clean)

        X_iqr_train = df_iqr_clean[feature_cols].fillna(df_iqr_clean[feature_cols].median())
        clf_iqr = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf_iqr.fit(X_iqr_train, df_iqr_clean[target_col])
        acc_iqr = accuracy_score(y_test, clf_iqr.predict(X_test)) * 100
        t_iqr = (time.perf_counter() - t0) * 1000

        results.append({
            "Framework / Baseline": "2. Univariate IQR / Static Rules",
            "Anomaly Precision (%)": round(precision_score(y_anom_true, pred_iqr, zero_division=0) * 100, 2),
            "Anomaly Recall (%)": round(recall_score(y_anom_true, pred_iqr, zero_division=0) * 100, 2),
            "Anomaly F1-Score (%)": round(f1_score(y_anom_true, pred_iqr, zero_division=0) * 100, 2),
            "Data Health Score (%)": score_iqr,
            "Downstream ML Acc. (%)": round(acc_iqr, 2),
            "Latency (ms)": round(t_iqr, 2),
            "Self-Healing & Rollback": "Alert / Hard Drop Only",
        })

        # 3. Standalone Isolation Forest
        t0 = time.perf_counter()
        iso = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        pred_iso = (iso.fit_predict(X_num) == -1).astype(int)
        df_iso_clean = dirty_train[pred_iso == 0].copy()
        score_iso = self.tools.calculate_quality_score(df_iso_clean)

        X_iso_train = df_iso_clean[feature_cols].fillna(df_iso_clean[feature_cols].median())
        clf_iso = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf_iso.fit(X_iso_train, df_iso_clean[target_col])
        acc_iso = accuracy_score(y_test, clf_iso.predict(X_test)) * 100
        t_iso = (time.perf_counter() - t0) * 1000

        results.append({
            "Framework / Baseline": "3. Standalone Isolation Forest",
            "Anomaly Precision (%)": round(precision_score(y_anom_true, pred_iso, zero_division=0) * 100, 2),
            "Anomaly Recall (%)": round(recall_score(y_anom_true, pred_iso, zero_division=0) * 100, 2),
            "Anomaly F1-Score (%)": round(f1_score(y_anom_true, pred_iso, zero_division=0) * 100, 2),
            "Data Health Score (%)": score_iso,
            "Downstream ML Acc. (%)": round(acc_iso, 2),
            "Latency (ms)": round(t_iso, 2),
            "Self-Healing & Rollback": "No Auto-Heal / No Guard",
        })

        # 4. DataQualityGuardian
        t0 = time.perf_counter()
        agent = DataQualityAgent()
        detected_indices = agent.detector.detect_outliers(dirty_train[feature_cols])
        pred_dqg = np.zeros(len(dirty_train), dtype=int)
        valid_idx = [i for i in detected_indices if 0 <= i < len(dirty_train)]
        pred_dqg[valid_idx] = 1

        output = agent.analyze(dirty_train, approved_outliers=True)
        df_dqg_clean = output["cleaned_data"]
        score_dqg = output["validation"].get(
            "cleaned_score", self.tools.calculate_quality_score(df_dqg_clean)
        )

        clf_dqg = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf_dqg.fit(df_dqg_clean[feature_cols], df_dqg_clean[target_col])
        acc_dqg = accuracy_score(y_test, clf_dqg.predict(X_test)) * 100
        t_dqg = (time.perf_counter() - t0) * 1000

        results.append({
            "Framework / Baseline": "4. DataQualityGuardian (Ours)",
            "Anomaly Precision (%)": round(precision_score(y_anom_true, pred_dqg, zero_division=0) * 100, 2),
            "Anomaly Recall (%)": round(recall_score(y_anom_true, pred_dqg, zero_division=0) * 100, 2),
            "Anomaly F1-Score (%)": round(f1_score(y_anom_true, pred_dqg, zero_division=0) * 100, 2),
            "Data Health Score (%)": score_dqg,
            "Downstream ML Acc. (%)": round(acc_dqg, 2),
            "Latency (ms)": round(t_dqg, 2),
            "Self-Healing & Rollback": "Auto-Heal + HITL + Rollback",
        })

        return pd.DataFrame(results)


if __name__ == "__main__":
    comp = BenchmarkComparator()
    df_res = comp.run_comparison("Breast Cancer Wisconsin")
    print(df_res.to_string(index=False))