import numpy as np

from ml_detector import MLQualityDetector
from data_tools import DataTools


class DataQualityAgent:

    def __init__(self, contamination=0.05):

        self.detector = MLQualityDetector(
            contamination=contamination
        )

        self.tools = DataTools()

    # ========================================================
    # DATASET UNDERSTANDING
    # ========================================================

    def understand_dataset(self, df):

        summary = self.tools.dataset_summary(df)

        return summary

    # ========================================================
    # DETECT PROBLEMS
    # ========================================================

    def detect_issues(self, df):

        analysis = self.detector.analyze(df)

        issues = []

        # Missing values
        missing = analysis["missing_values"]

        if missing:

            total_missing = sum(
                missing.values()
            )

            issues.append({

                "type": "missing",

                "title": "Missing information",

                "description":
                    f"{total_missing} empty value(s) "
                    f"were found across "
                    f"{len(missing)} column(s).",

                "severity": "Medium",

                "confidence": 98,

                "action": "Fill missing values",

                "reason":
                    "Missing values can reduce the reliability "
                    "of analysis and machine-learning models.",

                "safe": True
            })

        # Duplicates
        duplicates = analysis[
            "duplicate_rows"
        ]

        if duplicates > 0:

            issues.append({

                "type": "duplicate",

                "title": "Duplicate records",

                "description":
                    f"{duplicates} duplicate record(s) "
                    f"were identified.",

                "severity": "Medium",

                "confidence": 99,

                "action": "Remove duplicates",

                "reason":
                    "Exact duplicate records can cause "
                    "certain observations to be counted twice.",

                "safe": True
            })

        # Outliers
        outliers = analysis[
            "outlier_rows"
        ]

        if len(outliers) > 0:

            confidence = min(
                98,
                75 + len(outliers) * 3
            )

            issues.append({

                "type": "outlier",

                "title": "Unusual records",

                "description":
                    f"{len(outliers)} record(s) appear "
                    f"significantly different from the "
                    f"normal data pattern.",

                "severity": "High",

                "confidence": confidence,

                "action": "Review unusual records",

                "reason":
                    "These observations differ strongly "
                    "from the normal numerical pattern. "
                    "They may represent errors or legitimate "
                    "rare cases.",

                "safe": False
            })

        return analysis, issues

    # ========================================================
    # AGENT PLANNING
    # ========================================================

    def create_plan(self, issues):

        plan = []

        for issue in issues:

            if issue["type"] == "missing":

                plan.append({

                    "type": "missing",

                    "decision":
                        "Automatically fill missing values",

                    "reason":
                        "The dataset contains empty values. "
                        "Numeric fields can be filled using "
                        "their median, while categorical fields "
                        "can use their most common value.",

                    "confidence":
                        issue["confidence"],

                    "requires_approval": False
                })

            elif issue["type"] == "duplicate":

                plan.append({

                    "type": "duplicate",

                    "decision":
                        "Automatically remove exact duplicates",

                    "reason":
                        "Exact duplicates can safely be removed "
                        "without changing the meaning of unique records.",

                    "confidence":
                        issue["confidence"],

                    "requires_approval": False
                })

            elif issue["type"] == "outlier":

                plan.append({

                    "type": "outlier",

                    "decision":
                        "Ask the user before removing unusual records",

                    "reason":
                        "An unusual value may be an error, but it "
                        "could also represent a legitimate rare event.",

                    "confidence":
                        issue["confidence"],

                    "requires_approval": True
                })

        return plan

    # ========================================================
    # EXECUTE SAFE ACTIONS
    # ========================================================

    def execute(
        self,
        df,
        plan,
        approved_outliers=False
    ):

        cleaned = df.copy()

        execution_log = []

        for action in plan:

            action_type = action["type"]

            # Missing values
            if action_type == "missing":

                cleaned, changes = (
                    self.tools.fill_missing_values(
                        cleaned
                    )
                )

                execution_log.extend(
                    changes
                )

            # Duplicates
            elif action_type == "duplicate":

                cleaned, removed = (
                    self.tools.remove_duplicates(
                        cleaned
                    )
                )

                if removed > 0:

                    execution_log.append(
                        f"Removed {removed} duplicate record(s)."
                    )

            # Outliers
            elif action_type == "outlier":

                if approved_outliers:

                    # Recalculate outliers on the current dataset
                    current_analysis = (
                        self.detector.analyze(
                            cleaned
                        )
                    )

                    indices = current_analysis[
                        "outlier_rows"
                    ]

                    cleaned, removed = (
                        self.tools.remove_outlier_rows(
                            cleaned,
                            indices
                        )
                    )

                    if removed > 0:

                        execution_log.append(
                            f"Removed {removed} unusual record(s) "
                            f"after user approval."
                        )

                else:

                    execution_log.append(
                        "Unusual records were flagged for "
                        "review and were not automatically removed."
                    )

        return cleaned, execution_log

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self, original_df, cleaned_df):

        original_score = (
            self.tools.calculate_quality_score(
                original_df
            )
        )

        cleaned_score = (
            self.tools.calculate_quality_score(
                cleaned_df
            )
        )

        improvement = round(
            cleaned_score - original_score,
            2
        )

        if cleaned_score >= original_score:

            status = "accepted"

        else:

            status = "rollback"

        return {

            "original_score": original_score,

            "cleaned_score": cleaned_score,

            "improvement": improvement,

            "status": status
        }

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        df,
        approved_outliers=False
    ):

        summary = self.understand_dataset(
            df
        )

        analysis, issues = (
            self.detect_issues(df)
        )

        plan = self.create_plan(
            issues
        )

        cleaned_df, execution_log = (
            self.execute(
                df,
                plan,
                approved_outliers
            )
        )

        validation = self.validate(
            df,
            cleaned_df
        )

        # Rollback if quality became worse
        if validation["status"] == "rollback":

            cleaned_df = df.copy()

            validation = self.validate(
                df,
                cleaned_df
            )

            execution_log.append(
                "The proposed changes were rolled back "
                "because the resulting quality did not improve."
            )

        return {

            "summary": summary,

            "analysis": analysis,

            "issues": issues,

            "plan": plan,

            "execution_log": execution_log,

            "validation": validation,

            "cleaned_data": cleaned_df
        }