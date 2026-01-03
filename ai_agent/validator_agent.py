import pandas as pd
from typing import Dict, List


class ValidatorAgent:
    """
    Deterministic dataset validator.
    Designed to run safely in Streamlit, Jupyter, Colab, or CLI.
    """

    def __init__(self, expected_schema: Dict[str, str] | None = None):
        """
        expected_schema example:
        {
            "Threat Level": "int64",
            "Cost": "float64",
            "Department Affected": "object"
        }
        """
        self.expected_schema = expected_schema or {}

    # --------------------------------------------------
    # 1. SAFE TYPE COERCION (NON-FATAL)
    # --------------------------------------------------

    def coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, expected_type in self.expected_schema.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(expected_type)
                except Exception:
                    # Do not fail — handled in warnings
                    pass

        return df

    # --------------------------------------------------
    # 2. CORE VALIDATION CHECKS
    # --------------------------------------------------

    def validate(self, df: pd.DataFrame) -> Dict:
        report = {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "errors": [],
            "warnings": [],
            "schema": {},
        }

        # ---- Column presence ----
        missing_columns = [
            col for col in self.expected_schema
            if col not in df.columns
        ]
        if missing_columns:
            report["errors"].append(
                f"Missing required columns: {missing_columns}"
            )

        # ---- Data types ----
        for col, expected_type in self.expected_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                report["schema"][col] = actual_type
                if actual_type != expected_type:
                    report["warnings"].append(
                        f"Type mismatch in '{col}': found {actual_type}, expected {expected_type}"
                    )

        # ---- Missing values ----
        nulls = df.isnull().sum()
        critical_nulls = {
            col: int(count)
            for col, count in nulls.items()
            if count > 0 and col in self.expected_schema
        }

        if critical_nulls:
            report["warnings"].append(
                f"Missing values detected: {critical_nulls}"
            )

        # ---- Duplicates ----
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            report["errors"].append(
                f"Duplicate rows detected: {duplicate_count}"
            )

        return report

    # --------------------------------------------------
    # 3. DECISION LOGIC
    # --------------------------------------------------

    def can_proceed(self, validation_report: Dict) -> bool:
        """
        Only block pipeline on hard errors.
        """
        return len(validation_report["errors"]) == 0
