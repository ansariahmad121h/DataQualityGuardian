# DataQualityGuardian (DataGuard AI)

An automated AI-driven data quality monitoring, anomaly detection, and self-healing remediation framework for structured tabular datasets. Built with Python, Pandas, NumPy, Scikit-Learn, and Streamlit, it runs locally on standard CPU hardware without external API keys or cloud LLM dependencies[cite: 1, 11].

---

## Key Features

* **Zero-Config Statistical Profiling:** Automatically profiles row and column counts, data types, missing cells, and duplicate records[cite: 1, 11].
* **Hybrid Anomaly Detection:** Combines deterministic structural checks with unsupervised **Isolation Forest** (`contamination=0.05`, `random_state=42`) to catch multivariate outliers missed by single-column rules[cite: 1, 11].
* **Risk-Aware Autonomous Planning:** Separates safe automated fixes (column median/mode imputation and exact deduplication) from high-risk outlier removal that requires human-in-the-loop approval[cite: 1, 11].
* **Post-Healing Validation & Rollback Guard:** Verifies the dataset health score after cleaning and automatically rolls back changes if quality degrades[cite: 1, 11].
* **Interactive Streamlit UI & CSV Export:** Displays a 0–100% Data Health score, issue cards, agent decisions, approval controls, and one-click cleaned dataset export[cite: 1, 9, 11].
* **Scientific Baseline Comparator (`benchmark_comparator.py`):** Evaluates DataQualityGuardian side-by-side against raw data, univariate IQR/static rules, and standalone Isolation Forest on real datasets.

---

## Project Structure

* `app.py` — Streamlit web dashboard, health overview, review/approval flow, and dataset export[cite: 9, 11].
* `agent.py` — `DataQualityAgent` orchestrating dataset understanding, issue detection, planning, execution, and rollback validation[cite: 10, 11].
* `ml_detector.py` — `MLQualityDetector` for missing/duplicate detection and Isolation Forest outlier scoring.
* `data_tools.py` — `DataTools` for median/mode imputation, deduplication, outlier removal, and quality scoring.
* `benchmark_comparator.py` — `BenchmarkComparator` for controlled corruption injection and multi-baseline benchmarking.

---

## Getting Started

### 1. Activate Your Environment & Install Dependencies
```powershell
cd D:\DataQualityGuardian
.\venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn streamlit
