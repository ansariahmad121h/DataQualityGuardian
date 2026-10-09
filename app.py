import streamlit as st
import pandas as pd

from agent import DataQualityAgent


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="DataGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 30px;
    padding-bottom: 60px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #2563eb;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

.brand-name {
    font-size: 23px;
    font-weight: 800;
    color: #0f172a;
}

.brand-subtitle {
    font-size: 13px;
    color: #64748b;
}

.section {
    color: #0f172a;
    font-size: 22px;
    font-weight: 800;
    margin-top: 32px;
    margin-bottom: 15px;
}

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px;
    min-height: 120px;
}

.card-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}

.card-value {
    color: #0f172a;
    font-size: 29px;
    font-weight: 850;
    margin-top: 7px;
}

.card-small {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 4px;
}

.score {
    background: #0f172a;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    color: white;
}

.score-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
}

.score-number {
    font-size: 52px;
    font-weight: 900;
    margin: 8px 0;
}

.score-status {
    color: #cbd5e1;
}

.issue {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 17px;
    margin-bottom: 12px;
}

.issue-title {
    color: #0f172a;
    font-size: 16px;
    font-weight: 750;
}

.issue-description {
    color: #64748b;
    font-size: 13px;
    margin-top: 5px;
}

.ai-card {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #f8fafc
    );
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    padding: 25px;
}

.ai-title {
    color: #1d4ed8;
    font-weight: 800;
    font-size: 18px;
    margin-bottom: 12px;
}

.decision {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 12px;
    padding: 15px;
    margin-top: 10px;
}

.decision-title {
    color: #0f172a;
    font-weight: 750;
}

.decision-text {
    color: #64748b;
    font-size: 13px;
    margin-top: 5px;
}

.confidence {
    color: #2563eb;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}

.safe {
    color: #15803d;
    font-size: 12px;
    font-weight: 700;
}

.review {
    color: #b45309;
    font-size: 12px;
    font-weight: 700;
}

.success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 15px;
    color: #166534;
    margin-bottom: 10px;
}

.review-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 16px;
    padding: 22px;
    margin-top: 15px;
}

.stButton > button {
    border-radius: 10px;
    height: 46px;
    font-weight: 750;
}

.stDownloadButton > button {
    border-radius: 10px;
    height: 46px;
    font-weight: 750;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BRAND
# ============================================================

st.markdown("""
<div class="brand">

    <div class="brand-icon">
        🛡️
    </div>

    <div>
        <div class="brand-name">
            DataGuard AI
        </div>

        <div class="brand-subtitle">
            Autonomous data quality management
        </div>
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section">Start with your dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

sample = st.selectbox(
    "Or choose a sample dataset",
    [
        "None",
        "Employee Data",
        "Student Performance",
        "IoT Sensor Data",
        "Sales Data"
    ]
)


# ============================================================
# DATA LOADING
# ============================================================

df = None
dataset_name = None


if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        dataset_name = uploaded_file.name

    except:

        st.error(
            "This file could not be read. "
            "Please upload a valid CSV file."
        )


elif sample != "None":

    if sample == "Employee Data":

        df = pd.DataFrame({

            "Age": [
                22, 24, 25, 26, 27,
                23, 25, 28, 24, 99
            ],

            "Salary": [
                35000, 42000, 45000,
                48000, 52000, 39000,
                46000, 58000, 41000,
                5000000
            ],

            "Experience": [
                1, 2, 3, 3, 4,
                2, 3, 5, 2, 75
            ],

            "Department": [
                "IT", "HR", "IT", "Finance",
                "IT", "HR", "IT", "Finance",
                "IT", "IT"
            ]
        })

        dataset_name = "Employee Data"


    elif sample == "Student Performance":

        df = pd.DataFrame({

            "StudentID": [
                "S001", "S002", "S003", "S004",
                "S005", "S006", "S007", "S008",
                "S009", "S010", "S011", "S012"
            ],

            "Age": [
                19, 20, 19, 21,
                20, 19, 21, 20,
                19, 22, 20, 19
            ],

            "Attendance": [
                92, 85, 90, 78,
                95, 82, 88, 76,
                91, 12, 87, 93
            ],

            "Math": [
                88, 76, 91, 65,
                94, 73, 84, 61,
                89, 5, 80, 90
            ],

            "Programming": [
                91, 82, 87, 72,
                96, 79, 86, 68,
                92, 8, 83, 88
            ],

            "Department": [
                "CSE", "CSE", "AIML", "CSE",
                "AIML", "CSE", "AIML", "CSE",
                "AIML", "CSE", "AIML", "CSE"
            ]
        })

        dataset_name = "Student Performance"


    elif sample == "IoT Sensor Data":

        df = pd.DataFrame({

            "Temperature": [
                42.1, 42.5, 43.0, 42.8,
                43.2, 43.5, 44.0, 44.2,
                44.5, 95.0, 44.8, 45.0
            ],

            "Pressure": [
                101.2, 101.3, 101.1, 101.2,
                101.4, 101.3, 101.2, 101.1,
                101.3, 150.0, 101.2, 101.4
            ],

            "Vibration": [
                0.21, 0.22, 0.20, 0.23,
                0.21, 0.22, 0.24, 0.25,
                0.23, 8.50, 0.24, 0.25
            ],

            "Humidity": [
                55, 54, 56, 55,
                54, 55, 53, 52,
                53, 5, 52, 51
            ],

            "MotorSpeed": [
                1450, 1460, 1455, 1462,
                1458, 1465, 1470, 1472,
                1468, 5000, 1470, 1475
            ]
        })

        dataset_name = "IoT Sensor Data"


    elif sample == "Sales Data":

        df = pd.DataFrame({

            "Quantity": [
                5, 3, 8, 4, 6,
                2, 9, 5, 7, 500,
                4, 6, 3, 8, 5
            ],

            "UnitPrice": [
                120, 250, 150, 320, 180,
                500, 110, 275, 190, 50000,
                220, 310, 175, 200, 240
            ],

            "Discount": [
                5, 10, 5, 15, 8,
                12, 5, 10, 7, 0,
                8, 12, 5, 10, 6
            ],

            "CustomerAge": [
                25, 32, 28, 41, 35,
                29, 26, 38, 31, 150,
                34, 42, 27, 36, 30
            ],

            "Region": [
                "South", "North", "South", "West",
                "East", "South", "North", "West",
                "East", "South", "North", "West",
                "East", "South", "North"
            ]
        })

        dataset_name = "Sales Data"


# ============================================================
# DATASET PREVIEW
# ============================================================

if df is not None:

    st.session_state.df = df
    st.session_state.dataset_name = dataset_name

    st.markdown(
        '<div class="section">Dataset overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    missing_count = int(
        df.isnull().sum().sum()
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    with col1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Dataset</div>
                <div class="card-value"
                     style="font-size:20px;">
                    {dataset_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Records</div>
                <div class="card-value">
                    {len(df):,}
                </div>
                <div class="card-small">
                    Rows
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Fields</div>
                <div class="card-value">
                    {len(df.columns)}
                </div>
                <div class="card-small">
                    Columns
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-label">Missing data</div>
                <div class="card-value">
                    {missing_count}
                </div>
                <div class="card-small">
                    Empty cells
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section">Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # ANALYZE
    # ========================================================

    if st.button(
        "✨ Analyze my data",
        use_container_width=True
    ):

        agent = DataQualityAgent()

        with st.spinner(
            "DataGuard is analyzing your dataset..."
        ):

            result = agent.analyze(
                df,
                approved_outliers=False
            )

        st.session_state.result = result


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result is not None:

    result = st.session_state.result

    validation = result[
        "validation"
    ]

    summary = result[
        "summary"
    ]

    issues = result[
        "issues"
    ]

    plan = result[
        "plan"
    ]


    # ========================================================
    # HEALTH
    # ========================================================

    score = validation[
        "cleaned_score"
    ]

    if score >= 90:
        status = "Excellent"
    elif score >= 70:
        status = "Good"
    elif score >= 50:
        status = "Needs attention"
    else:
        status = "Poor"


    st.markdown(
        '<div class="section">Your data health</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        [1, 1, 1]
    )

    with col1:

        st.markdown(
            f"""
            <div class="score">

                <div class="score-label">
                    DATA HEALTH
                </div>

                <div class="score-number">
                    {score}%
                </div>

                <div class="score-status">
                    {status}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-label">
                    Issues found
                </div>

                <div class="card-value">
                    {len(issues)}
                </div>

                <div class="card-small">
                    Problems requiring attention
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-label">
                    Records analyzed
                </div>

                <div class="card-value">
                    {summary["rows"]:,}
                </div>

                <div class="card-small">
                    Across {summary["columns"]} fields
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FINDINGS
    # ========================================================

    st.markdown(
        '<div class="section">What we found</div>',
        unsafe_allow_html=True
    )

    if not issues:

        st.markdown(
            """
            <div class="success">

                <b>Your dataset looks healthy.</b><br>

                No significant quality problems were
                identified.

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        for issue in issues:

            st.markdown(
                f"""
                <div class="issue">

                    <div class="issue-title">
                        {issue["title"]}
                    </div>

                    <div class="issue-description">
                        {issue["description"]}
                    </div>

                    <div style="margin-top:8px;">
                        <b>Severity:</b>
                        {issue["severity"]}
                        &nbsp;&nbsp;
                        <b>Confidence:</b>
                        {issue["confidence"]}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # AGENT DECISIONS
    # ========================================================

    st.markdown(
        "<div class='section'>DataGuard's decisions</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="ai-card">

            <div class="ai-title">
                🤖 Recommended actions
            </div>

            <div style="color:#64748b;font-size:14px;">
                DataGuard evaluates each issue separately and
                chooses an appropriate action.
            </div>

        """,
        unsafe_allow_html=True
    )

    for action in plan:

        if action["requires_approval"]:

            label = "Review required"

            label_class = "review"

        else:

            label = "Safe to automate"

            label_class = "safe"


        st.markdown(
            f"""
            <div class="decision">

                <div class="decision-title">
                    {action["decision"]}
                </div>

                <div class="decision-text">
                    {action["reason"]}
                </div>

                <div class="confidence">
                    Confidence: {action["confidence"]}%
                </div>

                <div class="{label_class}">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # APPROVAL FOR OUTLIERS
    # ========================================================

    needs_review = any(
        action["requires_approval"]
        for action in plan
    )


    if needs_review:

        st.markdown(
            '<div class="section">Review required</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="review-box">

                <b>Some unusual records need your approval.</b>

                <p style="color:#92400e;">
                Unusual values are not always errors.
                They could represent legitimate rare cases.
                DataGuard therefore does not remove them
                without your permission.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "✓ Approve recommended cleaning",
            use_container_width=True
        ):

            agent = DataQualityAgent()

            with st.spinner(
                "Applying approved changes and validating the result..."
            ):

                new_result = agent.analyze(
                    st.session_state.df,
                    approved_outliers=True
                )

            st.session_state.result = new_result

            st.rerun()


    # ========================================================
    # EXECUTION
    # ========================================================

    st.markdown(
        '<div class="section">Cleaning activity</div>',
        unsafe_allow_html=True
    )

    execution_log = result[
        "execution_log"
    ]

    if execution_log:

        for item in execution_log:

            st.markdown(
                f"""
                <div class="success">
                    ✓ {item}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <div class="success">
                ✓ No automatic changes were necessary.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    st.markdown(
        '<div class="section">Validation</div>',
        unsafe_allow_html=True
    )

    if validation["status"] == "accepted":

        st.markdown(
            f"""
            <div class="success">

                <b>✓ Changes verified</b><br>

                The cleaned dataset passed the final
                quality check.

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "The proposed changes did not improve the dataset, "
            "so DataGuard restored the original data."
        )


    # ========================================================
    # CLEANED DATA
    # ========================================================

    st.markdown(
        '<div class="section">Cleaned dataset</div>',
        unsafe_allow_html=True
    )

    cleaned_df = result[
        "cleaned_data"
    ]

    st.dataframe(
        cleaned_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv = cleaned_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download cleaned dataset",
        csv,
        "dataguard_cleaned_dataset.csv",
        "text/csv",
        use_container_width=True
    )