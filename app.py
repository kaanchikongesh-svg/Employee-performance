"""
Employee Performance Prediction & HR Analytics Platform.
Full-featured Streamlit application for individual prediction, batch inference,
dataset exploration, model benchmarking, and explainable AI insights.
"""

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="HR Insight AI | Employee Performance Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Relative Path Setup
BASE_DIR = Path(__file__).resolve().parent
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    THRESHOLD_HIGH,
    THRESHOLD_MEDIUM,
    CATEGORY_HIGH,
    CATEGORY_MEDIUM,
    CATEGORY_LOW,
    CATEGORIES,
    CATEGORY_COLORS,
    GENDER_OPTIONS,
    DEPARTMENT_OPTIONS,
    JOB_ROLE_OPTIONS,
    ALL_JOB_ROLES,
    PROMOTION_OPTIONS,
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    SAMPLE_DATA_PATH,
    MODEL_FILE_PATH,
    score_to_category,
)
from src.preprocessing import validate_dataframe, clean_and_prepare_data
from src.train_model import (
    ensure_sample_dataset,
    train_and_compare_models,
    save_model_package,
    load_model_package
)
from src.prediction import predict_single_employee, predict_batch_employees
from src.visualization import (
    plot_performance_distribution,
    plot_department_performance,
    plot_scatter_correlation,
    plot_satisfaction_box,
    plot_feature_importance,
    plot_confusion_matrix_heatmap,
    plot_model_comparison,
    plot_probability_breakdown
)

# Custom Styling (Glassmorphism & Clean Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        padding: 0;
    }
    
    .main-header p {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .metric-card .title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .badge-high {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #A7F3D0;
    }
    
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #FECACA;
    }
    
    .impact-pill-pos {
        background-color: #ECFDF5;
        color: #047857;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
    }
    
    .impact-pill-neg {
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #FECACA;
    }
    
    .card-container {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_or_initialize_model():
    """Loads saved model package or trains on sample dataset if missing."""
    package = load_model_package()
    if package is None:
        sample_df = ensure_sample_dataset()
        package = train_and_compare_models(sample_df)
        save_model_package(package)
    return package


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Loads default sample employee dataset."""
    return ensure_sample_dataset()


# Initialize state
if "model_package" not in st.session_state:
    with st.spinner("Initializing ML Performance Models..."):
        st.session_state["model_package"] = get_or_initialize_model()

if "current_dataset" not in st.session_state:
    st.session_state["current_dataset"] = load_default_data()


# Top Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>HR Insight AI • Employee Performance Prediction</h1>
            <p>Enterprise Machine Learning Platform for Talent Analytics, Benchmarking & Performance Forecasting</p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(59, 130, 246, 0.25); color: #93C5FD; border: 1px solid #3B82F6; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                Production ML v2.0
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("⚙️ Control Panel")
    
    st.subheader("Model Status")
    active_model_name = st.session_state["model_package"].get("best_model_name", "Random Forest")
    st.success(f"**Active Model:** {active_model_name}")
    
    st.markdown("---")
    st.subheader("Classification Thresholds")
    st.markdown(f"""
    - 🟢 **High:** `{THRESHOLD_HIGH} - 100.0`
    - 🟡 **Medium:** `{THRESHOLD_MEDIUM} - {THRESHOLD_HIGH - 0.1}`
    - 🔴 **Low:** `0.0 - {THRESHOLD_MEDIUM - 0.1}`
    """)

    st.markdown("---")
    st.subheader("Current Dataset")
    st.info(f"**Loaded Records:** {len(st.session_state['current_dataset']):,} employees")
    
    # Reset to default button
    if st.button("🔄 Reset to Synthetic Dataset", use_container_width=True):
        st.session_state["current_dataset"] = load_default_data()
        st.session_state["model_package"] = get_or_initialize_model()
        st.success("Reset successfully!")
        st.rerun()

    st.markdown("---")
    st.caption("Built for HR Executives, People Analytics Teams & Viva Presentations.")


# Navigation Tabs
tab_home, tab_single, tab_csv_train, tab_batch, tab_dashboard, tab_model_insights = st.tabs([
    "🏠 Overview",
    "👤 Individual Prediction",
    "📁 Dataset & Model Training",
    "⚡ Batch Prediction",
    "📊 HR Analytics Dashboard",
    "🔍 Explainable AI & Metrics"
])


# ==============================================================================
# TAB 1: OVERVIEW / HOME
# ==============================================================================
with tab_home:
    col1, col2, col3, col4 = st.columns(4)
    df_curr = st.session_state["current_dataset"]
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Employees</div>
            <div class="value">{len(df_curr):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_score = df_curr["Performance_Score"].mean() if "Performance_Score" in df_curr.columns else 76.4
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Avg Performance Score</div>
            <div class="value">{avg_score:.1f} <span style="font-size: 1rem; color: #10B981;">/ 100</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        high_cnt = (df_curr["Performance_Category"] == CATEGORY_HIGH).sum() if "Performance_Category" in df_curr.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Top Performers</div>
            <div class="value" style="color: #10B981;">{high_cnt:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_acc = st.session_state["model_package"]["metrics"][active_model_name]["accuracy"] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Model Accuracy</div>
            <div class="value" style="color: #2563EB;">{best_acc:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.subheader("🎯 Key Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card-container">
            <h4>🧠 Multi-Model Machine Learning</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Trained and evaluated on Random Forest, Gradient Boosting, and Logistic Regression with automated hyperparameter optimization.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        c2.markdown("""
        <div class="card-container">
            <h4>💡 Explainable AI (XAI)</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Transparency factors decompose predictions into positive and negative drivers (attendance, training, overtime, experience).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        c3.markdown("""
        <div class="card-container">
            <h4>📈 Interactive HR Dashboards</h4>
            <p style="color: #64748B; font-size: 0.9rem;">
                Real-time visual intelligence across departments, compensation brackets, work-life balance, and career trajectories.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("🚀 How to Use this Application")
    st.markdown("""
    1. **👤 Individual Prediction:** Enter employee profile parameters to instantly compute predicted performance category, score, and factor explanations.
    2. **📁 Dataset & Training:** Upload custom organizational HR CSV files, explore distributions, and retrain/compare ML algorithms.
    3. **⚡ Batch Prediction:** Bulk-predict thousands of employee records simultaneously and download classified CSV results.
    4. **📊 Analytics Dashboard:** Filter and visualize company-wide performance trends with interactive charts.
    5. **🔍 Explainable AI & Metrics:** Inspect feature importances, confusion matrices, and precision/recall tradeoffs.
    """)


# ==============================================================================
# TAB 2: INDIVIDUAL PREDICTION
# ==============================================================================
with tab_single:
    st.subheader("👤 Individual Employee Performance Forecasting")
    st.caption("Fill in employee attributes to generate real-time machine learning predictions.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("##### 🏢 Organizational Profile")
        emp_id = st.text_input("Employee ID", value="EMP-2045")
        dept = st.selectbox("Department", options=DEPARTMENT_OPTIONS, index=0)
        role_options = JOB_ROLE_OPTIONS.get(dept, ALL_JOB_ROLES)
        job_role = st.selectbox("Job Role", options=role_options, index=0)
        job_level = st.selectbox("Job Level", options=[1, 2, 3, 4, 5], index=2)
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000.0, max_value=35000.0, value=7500.0, step=250.0)
        team_size = st.number_input("Team Size", min_value=1, max_value=50, value=8, step=1)

    with col_b:
        st.markdown("##### 👤 Demographics & Experience")
        age = st.number_input("Age", min_value=18, max_value=65, value=33, step=1)
        gender = st.selectbox("Gender", options=GENDER_OPTIONS, index=0)
        experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=7.0, step=0.5)
        promotion_history = st.selectbox("Promoted in Company?", options=PROMOTION_OPTIONS, index=1)
        years_since_promo = st.number_input("Years Since Last Promotion", min_value=0.0, max_value=20.0, value=1.5, step=0.5)
        work_life_balance = st.slider("Work-Life Balance Rating", min_value=1, max_value=5, value=4)

    with col_c:
        st.markdown("##### ⚡ Productivity & Engagement")
        prev_perf = st.slider("Previous Performance Score", min_value=30.0, max_value=100.0, value=82.0, step=0.5)
        attendance = st.slider("Attendance Percentage (%)", min_value=50.0, max_value=100.0, value=96.0, step=0.5)
        training_hours = st.number_input("Training Hours (Annual)", min_value=0.0, max_value=200.0, value=45.0, step=2.0)
        num_trainings = st.number_input("Number of Trainings Attended", min_value=0, max_value=15, value=4, step=1)
        projects_completed = st.number_input("Projects Completed", min_value=0, max_value=40, value=9, step=1)
        working_hours = st.number_input("Working Hours / Week", min_value=20.0, max_value=80.0, value=42.0, step=1.0)
        overtime_hours = st.number_input("Overtime Hours / Month", min_value=0.0, max_value=80.0, value=8.0, step=1.0)
        job_satisfaction = st.slider("Job Satisfaction Rating", min_value=1, max_value=5, value=4)

    submit_btn = st.button("🔮 Predict Employee Performance", use_container_width=True, type="primary")

    if submit_btn:
        employee_payload = {
            "Employee_ID": emp_id,
            "Age": age,
            "Gender": gender,
            "Department": dept,
            "Job_Role": job_role,
            "Years_of_Experience": experience,
            "Monthly_Income": monthly_income,
            "Job_Level": job_level,
            "Job_Satisfaction": job_satisfaction,
            "Work_Life_Balance": work_life_balance,
            "Working_Hours_per_Week": working_hours,
            "Attendance_Percentage": attendance,
            "Training_Hours": training_hours,
            "Number_of_Trainings": num_trainings,
            "Projects_Completed": projects_completed,
            "Overtime_Hours": overtime_hours,
            "Previous_Performance_Score": prev_perf,
            "Promotion_History": promotion_history,
            "Years_Since_Last_Promotion": years_since_promo,
            "Team_Size": team_size
        }

        with st.spinner("Executing model inference..."):
            result = predict_single_employee(employee_payload, st.session_state["model_package"])

        st.write("")
        st.subheader("📋 Prediction Results")

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        cat = result["category"]
        badge_class = f"badge-{cat.lower()}"

        with res_col1:
            st.markdown(f"""
            <div class="card-container" style="text-align: center; padding: 1.8rem 1rem;">
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Predicted Performance Category</div>
                <div style="margin: 0.8rem 0;">
                    <span class="{badge_class}" style="font-size: 1.4rem; padding: 8px 24px;">{cat.upper()} PERFORMER</span>
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #0F172A;">
                    {result['score']} <span style="font-size: 1.1rem; color: #64748B;">/ 100</span>
                </div>
                <div style="font-size: 0.9rem; color: #64748B; margin-top: 0.5rem;">
                    Model Confidence: <strong>{result['confidence']}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.plotly_chart(plot_probability_breakdown(result["probabilities"]), use_container_width=True)

        with res_col3:
            st.markdown("""
            <div class="card-container">
                <h5 style="margin-top: 0;">💡 Recommendation</h5>
            """, unsafe_allow_html=True)
            if cat == CATEGORY_HIGH:
                st.success("🌟 **High Potential Talent:** Suitable for fast-track leadership, mentoring initiatives, and strategic project ownership.")
            elif cat == CATEGORY_MEDIUM:
                st.warning("⚖️ **Solid Contributor:** Performance can be elevated through targeted skill training and optimizing workload balance.")
            else:
                st.error("⚠️ **Performance Support Required:** Recommend 1-on-1 coaching, root-cause attendance review, and workload realignment.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.subheader("🔍 Explainable AI: Why this prediction?")
        st.caption("Top factors influencing this employee's predicted performance score:")

        factor_cols = st.columns(3)
        for idx, factor in enumerate(result["factors"]):
            with factor_cols[idx % 3]:
                pill_class = "impact-pill-pos" if factor["is_positive"] else "impact-pill-neg"
                st.markdown(f"""
                <div class="card-container" style="margin-bottom: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <strong>{factor['feature']}</strong>
                        <span class="{pill_class}">{factor['impact']}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B;">{factor['description']}</div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: DATASET & MODEL TRAINING
# ==============================================================================
with tab_csv_train:
    st.subheader("📁 Dataset Explorer & Model Training Benchmarks")
    st.caption("Upload your custom HR dataset in CSV format, analyze distributions, and retrain ML pipelines.")

    uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type=["csv"], help="Must contain employee feature columns.")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            is_valid, validation_msgs = validate_dataframe(df_upload, require_target=False)

            if is_valid:
                st.success(f"✅ Successfully loaded `{uploaded_file.name}` with {len(df_upload):,} rows and {len(df_upload.columns)} columns.")
                st.session_state["current_dataset"] = df_upload
            else:
                st.warning(f"Uploaded CSV had schema warnings: {'; '.join(validation_msgs)}")
                st.session_state["current_dataset"] = df_upload
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    df_view = st.session_state["current_dataset"]

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Dataset Rows", f"{len(df_view):,}")
    col_meta2.metric("Total Columns", f"{len(df_view.columns)}")
    col_meta3.metric("Missing Values", f"{df_view.isnull().sum().sum():,}")
    col_meta4.metric("Numeric Features", f"{len(df_view.select_dtypes(include=[np.number]).columns)}")

    with st.expander("🔍 Preview Raw Dataset (First 15 Rows)", expanded=True):
        st.dataframe(df_view.head(15), use_container_width=True)

    with st.expander("📊 Statistical Summary"):
        st.dataframe(df_view.describe().round(2), use_container_width=True)

    st.write("")
    st.subheader("🏋️ Retrain & Compare Machine Learning Algorithms")
    st.markdown("Train and benchmark **Random Forest**, **Logistic Regression**, and **Gradient Boosting** models.")

    train_col1, train_col2 = st.columns([1, 2])
    with train_col1:
        test_split = st.slider("Test Split Size", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
        train_btn = st.button("🚀 Train & Benchmark Models", type="primary", use_container_width=True)

    if train_btn:
        with st.spinner("Training Random Forest, Logistic Regression, and Gradient Boosting models..."):
            try:
                new_model_package = train_and_compare_models(df_view, test_size=test_split)
                save_model_package(new_model_package)
                st.session_state["model_package"] = new_model_package
                st.success(f"🎉 Training complete! Best performing algorithm: **{new_model_package['best_model_name']}**")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Display benchmark comparison
    metrics = st.session_state["model_package"]["metrics"]
    st.write("")
    st.markdown("#### 🏆 Algorithm Benchmark Performance")

    comp_rows = []
    for model_name, m in metrics.items():
        is_best = (model_name == st.session_state["model_package"]["best_model_name"])
        comp_rows.append({
            "Model": f"{'⭐ ' if is_best else ''}{model_name}",
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision (Weighted)": f"{m['precision'] * 100:.2f}%",
            "Recall (Weighted)": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1_score'] * 100:.2f}%",
            "Status": "Active Selected Model" if is_best else "Benchmarked"
        })

    st.table(pd.DataFrame(comp_rows))

    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        st.plotly_chart(plot_model_comparison(metrics), use_container_width=True)

    with col_bench2:
        best_name = st.session_state["model_package"]["best_model_name"]
        cm_best = metrics[best_name]["confusion_matrix"]
        st.plotly_chart(
            plot_confusion_matrix_heatmap(
                cm_best,
                model_name=best_name
            ),
            use_container_width=True,
            key=f"benchmark_cm_{best_name}"
        )


# ==============================================================================
# TAB 4: BATCH PREDICTION
# ==============================================================================
with tab_batch:
    st.subheader("⚡ Batch Employee Performance Prediction")
    st.caption("Upload a batch employee CSV to classify multiple employees at once and export predictions.")

    batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_uploader")

    batch_df_to_use = None
    if batch_file is not None:
        try:
            batch_df_to_use = pd.read_csv(batch_file)
            st.success(f"Loaded {len(batch_df_to_use):,} employees from `{batch_file.name}`")
        except Exception as e:
            st.error(f"Error loading batch file: {e}")
    else:
        st.info("💡 No file uploaded yet. You can run batch prediction on the active demo dataset below:")
        batch_df_to_use = st.session_state["current_dataset"]

    if batch_df_to_use is not None and not batch_df_to_use.empty:
        if st.button("⚡ Run Batch Prediction Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing batch inference..."):
                pred_df, batch_stats = predict_batch_employees(batch_df_to_use, st.session_state["model_package"])

            st.write("")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Processed", f"{batch_stats['total_employees']:,}")
            b2.metric("Avg Predicted Score", f"{batch_stats['avg_predicted_score']:.1f}")
            b3.metric("High Performers", f"{batch_stats['high_count']:,} ({batch_stats['high_count']/batch_stats['total_employees']*100:.1f}%)")
            b4.metric("Avg Model Confidence", f"{batch_stats['avg_confidence']:.1f}%")

            st.write("")
            st.markdown("#### 📋 Batch Prediction Results")
            st.dataframe(pred_df, use_container_width=True)

            # Download CSV
            csv_buffer = io.StringIO()
            pred_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                label="📥 Download employee_predictions.csv",
                data=csv_data,
                file_name="employee_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )


# ==============================================================================
# TAB 5: ANALYTICS DASHBOARD
# ==============================================================================
with tab_dashboard:
    st.subheader("📊 People Analytics & Performance Dashboard")
    st.caption("Strategic workforce intelligence and multidimensional correlation analysis.")

    df_dash = st.session_state["current_dataset"].copy()
    if "Performance_Score" not in df_dash.columns or "Performance_Category" not in df_dash.columns:
        # Generate predictions to power dashboard
        try:
            df_dash, _ = predict_batch_employees(df_dash, st.session_state["model_package"])
            df_dash["Performance_Score"] = df_dash["Predicted_Score"]
            df_dash["Performance_Category"] = df_dash["Predicted_Category"]
        except Exception as err:
            st.error(f"Could not generate dashboard predictions for current dataset: {err}")

    # Ensure Department column exists
    if "Department" not in df_dash.columns:
        df_dash["Department"] = "General"

    dept_options = sorted([str(d) for d in df_dash["Department"].dropna().unique()])
    selected_depts = st.multiselect("Filter by Department", options=dept_options, default=dept_options)
    selected_cats = st.multiselect("Filter by Performance Tier", options=CATEGORIES, default=CATEGORIES)

    filtered_df = df_dash[
        (df_dash["Department"].isin(selected_depts)) &
        (df_dash["Performance_Category"].isin(selected_cats))
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
    else:
        # Row 1: KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Filtered Employees", f"{len(filtered_df):,}")
        avg_s = filtered_df['Performance_Score'].mean() if "Performance_Score" in filtered_df.columns else 0.0
        kpi2.metric("Mean Score", f"{avg_s:.1f}")
        kpi3.metric("Avg Attendance", f"{filtered_df['Attendance_Percentage'].mean():.1f}%" if "Attendance_Percentage" in filtered_df.columns else "N/A")
        kpi4.metric("Avg Training Hours", f"{filtered_df['Training_Hours'].mean():.1f} hrs" if "Training_Hours" in filtered_df.columns else "N/A")

        st.write("")
        # Row 2: Charts
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if "Performance_Score" in filtered_df.columns:
                st.plotly_chart(plot_performance_distribution(filtered_df), use_container_width=True)
        with row2_col2:
            if "Department" in filtered_df.columns and "Performance_Score" in filtered_df.columns:
                st.plotly_chart(plot_department_performance(filtered_df), use_container_width=True)

        # Row 3: Correlations
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            if "Years_of_Experience" in filtered_df.columns and "Performance_Score" in filtered_df.columns:
                st.plotly_chart(plot_scatter_correlation(filtered_df, "Years_of_Experience", title="<b>Experience vs Performance Score</b>"), use_container_width=True)
            else:
                st.info("Experience vs Performance chart requires 'Years_of_Experience' column.")
        with row3_col2:
            if "Attendance_Percentage" in filtered_df.columns and "Performance_Score" in filtered_df.columns:
                st.plotly_chart(plot_scatter_correlation(filtered_df, "Attendance_Percentage", title="<b>Attendance Rate vs Performance Score</b>"), use_container_width=True)
            else:
                st.info("Attendance vs Performance chart requires 'Attendance_Percentage' column.")

        # Row 4: Training & Satisfaction
        row4_col1, row4_col2 = st.columns(2)
        with row4_col1:
            if "Training_Hours" in filtered_df.columns and "Performance_Score" in filtered_df.columns:
                st.plotly_chart(plot_scatter_correlation(filtered_df, "Training_Hours", title="<b>Training Hours vs Performance Score</b>"), use_container_width=True)
            else:
                st.info("Training vs Performance chart requires 'Training_Hours' column.")
        with row4_col2:
            if "Job_Satisfaction" in filtered_df.columns and "Performance_Score" in filtered_df.columns:
                st.plotly_chart(plot_satisfaction_box(filtered_df), use_container_width=True)
            else:
                st.info("Job Satisfaction chart requires 'Job_Satisfaction' column.")

        # Row 5: Salary Analysis
        if "Monthly_Income" in filtered_df.columns and "Performance_Score" in filtered_df.columns:
            st.plotly_chart(plot_scatter_correlation(filtered_df, "Monthly_Income", title="<b>Monthly Compensation vs Performance Score</b>", x_label="Monthly Income ($)"), use_container_width=True)
        else:
            st.info("Compensation chart requires 'Monthly_Income' column.")


# ==============================================================================
# TAB 6: EXPLAINABLE AI & METRICS
# ==============================================================================
with tab_model_insights:
    st.subheader("🔍 Explainable AI & Global Feature Importance")
    st.caption("Understand what factors drive organizational performance according to the trained ML model.")

    feat_imp = st.session_state["model_package"].get("feature_importances", {})

    col_imp1, col_imp2 = st.columns([1.4, 1])
    with col_imp1:
        st.plotly_chart(plot_feature_importance(feat_imp, top_n=15), use_container_width=True)

    with col_imp2:
        st.markdown("#### 📌 Key Influencing Drivers")
        st.markdown("""
        1. **Previous Performance Score:** The strongest historical predictor of sustained execution and competence.
        2. **Attendance Percentage:** Reliable presence strongly correlates with project completion and team consistency.
        3. **Training Hours:** Proactive upskilling translates to measurable performance velocity.
        4. **Job Satisfaction & Work-Life Balance:** Moderate to high satisfaction prevents burnout and stabilizes output.
        5. **Controlled Overtime:** Moderate overtime supports delivery; excessive overtime indicates operational strain.
        """)

    st.write("")
    st.markdown("#### 🔬 Detailed Confusion Matrix Breakdown")
    cm_cols = st.columns(3)
    all_metrics = st.session_state["model_package"]["metrics"]
    for idx, (m_name, m_data) in enumerate(all_metrics.items()):
        with cm_cols[idx]:
            st.plotly_chart(plot_confusion_matrix_heatmap(m_data["confusion_matrix"], model_name=m_name), use_container_width=True, key=f"insights_cm_{m_name}")
