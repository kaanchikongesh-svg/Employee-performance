"""
Visualization Module for HR Analytics Dashboard and Model Insights.
Provides modern, interactive Plotly charts with custom color palettes and responsive layouts.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    CATEGORY_COLORS,
    CATEGORY_HIGH,
    CATEGORY_MEDIUM,
    CATEGORY_LOW,
    CATEGORIES,
)

# Custom Theme Styling
THEME_TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#3B82F6"      # Blue
COLOR_ACCENT = "#8B5CF6"       # Purple
COLOR_BACKGROUND = "#FFFFFF"
FONT_FAMILY = "Inter, sans-serif"


def plot_performance_distribution(df: pd.DataFrame, score_col: str = "Performance_Score") -> go.Figure:
    """Histogram and density curve of employee performance scores."""
    fig = px.histogram(
        df,
        x=score_col,
        nbins=25,
        color="Performance_Category" if "Performance_Category" in df.columns else None,
        color_discrete_map=CATEGORY_COLORS,
        marginal="box",
        title="<b>Employee Performance Score Distribution</b>",
        labels={score_col: "Performance Score (0 - 100)"},
        template=THEME_TEMPLATE,
    )
    fig.update_layout(
        font_family=FONT_FAMILY,
        legend_title_text="Category",
        bargap=0.08,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_department_performance(df: pd.DataFrame, score_col: str = "Performance_Score") -> go.Figure:
    """Department-wise average performance and category distribution."""
    dept_stats = (
        df.groupby("Department")[score_col]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )

    fig = px.bar(
        dept_stats,
        x="Department",
        y="mean",
        color="mean",
        color_continuous_scale="Blues",
        text=dept_stats["mean"].round(1),
        title="<b>Average Performance by Department</b>",
        labels={"mean": "Average Score", "Department": "Department"},
        template=THEME_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        font_family=FONT_FAMILY,
        height=400,
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_scatter_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "Performance_Score",
    title: Optional[str] = None,
    x_label: Optional[str] = None
) -> go.Figure:
    """Generic scatter plot with trendline showing relationships with performance."""
    display_title = title or f"<b>{x_col.replace('_', ' ')} vs Performance</b>"
    try:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Performance_Category" if "Performance_Category" in df.columns else None,
            color_discrete_map=CATEGORY_COLORS,
            trendline="ols",
            opacity=0.75,
            title=display_title,
            labels={x_col: x_label or x_col.replace('_', ' '), y_col: "Performance Score"},
            template=THEME_TEMPLATE,
        )
    except Exception:
        # Graceful fallback without trendline if statsmodels encounters singular matrix or is unavailable
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color="Performance_Category" if "Performance_Category" in df.columns else None,
            color_discrete_map=CATEGORY_COLORS,
            opacity=0.75,
            title=display_title,
            labels={x_col: x_label or x_col.replace('_', ' '), y_col: "Performance Score"},
            template=THEME_TEMPLATE,
        )
    fig.update_layout(
        font_family=FONT_FAMILY,
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_satisfaction_box(df: pd.DataFrame, score_col: str = "Performance_Score") -> go.Figure:
    """Box plot showing performance across job satisfaction ratings."""
    fig = px.box(
        df,
        x="Job_Satisfaction",
        y=score_col,
        color="Job_Satisfaction",
        title="<b>Job Satisfaction Rating vs Performance Score</b>",
        labels={"Job_Satisfaction": "Job Satisfaction Rating (1 - 5)", score_col: "Performance Score"},
        template=THEME_TEMPLATE,
    )
    fig.update_layout(
        font_family=FONT_FAMILY,
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 12) -> go.Figure:
    """Horizontal bar chart showing top global model feature importances."""
    if not importance_dict:
        fig = go.Figure()
        fig.update_layout(title="No feature importance data available")
        return fig

    items = list(importance_dict.items())[:top_n]
    features = [item[0].replace("_", " ") for item in items][::-1]
    scores = [item[1] for item in items][::-1]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale="Viridis",
                showscale=False,
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="<b>Top Model Feature Importances</b>",
        xaxis_title="Relative Importance Weight",
        yaxis_title="Feature",
        template=THEME_TEMPLATE,
        font_family=FONT_FAMILY,
        height=450,
        margin=dict(l=30, r=20, t=50, b=20),
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: List[str] = CATEGORIES,
    model_name: str = "Model"
) -> go.Figure:
    """Interactive heatmap for classification confusion matrix."""
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title=f"<b>Confusion Matrix - {model_name}</b>",
        labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
        template=THEME_TEMPLATE,
    )
    fig.update_layout(
        font_family=FONT_FAMILY,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, and F1 across trained models."""
    rows = []
    for model_name, metrics in metrics_dict.items():
        rows.append({"Model": model_name, "Metric": "Accuracy", "Score": metrics.get("accuracy", 0)})
        rows.append({"Model": model_name, "Metric": "Precision", "Score": metrics.get("precision", 0)})
        rows.append({"Model": model_name, "Metric": "Recall", "Score": metrics.get("recall", 0)})
        rows.append({"Model": model_name, "Metric": "F1 Score", "Score": metrics.get("f1_score", 0)})

    comp_df = pd.DataFrame(rows)
    comp_df["Score_Pct"] = (comp_df["Score"] * 100).round(1)

    fig = px.bar(
        comp_df,
        x="Model",
        y="Score_Pct",
        color="Metric",
        barmode="group",
        text="Score_Pct",
        title="<b>ML Algorithm Benchmark Comparison</b>",
        labels={"Score_Pct": "Score (%)", "Model": "Machine Learning Model"},
        template=THEME_TEMPLATE,
        color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        font_family=FONT_FAMILY,
        yaxis_range=[0, 115],
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_probability_breakdown(probabilities: Dict[str, float]) -> go.Figure:
    """Donut chart for prediction class probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [CATEGORY_COLORS.get(cat, "#3B82F6") for cat in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hoverinfo="label+value",
            )
        ]
    )
    fig.update_layout(
        title="<b>Class Probability Confidence</b>",
        showlegend=False,
        template=THEME_TEMPLATE,
        font_family=FONT_FAMILY,
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
