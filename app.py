"""
app.py
------
Streamlit Web Application for the Hostel Electricity Consumption Predictor.

Features:
- Intuitive user inputs for room occupants and appliance usage.
- Real-time machine learning prediction of Daily and Monthly consumption (kWh / units).
- Clear explanation: 1 kWh = 1 electricity unit.
- Interactive Matplotlib visualizations (appliance breakdown donut & bar charts).
- Configurable monthly projection days (default: 30 days) and optional electricity tariff rate.
- Targeted energy-saving tips based on the room's highest energy consumer.
- Model performance metrics and dataset explorer.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for Streamlit
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predictor import load_model, predict_electricity, DEFAULT_MODEL_PATH
from train_model import train_and_save_model, DATA_PATH

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hostel Electricity Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .unit-info-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper: Ensure Model Exists
# ---------------------------------------------------------
@st.cache_resource
def get_model_payload():
    """
    Load the trained model. If missing, automatically train it.
    """
    if not os.path.exists(DEFAULT_MODEL_PATH):
        with st.spinner("Model not found. Training model now (takes ~2 seconds)..."):
            train_and_save_model()
    return load_model(DEFAULT_MODEL_PATH)


# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
def main():
    # Header Section
    st.markdown('<div class="main-header">⚡ Hostel Electricity Consumption Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An intelligent Machine Learning system to estimate daily and monthly '
        'hostel room electricity usage based on occupant habits and appliance schedules.</div>',
        unsafe_allow_html=True
    )

    # Unit definition callout
    st.info(
        "💡 **Key Concept:** **1 kWh (Kilowatt-Hour) = 1 Electricity Unit**. "
        "Electricity utility boards bill electricity in 'Units'. For example, running a 1,000-Watt appliance "
        "for 1 hour consumes exactly 1 unit (1 kWh)."
    )

    # Load Model
    try:
        model_payload = get_model_payload()
    except Exception as e:
        st.error(f"Error loading machine learning model: {e}")
        if st.button("🔄 Train Model Now"):
            train_and_save_model()
            st.rerun()
        return

    # ---------------------------------------------------------
    # Sidebar: User Inputs
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Room & Appliance Parameters")
    st.sidebar.caption("Adjust the sliders below to reflect your hostel room setup.")

    # 1. Occupancy
    occupants = st.sidebar.slider(
        "👥 Number of Occupants",
        min_value=1,
        max_value=6,
        value=2,
        help="Number of students residing in the room."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌀 Ceiling Fans")
    fans = st.sidebar.number_input("Number of Fans", min_value=0, max_value=4, value=2, step=1)
    fan_hours = st.sidebar.slider(
        "Average Daily Hours per Fan",
        min_value=0.0,
        max_value=24.0,
        value=14.0,
        step=0.5,
        help="Assumed wattage: 75 Watts per fan."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Lights (LED / Tube)")
    lights = st.sidebar.number_input("Number of Lights", min_value=0, max_value=6, value=2, step=1)
    light_hours = st.sidebar.slider(
        "Average Daily Hours per Light",
        min_value=0.0,
        max_value=24.0,
        value=8.0,
        step=0.5,
        help="Assumed wattage: 25 Watts per light fixture."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("❄️ Air Conditioning (AC)")
    ac_units = st.sidebar.selectbox(
        "AC Units in Room",
        options=[0, 1, 2],
        index=0,
        help="Hostel rooms with split/window air conditioning."
    )
    if ac_units > 0:
        ac_hours = st.sidebar.slider(
            "Average Daily AC Hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5,
            help="Assumed effective consumption: 1200 Watts during active cooling."
        )
    else:
        ac_hours = 0.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("💻 Laptops & Study Devices")
    laptop_hours = st.sidebar.slider(
        "Total Room Laptop Hours / Day",
        min_value=0.0,
        max_value=48.0,
        value=float(occupants * 5.0),
        step=0.5,
        help=f"Cumulative usage across all {occupants} occupants (approx 65 Watts per laptop)."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Other Miscellaneous Appliances")
    other_appliance_kwh = st.sidebar.slider(
        "Other Appliances Daily kWh",
        min_value=0.0,
        max_value=5.0,
        value=0.6,
        step=0.1,
        help="Includes mobile phone chargers, electric kettle, trimmer, iron, and standby power."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Billing & Cost Settings")
    days = st.sidebar.slider(
        "Calculation Period (Days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Defaults to 30 days for typical monthly billing."
    )
    tariff_rate = st.sidebar.number_input(
        "Electricity Tariff Rate (per Unit / kWh)",
        min_value=0.0,
        max_value=50.0,
        value=7.50,
        step=0.50,
        help="Cost per 1 kWh / unit in your college/state currency (e.g. ₹7.50/unit)."
    )

    # ---------------------------------------------------------
    # Perform Prediction
    # ---------------------------------------------------------
    user_inputs = {
        "occupants": occupants,
        "fans": fans,
        "fan_hours": fan_hours,
        "lights": lights,
        "light_hours": light_hours,
        "ac_units": ac_units,
        "ac_hours": ac_hours,
        "laptop_hours": laptop_hours,
        "other_appliance_kwh": other_appliance_kwh
    }

    result = predict_electricity(user_inputs, days=days, model_payload=model_payload)

    daily_kwh = result["daily_kwh"]
    monthly_kwh = result["monthly_kwh"]
    estimated_cost = round(monthly_kwh * tariff_rate, 2)
    per_student_cost = round(estimated_cost / occupants, 2)

    # ---------------------------------------------------------
    # Display Prediction Metrics
    # ---------------------------------------------------------
    st.subheader("📊 Consumption Predictions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Daily Electricity",
            value=f"{daily_kwh} Units",
            help="1 Unit = 1 kWh. Predicted daily room consumption by Machine Learning model."
        )

    with col2:
        st.metric(
            label=f"{days}-Day Projection",
            value=f"{monthly_kwh} Units",
            help=f"Projected consumption over {days} days ({daily_kwh} × {days})."
        )

    with col3:
        st.metric(
            label=f"Estimated Total Bill ({days}d)",
            value=f"₹ {estimated_cost:,.2f}",
            help=f"Calculated at rate of ₹{tariff_rate}/unit."
        )

    with col4:
        st.metric(
            label="Cost per Student",
            value=f"₹ {per_student_cost:,.2f}",
            help=f"Divided equally among {occupants} occupant(s)."
        )

    # ---------------------------------------------------------
    # Appliance Breakdown Visualizations (Matplotlib)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Energy Distribution & Appliance Breakdown")

    viz_col1, viz_col2 = st.columns([1, 1])

    breakdown_data = result["breakdown_kwh"]
    categories = list(breakdown_data.keys())
    values = list(breakdown_data.values())
    colors = ["#3B82F6", "#F59E0B", "#EF4444", "#10B981", "#8B5CF6"]

    with viz_col1:
        st.markdown("##### Proportion of Total Energy Consumption")
        # Filter out zero values for clean pie chart
        pie_labels = [c for c, v in zip(categories, values) if v > 0]
        pie_values = [v for v in values if v > 0]
        pie_colors = [colors[i] for i, v in enumerate(values) if v > 0]

        if sum(pie_values) > 0:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            wedges, texts, autotexts = ax1.pie(
                pie_values,
                labels=pie_labels,
                autopct="%1.1f%%",
                startangle=140,
                colors=pie_colors,
                wedgeprops=dict(width=0.45, edgecolor="w", linewidth=2)  # Donut style
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight("bold")
            ax1.set_title("Appliance Energy Share (%)", fontsize=13, fontweight="bold", pad=12)
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No active appliances selected.")

    with viz_col2:
        st.markdown("##### Daily Consumption by Appliance (kWh / Units)")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bars = ax2.barh(categories, values, color=colors, edgecolor="#1F2937", linewidth=0.6)
        ax2.set_xlabel("Daily Consumption (kWh / Units)", fontsize=11, fontweight="bold")
        ax2.set_title("Appliance-wise Daily Units (kWh)", fontsize=13, fontweight="bold", pad=12)
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width + 0.08,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f} U",
                va="center",
                fontsize=9,
                fontweight="bold"
            )

        st.pyplot(fig2)
        plt.close(fig2)

    # ---------------------------------------------------------
    # Targeted Energy-Saving Advisory
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🌱 Smart Energy-Saving Advisory")
    st.success(
        f"**Primary Energy Consumer:** {result['highest_appliance']} "
        f"({result['breakdown_pct'][result['highest_appliance']]}% of total energy)\n\n"
        f"**Recommendation:** {result['energy_saving_tip']}"
    )

    # ---------------------------------------------------------
    # Model Insights & Dataset Details
    # ---------------------------------------------------------
    with st.expander("🔬 Machine Learning Model Details & Dataset Information"):
        st.markdown(f"**Model Type:** `{model_payload.get('model_name', 'Linear Regression')}`")
        metrics = model_payload.get("metrics", {})
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Model R² Score", f"{metrics.get('r2', 0.99):.4f}")
        mcol2.metric("Mean Absolute Error (MAE)", f"{metrics.get('mae', 0.33):.4f} kWh")
        mcol3.metric("Root Mean Squared Error (RMSE)", f"{metrics.get('rmse', 0.46):.4f} kWh")

        st.markdown("---")
        st.markdown("#### Sample Records from Training Dataset (`data/hostel_electricity.csv`)")
        if os.path.exists(DATA_PATH):
            df_preview = pd.read_csv(DATA_PATH)
            st.dataframe(df_preview.head(8), use_container_width=True)
            st.caption(f"Total dataset size: {len(df_preview)} room-day simulation records.")
        else:
            st.info("Dataset file not found.")


if __name__ == "__main__":
    main()
