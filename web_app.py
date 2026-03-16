import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px  # Needed for Pie and Scatter plots

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SmartAdFlow | AI Analytics", layout="wide", page_icon="🚀")

# --- 2. PATH LOGIC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..") if os.path.basename(BASE_DIR) == "scripts" else BASE_DIR

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'budget_model.pkl')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'models', 'channel_encoder.pkl')

# --- 3. SIDEBAR (Logo & Inputs) ---
# Locate logo.png
possible_paths = [
    os.path.join(PROJECT_ROOT, "logo.png"),
    os.path.join(BASE_DIR, "logo.png"),
    "logo.png"
]

logo_path = None
for p in possible_paths:
    if os.path.exists(p):
        logo_path = p
        break

if logo_path:
    # EDITABLE SIZE: Change 'width=200' to any number (e.g., 150, 250) to resize
    st.sidebar.image(logo_path, width=200) 
else:
    st.sidebar.error("⚠️ logo.png not found!")

st.sidebar.header("Campaign Settings")
total_budget = st.sidebar.number_input("Total Daily Budget (₹)", min_value=1000, value=50000, step=1000)
day_type = st.sidebar.selectbox("Select Day Type", ["Weekday", "Weekend"])
is_weekend = 1 if day_type == "Weekend" else 0

st.sidebar.markdown("---")
if os.path.exists(MODEL_PATH):
    st.sidebar.success("✅ AI Model: Ready")
else:
    st.sidebar.error("❌ AI Model: Not Found")
# CHART SIZE CONTROLS (Editable in Sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart Settings")
chart_height = st.sidebar.slider("Chart Height", 300, 800, 450)

# --- 4. MAIN UI ---
st.title("🚀 SmartAdFlow: Automated Marketing Budget Allocation")
st.markdown("*Optimize your Ads, Amplify your Growth...*")
st.divider()

if st.sidebar.button("Optimize Allocation"):
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        
        channels = le.classes_
        base_share = total_budget / len(channels)
        results = []
        
        for channel in channels:
            channel_idx = le.transform([channel])[0]
            input_data = pd.DataFrame([[channel_idx, base_share, is_weekend]], 
                                      columns=['channel_enc', 'spend', 'is_weekend'])
            prediction = model.predict(input_data)[0]
            results.append({
                "Channel": channel,
                "Budget (₹)": round(base_share, 2),
                "Revenue (₹)": round(max(0, prediction), 2),
                "ROI": round(prediction / base_share, 4) if base_share > 0 else 0
            })
        
        res_df = pd.DataFrame(results).sort_values(by="ROI", ascending=False)
        
        # --- METRICS ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Predicted Revenue", f"₹{res_df['Revenue (₹)'].sum():,.2f}")
        m2.metric("Average ROI", f"{res_df['ROI'].mean():.2f}x")
        m3.metric("Top Channel", res_df.iloc[0]['Channel'])

        # --- CHARTS SECTION ---
        tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📉 Performance Charts", "🎯 Strategic Insights"])

        with tab1:
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Budget Share (Pie Chart)")
                fig_pie = px.pie(res_df, values='Budget (₹)', names='Channel', hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True, theme="streamlit")

            with col2:
                st.subheader("Revenue Trend (Line Chart)")
                st.line_chart(res_df.set_index('Channel')['Revenue (₹)'], height=chart_height)

        with tab3:
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("ROI vs Revenue (Scatter Plot)")
                fig_scatter = px.scatter(res_df, x="ROI", y="Revenue (₹)", 
                                         size="Budget (₹)", color="Channel",
                                         hover_name="Channel", size_max=60)
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col4:
                st.subheader("Channel Efficiency (Bar)")
                st.bar_chart(res_df.set_index('Channel')['ROI'], height=chart_height)

        st.success(f"Strategy successfully generated for a {day_type}!")
    else:
        st.error("Model files not found! Run 'python scripts/train.py' first.")
else:
    st.info("👈 Enter budget details in the sidebar and click 'Optimize Allocation' to visualize insights.")