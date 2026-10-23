import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    path = Path("models/threat_detection_model.pkl")

    if not path.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    return joblib.load(path)


# Dashboard title
st.title("🛡️ AI Cyber Threat Detection System")

st.markdown(
    "### Machine Learning-Based Network Security Monitoring"
)

st.info(
    "Upload CIC-IDS2017 network traffic data "
    "to detect potential DDoS attacks."
)

# Sidebar
st.sidebar.title("🛡️ Security Control Panel")

st.sidebar.success("AI Detection Dashboard")

st.sidebar.markdown("### System Information")
st.sidebar.write("Detection: Random Forest")
st.sidebar.write("Mode: CSV Analysis")
st.sidebar.write("Platform: Windows 11")

# Load model
try:
    model_data = load_model()

    model = model_data["model"]
    features = model_data["features"]

    st.sidebar.success("AI Model Loaded Successfully")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# CSV Upload
st.header("📂 Upload Network Traffic Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Limit rows for 8GB RAM
        df = pd.read_csv(
            uploaded_file,
            nrows=10000
        )

        df.columns = df.columns.str.strip()

        st.success("Dataset Uploaded Successfully!")

        st.subheader("📋 Dataset Preview")

        st.dataframe(df.head(10))

        st.write("Total Records Loaded:", len(df))

        # Prepare features
        X = df.drop(
            columns=["Label"],
            errors="ignore"
        )

        X = X.select_dtypes(
            include=[np.number]
        )

        # Match training features
        missing_features = [
            col for col in features
            if col not in X.columns
        ]

        if missing_features:
            st.error(
                "Dataset is missing required features: "
                + ", ".join(missing_features[:10])
            )
            st.stop()

        X = X.reindex(columns=features)

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0)

        # Run AI prediction
        st.header("🤖 AI Threat Detection")

        if st.button("🚀 Start Threat Detection"):

            with st.spinner("Analyzing network traffic..."):

                predictions = model.predict(X)

                results = df.copy()

                results["AI_Prediction"] = predictions

                # Statistics
                total = len(results)

                normal = (
                    results["AI_Prediction"] == "BENIGN"
                ).sum()

                attacks = (
                    results["AI_Prediction"] == "DDoS"
                ).sum()

                attack_rate = (
                    attacks / total * 100
                    if total > 0 else 0
                )

                # Dashboard metrics
                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Traffic",
                    total
                )

                col2.metric(
                    "Normal Traffic",
                    normal
                )

                col3.metric(
                    "DDoS Alerts",
                    attacks
                )

                col4.metric(
                    "Attack Rate",
                    f"{attack_rate:.2f}%"
                )

                # Security alert
                if attacks > 0:

                    st.error(
                        f"🚨 ALERT: {attacks} traffic records "
                        "classified as potential DDoS!"
                    )

                else:

                    st.success(
                        "✅ No DDoS traffic detected "
                        "in this sample."
                    )

                # Pie chart
                st.subheader("📊 Traffic Classification")

                counts = (
                    results["AI_Prediction"]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "Traffic Type",
                    "Count"
                ]

                fig = px.pie(
                    counts,
                    names="Traffic Type",
                    values="Count",
                    title="Normal vs DDoS Traffic"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Bar chart
                st.subheader("📈 Threat Statistics")

                fig2 = px.bar(
                    counts,
                    x="Traffic Type",
                    y="Count",
                    title="Traffic Detection Results"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

                # Results table
                st.subheader("🔍 Detailed Detection Results")

                st.dataframe(
                    results[
                        ["AI_Prediction"]
                    ].head(100)
                )

                # Download report
                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Security Report",
                    data=csv,
                    file_name="security_report.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"Error processing dataset: {e}")

else:

    st.info(
        "👆 Upload a network traffic CSV file "
        "to begin AI threat detection."
    )

# Footer
st.divider()

st.caption(
    "AI Cyber Threat Detection System | "
    "Educational Cybersecurity Project"
)