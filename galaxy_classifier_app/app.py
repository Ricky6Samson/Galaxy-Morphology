import os
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Galaxy Morphology Classifier", page_icon="🌌", layout="wide"
)

st.markdown(
    """
    <style>
    div[data-testid="stColumn"] img {
        height: 380px !important;
        object-fit: contain !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 2. Main Title and Subtitle
st.title("🌌 Automated Galaxy Morphology Classifier")
st.markdown("""
Welcome to the Galaxy Morphology Analysis Platform. This application leverages deep learning 
to classify astronomical imagery into standard Hubble sequence morphological types.
""")

st.divider()

# 3. High-Level Performance Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Model Accuracy", value="71.98%", delta="+2.39%")
with col2:
    st.metric(label="Precision (Macro)", value="0.70")
with col3:
    st.metric(label="Recall (Macro)", value="0.70")
with col4:
    st.metric(label="Total Test Images", value="3548")

st.divider()

# 4. Galaxy Classes Reference Image
st.subheader("🌌 Galaxy10 DECals Morphological Classes")
IMAGE_PATH = "assets/galaxy10.png"

if os.path.exists(IMAGE_PATH):
    st.image(
        IMAGE_PATH,
        caption="Galaxy10 DECals Dataset: 10 Morphological Classes (0–9)",
        use_container_width=True,
    )
else:
    st.info("Place `galaxy10.png` in your `assets/` folder to render reference classes.")

st.divider()

# 5. Dataset Class Distribution Chart
st.subheader("📊 Dataset Class Distribution")

distribution_data = {
    "Class": list(range(10)),
    "Class Name": [
        "Distorted Galaxy",
        "Merging Galaxies",
        "Round Elliptical",
        "In-between Elliptical",
        "Cigar Elliptical",
        "Barred Spiral",
        "Unbarred Tight Spiral",
        "Unbarred Loose Spiral",
        "Edge-on (No Bulge)",
        "Edge-on (With Bulge)",
    ],
    "Samples": [216, 371, 529, 405, 67, 409, 366, 526, 284, 375],
}

df_dist = pd.DataFrame(distribution_data)

fig = px.bar(
    df_dist,
    x="Class Name",
    y="Samples",
    color="Samples",
    color_continuous_scale="Viridis",
    labels={"Samples": "Image Count", "Class Name": "Morphological Class"},
    text="Samples",
)

fig.update_traces(
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    textposition="outside",
)

fig.update_layout(
    xaxis_tickangle=-45,
    showlegend=False,
    height=450,
    margin=dict(l=20, r=20, t=20, b=100),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note the class imbalance across categories—for instance, Class 2 (Round Elliptical) "
    "and Class 7 (Unbarred Loose Spiral) have over 500 samples each, whereas Class 4 (Cigar Elliptical) "
    "represents a minority class with 67 samples."
)

st.divider()

# 6. Model Diagnostics
st.subheader("🔬 Model Diagnostics & Progression")

diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.markdown("#### Confusion Matrix (Phase 2, Run 2)")
    cm_path = "assets/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(
            cm_path,
            caption="Confusion Matrix across 10 Morphological Classes",
            use_container_width=True,
        )
    else:
        st.info("Place `confusion_matrix.png` in your `assets/` folder to render the heatmap.")

with diag_col2:
    st.markdown("#### Training & Validation History")
    history_path = "assets/training_history.png"
    if os.path.exists(history_path):
        st.image(
            history_path,
            caption="Accuracy and Loss progression across epochs",
            use_container_width=True,
        )
    else:
        st.info("Place `training_history.png` in your `assets/` folder to render the curves.")

st.markdown("""
**Key Diagnostic Insights:**
* **Highest Performing:** Edge-on galaxies (Classes 8 & 9) achieved the strongest classification accuracy (>88% F1-score) due to distinct linear profile features.
* **Primary Confusion:** Class 0 (Distorted) and Class 1 (Merging) showed significant overlap due to shared irregular, low-density tidal features.
""")

st.divider()

# 7. Model Iteration Table
st.subheader("📈 Model Iteration Comparison")

runs_data = {
    "Run / Phase": [
        "Phase 1",
        "Phase 2 - Run 1",
        "Phase 2 - Run 2 (Best)",
        "Phase 3 (Augmented)",
    ],
    "Architecture / Details": [
        "Baseline EfficientNetB0",
        "Top 40 layers unfrozen",
        "Top 40 layers unfrozen run 2",
        "Augmented Images",
    ],
    "Val Accuracy": ["63.36%", "69.59%", "71.98%", "71.65%"],
    "Macro F1-Score": ["0.6062", "0.6759", "0.6979", "0.6910"],
}

df_runs = pd.DataFrame(runs_data)
st.dataframe(df_runs, hide_index=True, use_container_width=True)