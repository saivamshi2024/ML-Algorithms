import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Set page config
st.set_page_config(page_title="Customer Segmentation (Hierarchical)", layout="centered")

st.title("Customer Segmentation with Hierarchical Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Feature selection
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    x = data[features].values

    # Scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Dendrogram
    st.subheader("Dendrogram")
    fig1, ax1 = plt.subplots()
    dendrogram = sch.dendrogram(sch.linkage(x_scaled, method='ward'), ax=ax1)
    ax1.set_title("Dendrogram")
    ax1.set_xlabel("Customers")
    ax1.set_ylabel("Euclidean distances")
    st.pyplot(fig1)

    # Cluster count selection
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)

    # Fit Agglomerative Clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    y_ac = model.fit_predict(x_scaled)
    data['Cluster'] = y_ac

    # Show clustered data
    st.subheader("Clustered Dataset")
    st.write(data)

    # Visualization
    st.subheader("Cluster Visualization")
    fig2, ax2 = plt.subplots()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(n_clusters):
        cluster_points = x_scaled[y_ac == i]
        if cluster_points.size > 0:
            ax2.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                s=100, color=colors[i % len(colors)], label=f'Cluster {i+1}'
            )

    ax2.set_title("Customer Clusters")
    ax2.set_xlabel("Annual Income (scaled)")
    ax2.set_ylabel("Spending Score (scaled)")
    ax2.legend()
    fig2.patch.set_facecolor('white')  # for dark mode clarity
    st.pyplot(fig2)

    # ---------------------------------------
    # Check cluster only for existing data points
    st.subheader("Check Cluster for Existing Customer Data")

    income_input = st.number_input("Annual Income (k$)", min_value=0.0, step=1.0, format="%.1f")
    score_input = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f")

    if st.button("Check Cluster"):
        # Exact match in original dataset
        matching_rows = data[
            (data['Annual Income (k$)'] == income_input) & 
            (data['Spending Score (1-100)'] == score_input)
        ]
        if not matching_rows.empty:
            cluster_label = matching_rows.iloc[0]['Cluster']
            st.success(f"Data point found in dataset. Cluster: {cluster_label}")
        else:
            st.error("This data point does NOT exist in the uploaded dataset.")