import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

st.title("Customer Segmentation with K-Means Clustering")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file for KMeans analysis", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Feature selection
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    # Standardize for better clustering
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[features])

    # Elbow Method
    st.subheader("Elbow Method")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x_scaled)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss)
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")
    st.pyplot(fig1)

    # Choose number of clusters
    k = st.slider("Choose number of clusters (k)", 1, 10, 5)

    # Fit model
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    data['Cluster'] = kmeans.fit_predict(x_scaled)

    # Show updated DataFrame
    st.subheader("Updated Dataset with Cluster Column")
    st.write(data.head())

    # Cluster plot
    st.subheader("Cluster Visualization")
    fig2, ax2 = plt.subplots()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(k):
        ax2.scatter(
            x_scaled[data['Cluster'] == i, 0],
            x_scaled[data['Cluster'] == i, 1],
            s=100,
            c=colors[i],
            label=f'Cluster {i+1}'
        )
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='yellow', edgecolor='black', label='Centroids')
    ax2.set_title("Customer Clusters")
    ax2.set_xlabel("Annual Income (scaled)")
    ax2.set_ylabel("Spending Score (scaled)")
    ax2.legend()
    st.pyplot(fig2)

    # Prediction input
    st.subheader("Predict Cluster for New Customer")

    income_input = st.number_input("Annual Income (k$)", min_value=0.0, step=1.0)
    score_input = st.number_input("Spending Score (1–100)", min_value=0.0, max_value=100.0, step=1.0)

    if st.button("Predict Cluster"):
        new_point = pd.DataFrame([[income_input, score_input]], columns=features)
        new_point_scaled = scaler.transform(new_point)
        prediction = kmeans.predict(new_point_scaled)[0]

        st.success(f"The customer belongs to **Cluster {prediction}**")

        # Log to file
        log = f"{datetime.now()} | Income: {income_input} | Score: {score_input} | Predicted Cluster: {prediction }\n"
        with open("logs.csv", "a") as log_file:
            log_file.write(log)

        st.info("Prediction logged to `logs.csv`.")