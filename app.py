import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

km = joblib.load("./model/kmeans.pkl")
ms = joblib.load("./model/meanshift.pkl")
X = joblib.load("./model/scaler/X.pkl")
X_scaled = joblib.load("./model/scaler/X_scaled.pkl")

st.title('Clustering Gaji')

work_year = st.selectbox('Work Year', options=[str(year) for year in range(2020, 2024)], index=0)
experience_level = st.selectbox('Experience Level', ("EX", "SE", "MI", "EN"), index=0)
employment_type = st.selectbox('Employment Type', ("FT", "PT", "CT"), index=0)
job_title = st.text_input('Job Title', placeholder="Inputkan job title")
salary_in_usd = st.number_input('Salary in USD', step=1000, min_value=0)
employee_residence = st.selectbox('Employee Residence', ("US", "IN", "UK", "DE"), index=0)
company_location = st.selectbox('Company Location', ("US", "IN", "UK", "DE"), index=0)
company_size = st.selectbox('Company Size', ("S", "M", "L"), index=1)


if st.button("Prediksi"):
    new_data = {
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'salary_in_usd': salary_in_usd,
        'employee_residence': employee_residence,
        'company_location': company_location,
        'company_size': company_size
    }
    new_data = pd.DataFrame([new_data])

    LE = LabelEncoder()

    # Apply Label Encoding to categorical features
    new_data['work_year'] = LE.fit_transform(new_data['work_year'])
    new_data['experience_level'] = LE.fit_transform(new_data['experience_level'])
    new_data['employment_type'] = LE.fit_transform(new_data['employment_type'])
    new_data['job_title'] = LE.fit_transform(new_data['job_title'])
    new_data['employee_residence'] = LE.fit_transform(new_data['employee_residence'])
    new_data['company_location'] = LE.fit_transform(new_data['company_location'])
    new_data['company_size'] = LE.fit_transform(new_data['company_size'])


    cluster_1 = km.predict(new_data)
    cluster_2 = ms.predict(new_data)

    st.success(f"Data input dikelompokan menggunakan algoritma K-means ke dalam cluster {cluster_1[0]}")
    st.success(f"Data input dikelompokan menggunakan algoritma Mean Shift ke dalam cluster {cluster_2[0]}")

   # reduksi dimensi ke 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    new_data_pca = pca.transform(new_data)

    # Visualize clustering with K-Means
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    # K-Means Clustering plot
    ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap='viridis', marker='o')
    ax[0].scatter(new_data_pca[:, 0], new_data_pca[:, 1], c='red', label='New Data', marker='x', s=100)
    ax[0].set_title('Visualisasi K-Means Pada Data Baru')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[0].legend()

    # MeanShift Clustering plot
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=ms.labels_, cmap='viridis', marker='o')
    ax[1].scatter(new_data_pca[:, 0], new_data_pca[:, 1], c='red', label='New Data', marker='x', s=100)
    ax[1].set_title('Visualisasi Mean Shift Pada Data Baru')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].legend()

    # Display both plots in Streamlit
    st.pyplot(fig)
