# Deforestation Detection (Fire Classification) using Satellite Data

## 📖 Overview

This project focuses on the classification of fire types in India using thermal anomaly data from NASA's MODIS satellite sensors for the years 2021, 2022, and 2023. The primary goal is to build and evaluate a machine learning model that can accurately distinguish between different fire sources, such as vegetation fires and other thermal anomalies.

The project includes a Jupyter Notebook (`classification.ipynb`) detailing the entire data science workflow from data cleaning and exploratory analysis to model training and evaluation. The best-performing model, a **Random Forest Classifier**, is then saved and deployed in an interactive web application using Streamlit (`app.py`).

## 🎯 Objective

To develop a machine learning classification model that can accurately predict the type of fire using MODIS fire detection data for India from 2021 to 2023.

## 📊 Dataset

The dataset used in this project is sourced from NASA’s Fire Information for Resource Management System (FIRMS) and contains thermal anomaly data captured by the MODIS sensor on the Terra and Aqua satellites.

The data is split across three files for each year:
* `1.Dataset for Deforestation (Fire Classification) 2021.csv`
* `2.Dataset for Deforestation (Fire Classification) 2022.csv`
* `3.Dataset for Deforestation (Fire Classification) 2023.csv`

Key features in the dataset include:
* **latitude** & **longitude**: Geographic coordinates of the fire.
* **brightness**: Brightness temperature in Kelvin.
* **scan** & **track**: Sensor's scan and track information.
* **acq_date** & **acq_time**: Date and time of acquisition.
* **confidence**: Confidence level of the fire detection.
* **frp**: Fire Radiative Power (a measure of fire intensity).
* **type**: The target variable, indicating the class of fire (0 = vegetation fire, 2 = other).

## 🛠️ Methodology

The project follows a structured data science pipeline:

1. **Data Loading and Integration**: The three yearly datasets are loaded using Pandas and concatenated into a single DataFrame for unified analysis. The data is verified to have no missing values or duplicates.
2. **Exploratory Data Analysis (EDA)**:
   * The distribution of fire types was analyzed, revealing a significant class imbalance.
   * A **Folium map** was generated to visualize the geographical distribution of fire incidents across India.
3. **Feature Engineering and Preprocessing**:
   * Temporal features such as `year`, `month`, `day_of_week`, and `day_of_year` were extracted from the `acq_date` column.
   * Categorical features (`satellite`, `daynight`) were one-hot encoded.
   * Numerical features were standardized using `StandardScaler` to ensure they are on a comparable scale for the models.
4. **Model Training and Evaluation**:
   * The data was split into training and testing sets.
   * Four different classification models were trained and compared: **Logistic Regression**, **K-Nearest Neighbors**, **Random Forest Classifier**, and **XGBoost Classifier**.
   * Based on accuracy and other metrics, the **Random Forest Classifier** was selected as the best-performing model.
5. **Model Persistence**: The trained Random Forest model and the data scaler were saved using `joblib` for later use in the web application.

## ⚙️ Installation

To run this project, you need to have Python installed. You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn folium xgboost streamlit
```

## 🚀 Usage

There are two main components to this project: the analysis notebook and the interactive web application.

### 1. Jupyter Notebook Analysis

To view the complete analysis, data preprocessing, and model training steps, you can run the `classification.ipynb` notebook using Jupyter:

```bash
jupyter notebook classification.ipynb
```

### 2. Streamlit Web Application

The trained model is deployed in a user-friendly web interface. To run the app:

1. Ensure all required libraries are installed.
2. Open your terminal or command prompt.
3. Navigate to the project's root directory.
4. Run the following command:

```bash
streamlit run app.py
```

This will launch a local web server, and you can interact with the model by providing input values in the sidebar to get a fire type prediction.

## 📁 File Descriptions

* **`classification.ipynb`**: Contains the full data analysis, EDA, feature engineering, model training, and evaluation process.
* **`app.py`**: The Python script for the Streamlit web application that serves the trained model.
* **`README.md`**: This file, providing an overview and instructions for the project.
* **`*.csv`**: The three data files containing the fire data for the years 2021, 2022, and 2023.
* **`scaler.joblib`**: The saved `StandardScaler` object used for preprocessing data.
* **`rfc_model.joblib`**: The saved, trained Random Forest Classifier model.
