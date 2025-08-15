# Bias Crime Intelligence Engine: Predicting, Mapping, and Classifying Hate Crimes Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)


---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Ethical Considerations](#ethical-considerations)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

---

## 📖 Overview
The **Bias Crime Intelligence Engine (BCIE)** is a machine learning and deep learning framework for **forecasting, classifying, and mapping hate crimes** using historical U.S. FBI Uniform Crime Reporting (UCR) data (**1991–2023**) combined with U.S. geospatial datasets.

It leverages **ensemble methods** (Random Forest, LightGBM) alongside **temporal deep learning architectures** (Temporal Convolutional Network, Spatio-Temporal GRU) to detect **trends, escalation patterns, and geographical hotspots**.

---

## 🚀 Features

### 1. Data Preprocessing & Feature Engineering
- Cleans and aggregates raw FBI UCR hate crime data  
- Encodes categorical features for classical and deep learning models  
- Creates lag features, rolling averages, and year-on-year percentage changes  

### 2. Exploratory Data Analysis (EDA)
- Visualisations of bias motivations, offence types, and state-level distributions  
- **Change point detection** to identify escalation or decline trends  

### 3. Predictive Modelling
- **Random Forest** & **LightGBM** for baseline ensemble learning  
- **Temporal Convolutional Networks (TCN)** for sequence learning  
- **Spatio-Temporal GRU (ST-GRU)** for combined temporal and spatial dynamics  

### 4. Explainability & Ethics
- **SHAP** analysis for feature importance and interpretability  
- Ethical AI practices:
  - Data anonymisation  
  - Bias checks  
  - GDPR alignment  

### 5. Geospatial Visualisation
- **Folium-based interactive maps** of hate crime hotspots  
- Category-specific temporal heatmaps  

---

## 🛠 Technology Stack

**Languages & Frameworks**  
- Python  
- PyTorch  
- Keras  
- Scikit-learn  
- LightGBM  

**Data Processing**  
- Pandas  
- NumPy  

**Visualisation**  
- Matplotlib  
- Seaborn  
- Plotly  
- Folium  

**Explainability**  
- SHAP  

**Change Point Detection**  
- ruptures  

---

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/vspvsp3355/Bias-Crime-Intelligence-Engine
cd Bias-Crime-Intelligence-Engine
