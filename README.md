# 🌍 Tourism Experience Analytics & Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tourism-experience-analytics-qpjv7pkwwtegtmw6zkyacy.streamlit.app/)

## 📌 Overview
This project is an end-to-end machine learning and data science application built to analyze tourist behavior and provide personalized travel recommendations. Through an interactive web dashboard, the system predicts a user's likely travel mode (e.g., Solo, Couples, Family) based on their demographics and suggests top-rated attractions using a sophisticated hybrid recommendation engine.

**🔗 [View the Live Interactive Dashboard Here](https://tourism-experience-analytics-qpjv7pkwwtegtmw6zkyacy.streamlit.app/)**

## 🚀 Features
* **Predictive Analytics (Classification):** Utilizes a trained **LightGBM** model to predict a user's `VisitMode` based on demographic inputs and historical rating behavior.
* **Hybrid Recommendation Engine:** * **Content-Based Filtering:** Dynamically narrows down attractions strictly based on the user's preferred attraction type (e.g., Beaches, Ancient Ruins).
  * **Collaborative Filtering:** Applies **Singular Value Decomposition (SVD)** via `scikit-surprise` to predict how a specific user would rate unvisited attractions, identifying hidden patterns in historical user data.
* **Interactive Dashboard:** A responsive user interface built with **Streamlit** that takes real-time inputs and outputs dynamic recommendations alongside global tourism analytics visualizations.
* **Rating Prediction (Regression):** Predicts the exact rating (1-5 scale) a user is likely to give an attraction using a highly accurate Linear Regression model, allowing businesses to gauge potential customer satisfaction.

## 📊 Exploratory Data Analysis & Visualizations
Extensive data cleaning and EDA were performed in the Jupyter Notebook to understand tourist behavior before modeling. Key visual insights include:
* **Global Tourism Dashboard:** The Streamlit app features live visualizations showing the **Top 10 Most Visited Attraction Types** and **User Distribution by Travel Mode**.
* **Data Engineering & Quality:** Successfully joined 9 distinct relational datasets (Transactions, Users, Cities, Attractions, etc.) into a single consolidated matrix. Handled missing target variables, standardized text formatting across categorical columns, and enforced valid 1-5 rating scales to ensure high model performance.

## 📈 Model Performance & Results

The project involved training and evaluating multiple models across three core data science objectives.

### Objective 1: Regression (Rating Prediction)
To understand the underlying relationships in the data, we compared baseline and advanced regression models:
* **Linear Regression (Best Performer):** Achieved an **MSE of 0.2525** and an **R² of 0.7319**. 
* **Random Forest Regressor:** Achieved an MSE of 0.3176 and an R² of 0.6627.
* **Key Insight:** The simpler Linear Regression model outperformed the complex Random Forest ensemble. This suggests a highly linear relationship between the engineered features and the target variable, making a linear approach both more accurate and computationally efficient.

### Objective 2: Classification (Visit Mode Prediction)
We evaluated algorithms to predict a user's `VisitMode` (e.g., Couples, Family, Solo) based on their profile:
* **Logistic Regression (Baseline):** Achieved an accuracy of 43.18%.
* **LightGBM (Selected Model):** Outperformed the baseline with an overall **accuracy of 49.70%**. 
* **Key Insight:** The detailed classification report for LightGBM reveals strong varying performance across segments. For example, it showed a strong recall (81%) for identifying "Couples", and high precision (78%) when identifying "Business" travelers, showcasing its ability to navigate imbalanced demographic classes.

### Objective 3: Recommendation System Performance
* **Collaborative Filtering (SVD):** The recommendation engine achieved a Root Mean Square Error (**RMSE) of 0.9244**. 
* **Practical Application:** This relatively low error rate allows the system to confidently generate high-quality, personalized "Top 5" attraction lists (e.g., suggesting Waterbom Bali and Tegalalang Rice Terrace for specific user IDs based on hidden historical patterns).

## 💡 Business Insights & Conclusion
By implementing this system, travel platforms and destination management organizations can:
* **Increase Engagement:** Guide users toward attractions they are statistically most likely to enjoy using the hybrid recommendation engine.
* **Targeted Marketing:** Utilize the LightGBM predictions to tailor marketing campaigns (e.g., offering family bundle discounts to users predicted to travel as a "Family").
* **Identify Trends:** Use the analytics dashboard to spot emerging popular attraction types and user segment distributions.

## 🏷️ Technical Tags & Skills Demonstrated
* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning (Regression, Classification, Recommendation)
* Streamlit Application Development

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** LightGBM, scikit-learn, scikit-surprise (SVD), joblib
* **Data Manipulation:** pandas, NumPy
* **Web Framework:** Streamlit
* **Data Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
* `main.ipynb` - The core Jupyter Notebook containing data cleaning, EDA, model training, evaluation, and business insights.
* `app.py` - The Streamlit application script containing the UI layout and inference logic.
* `tourism_models_and_preprocessors.pkl` - The serialized LightGBM model, SVD model, standard scaler, and label encoders.
* `Cleaned_Tourism_Dataset.csv` - The consolidated dataset used for generating real-time predictions and EDA visualizations.
* `requirements.txt` - Python dependencies required to run the application.

## ⚙️ Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SourabhKhamankar22/Tourism-Experience-Analytics.git
   cd Tourism-Experience-Analytics
   ```
2. **Set up a virtual environment:**
    ```
    conda create --name tourism_analytics python=3.10
    conda activate tourism_analytics
    ```
3. **Install dependencies:**
    ```
    pip install streamlit pandas numpy joblib matplotlib seaborn scikit-learn lightgbm scikit-surprise
    ```
4. **Run the Streamlit app:**
    ```
    streamlit run app.py
    ```
