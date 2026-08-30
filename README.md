Depression Risk Prediction

A machine learning project that predicts depression risk patterns using screen habits, sleep quality, and sleep schedule data.

The project includes the complete model development process in Jupyter Notebook and a Streamlit web application where users can enter their daily habits and get an estimated risk score.

This project is for educational purposes only and is not intended to provide a medical diagnosis.

Live App

The project is deployed using Streamlit.

You can try the application using the link provided in the repository's About section.

How it works

The model uses behavioral and sleep-related information such as:

Screen time

Sleep quality

Average sleep duration

Weekend sleep schedule

Sex

Some additional features are calculated from the user's input before prediction.

The application then uses the trained machine learning model to estimate the probability of belonging to an elevated-risk group.

Model Development

The DRP.ipynb notebook contains the main machine learning workflow, including:

Data cleaning

Exploratory data analysis

Feature selection

Feature engineering

Removing data leakage

Train/test splitting

Feature scaling and encoding

Model training

Model evaluation

Classification threshold tuning

Since the target classes are imbalanced, the model was evaluated using more than accuracy alone.

Metrics such as precision, recall, F1-score, and ROC-AUC were also considered.

Project Structure

Depression_Risk_Prediction/
│
├── DRP.ipynb
├── app.py
├── mental_health_features.pkl
├── mental_health_risk_model.pkl
├── mental_health_threshold.pkl
└── requirements.txt

DRP.ipynb

Contains the complete data analysis and machine learning workflow.

app.py

Contains the Streamlit frontend and prediction logic.

mental_health_risk_model.pkl

Saved trained machine learning model used by the Streamlit application.

mental_health_features.pkl

Stores the feature information required by the trained model.

mental_health_threshold.pkl

Stores the selected classification threshold used for the final prediction.

requirements.txt

Contains the Python dependencies required to run the application.

Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Jupyter Notebook

Streamlit

Run Locally

Clone the repository:

git clone https://github.com/Muzammil-Ansari-17/Depression_Risk_Prediction.git

Move into the project folder:

cd Depression_Risk_Prediction

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

What I learned

This project gave me practical experience with building a complete machine learning workflow instead of stopping after model training.

I worked with an imbalanced classification problem, compared different evaluation metrics, experimented with prediction thresholds, saved the trained model using pickle files, and connected the model to a Streamlit frontend.

It also helped me understand why a good machine learning model should be evaluated based on the actual problem instead of focusing only on accuracy.

Disclaimer

This project is made for learning and experimentation with machine learning.

The prediction produced by the application should not be treated as a diagnosis or as medical advice. Anyone concerned about their mental health should consult a qualified healthcare professional.
