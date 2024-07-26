# InfrastructureDeficit_GenAI -  African Infrastructure Development Index (AIDI) Prediction

This repository contains a comprehensive analysis and predictive modeling for the African Infrastructure Development Index (AIDI). The analysis utilizes various machine learning models to predict the AIDI based on four key infrastructure sectors: Transport (TCI), Electricity (ECI), Information and Communication Technology (ICT), and Water and Sanitation (WSSCI).

## Table of Contents 🔖

- [Introduction](#Introduction)
- [Data Collection](#Data-Collection)
3. Data Preprocessing
4. Data Visualizations
5. Modeling
6. Evaluation
7. Cross Validation
8. Predictions
9. Data Analytics Chatbot Development 
10 Conclusion and Recommendations
11. How to Use
12. Dependencies

## Introduction
The African Infrastructure Development Index (AIDI) is a composite index used to evaluate infrastructure development across African countries. This project aims to predict the AIDI for different countries using data from various infrastructure sectors. The models built and evaluated in this project include Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest, and Gradient Boosting.


## Data Collection :floppy_disk:
Data for the following infrastructure sectors were collected:

- Transport (TCI)
- Electricity (ECI)
- Information and Communication Technology (ICT)
- Water and Sanitation (WSSCI)

The data is normalized to ensure comparability across sectors.


## Data Visualizations


[AIDI Prediction by Region](InfrastructureDeficit_GenAI\Images\AIDI.jpeg)

[ECI Prediction by Region](InfrastructureDeficit_GenAI\Images\ECI.jpeg)

[Top And Bottom Performere Prediction](InfrastructureDeficit_GenAI\Images\perfromer.jpeg)

[TCI Prediction](InfrastructureDeficit_GenAI\Images\TCI.jpeg)

## Data Preprocessing
The collected data was preprocessed to normalize the values between 0 and 1. This step ensures that the different infrastructure sector indices are on a comparable scale. The normalization process allows for a fair assessment and combination of the indices.

## Modeling
Multiple models were trained to predict the AIDI:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Each model was evaluated based on Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.

## Evaluation
The models were evaluated using the training data, and their performance metrics were recorded:

- Linear Regression: MAE = 2.45, MSE = 10.82, R² = 0.97
- Polynomial Regression: MAE = 1.59, MSE = 4.85, R² = 0.99
- Decision Tree: MAE = 0.00, MSE = 0.00, R² = 1.00
- Random Forest: MAE = 1.48, MSE = 7.24, R² = 0.98
- Gradient Boosting: MAE = 0.18, MSE = 0.05, R² = 0.99

## Cross Validation
Cross-validation was performed to ensure the robustness of the models. The following metrics were obtained using 5-fold cross-validation:

-Linear Regression: CV_MAE = 2.89, CV_MSE = 16.77, CV_R² = 0.95
- Ridge Regression: CV_MAE = 2.88, CV_MSE = 16.72, CV_R² = 0.95
- Lasso Regression: CV_MAE = 2.88, CV_MSE = 16.70, CV_R² = 0.95
- Decision Tree: CV_MAE = 5.98, CV_MSE = 93.70, CV_R² = 0.71
- Random Forest: CV_MAE = 3.83, CV_MSE = 46.39, CV_R² = 0.86
- Gradient Boosting: CV_MAE = 4.67, CV_MSE = 60.47, CV_R² = 0.81


## Predictions
The models were used to predict the AIDI for specific countries. For example, the predicted AIDI mean% for Nigeria was 17.40, compared to the actual AIDI mean% of 17.16.

## Data Analytics Chatbot Development

The T5 Data Analytics Chatbot is an interactive Streamlit application that allows users to analyze various datasets related to African infrastructure development. Powered by a large language model, this chatbot can answer questions about the data, providing insights and generating visualizations on demand.

## Features
- Interactive data selection from multiple datasets
- AI-powered question answering about the selected data
- Dynamic data visualization
- Question history tracking

## Datasets
The application includes the following datasets:
- Africa Infrastructure Development Index (AIDI)
- Water and Sanitation Service (WSS) Composite Index
- Electricity Composite Index
- ICT Composite Index
- Transport Composite Index
- Fused Dataset (combined metrics)

## Requirements
- Python 3.7+
- Streamlit
- Pandas
- PandasAI
- LangChain
- Groq API access

## Usage
1. Run the Streamlit app: streamlit run app.py

2. 2. Select a dataset from the sidebar.
3. Enter your question about the data in the text input field.
4. Click "Get Answer" to receive AI-generated insights.
5. View any generated charts and your question history.

## Configuration
- The `datasets` dictionary in the script can be modified to include additional datasets.
- The AI model can be adjusted by changing the `model_name` in the `ChatGroq` initialization.


## Conclusion
The models developed in this project demonstrate a high degree of accuracy in predicting the AIDI. The Gradient Boosting model, in particular, showed excellent performance with an R² score close to 1.0, indicating a strong predictive capability.

## Recommendations
- Improved Data Collection: Future work should focus on collecting more granular data for each infrastructure sector to improve model accuracy.

- Scaling Up: The current model is a prototype. Scaling it to real-world applications would involve using larger datasets and more sophisticated algorithms.

- Additional Features: Incorporating additional features such as economic indicators and demographic data may improve prediction accuracy.

## How to Use
To run the project, follow these steps:

1. Clone the repository:
```bash
git https://github.com/JeevalShah/InfrastructureDeficit_GenAI.git
```

2. Install the required dependencies (see the Dependencies section).

3. Run the analysis and prediction scripts
```bash
python main.py
```


## Dependencies
The project requires the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib


Install the dependencies using:
```bash
pip install -r requirements.txt
```


