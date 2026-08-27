# Telco Customer Churn Prediction — Machine Learning

A Python-based machine learning project focused on identifying customers at risk of churn and improving the model's ability to detect high-risk customers.

## Business Problem

Customer churn directly affects revenue and retention costs. The goal of this project was to develop a predictive workflow that identifies customers who are more likely to leave, allowing a business to prioritize proactive retention efforts.

## Project Approach

The project uses a telecom customer churn dataset and compares multiple classification approaches, including:

- Logistic Regression
- Multi-Layer Perceptron (MLP) Neural Network
- Classification threshold optimization
- Exploratory data analysis
- Model performance evaluation

The analysis focuses not only on overall accuracy, but also on recall because failing to identify customers who actually churn can limit the usefulness of a retention model.

## Machine Learning Workflow

1. Load and inspect customer churn data
2. Prepare variables for modeling
3. Perform exploratory data analysis
4. Split data for training and evaluation
5. Train Logistic Regression and MLP models
6. Evaluate classification performance
7. Examine churn probability distributions
8. Adjust the classification threshold
9. Compare model performance after threshold optimization

## Model Results

### Initial Modeling

The initial MLP model achieved high overall accuracy but very low churn recall, meaning it failed to identify many customers who actually churned.

### Threshold-Optimized MLP

Adjusting the classification threshold improved the balance between overall predictive performance and churn detection.

Key result:

- Churn Recall improved from approximately **4.66% to 59.48%**

This demonstrates an important business analytics lesson: the default classification threshold is not always the most useful decision threshold for the business problem.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Logistic Regression
- Multi-Layer Perceptron (MLP)
- Data preprocessing and feature transformation
- Classification metrics and model evaluation

## Business Value

For customer-retention decisions, identifying more true churners may be more valuable than maximizing accuracy alone.

A model with stronger churn recall can help organizations:

- Identify more at-risk customers
- Prioritize retention campaigns
- Reduce missed churn cases
- Support data-driven customer intervention strategies

## Repository Contents

- `q1_first_pass.py` — initial analysis and modeling workflow
- `q3_skill_guided.py` — refined modeling and evaluation workflow
- `q1_model_metrics.csv` — initial model performance metrics
- `q3_model_metrics.csv` — refined model performance metrics

## Project Context

Developed as an individual academic machine learning assignment for TOM 5300.

## Disclaimer

This project was developed for academic and portfolio purposes. Model results should be interpreted within the context of the dataset and experimental workflow used in the project.
