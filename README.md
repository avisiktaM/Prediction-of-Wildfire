# Wildfire Occurance Prediction

Wildfires can cause significant environmental and economic damage, making early risk assessment an important task. In this project, I developed a machine learning pipeline to predict whether a wildfire is likely to occur at a given location based on weather and environmental conditions.

The problem is formulated as a binary classification task where:

1 = wildfire occurred
0 = wildfire did not occur

To understand which approach works best for this dataset, I trained and compared five commonly used machine learning models:

Logistic Regression
Decision Tree
K-Nearest Neighbours (KNN)
Gaussian Naive Bayes
Support Vector Machine (SVM)

The trained models were later deployed through a Streamlit dashboard where users can upload a dataset, select a model, and evaluate its performance without retraining.

Data Description

The dataset is sourced from Kaggle: https://www.kaggle.com/datasets/vijayaragulvr/wildfire-prediction/data

The dataset has 118858 records with 17 features. For Binary Classification only 15 columns are used where the target cariable is "occured" and "frp" is dropped. There is neither missing value nnor duplicate rows. The class distribution is approximately equal.

The FRP (Fire Radiative Power) removed: Initially, I explored whether FRP should be included in the model. However, FRP directly measures the intensity of a fire event itself. During exploratory analysis, I found that FRP values were substantially higher for wildfire observations than non-wildfire observations.

For example:

Median FRP for fire observations ≈ 15.41
Median FRP for non-fire observations ≈ 4.92

Since FRP is closely related to the existence and strength of a fire, including it would effectively provide information about the target variable and lead to data leakage. To ensure a fair prediction task, FRP was removed before model training.

The live Streamlit App: .........

Models Used

To understand which algorithm is most suitable for wildfire occurrence prediction, five different classification techniques were trained and evaluated using the same training and testing datasets. Hyperparameters were selected through cross-validation on the training data.

| Model | Configuration |
| :--- | :--- |
| **Logistic Regression** | C = 0.01 |
| **Decision Tree** | max_depth = 7 |
| **KNN** | k = 11 |
| **Gaussian Naive Bayes** | Default parameters |
| **SVM** | RBF kernel, C = 1.0 |   

The intention was not only to find the most accurate model but also to compare how different machine learning approaches behave on wildfire-related environmental data.

Model Performance

All models were evaluated using the same held-out test dataset consisting of 6,000 observations.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6332 | 0.6829 | 0.6043 | 0.7724 | 0.6781 | 0.2772 |
| **Decision Tree** | 0.6257 | 0.6772 | 0.6139 | 0.6778 | 0.6443 | 0.2527 |
| **KNN** | 0.6285 | 0.6805 | 0.6173 | 0.6771 | 0.6458 | 0.2582 |
| **Gaussian Naive Bayes** | 0.6203 | 0.6764 | 0.5974 | 0.7391 | 0.6607 | 0.2477 |
| **Support Vector Machine** | 0.6437 | 0.6921 | 0.6115 | 0.7887 | 0.6889 | 0.3002 |   

Although the differences between models are not very large, some noticeable trends can be observed.

Logistic Regression

Logistic Regression performed surprisingly well considering its simplicity. It achieved the second-highest accuracy and produced results that were very close to the SVM model.

The model was particularly strong at identifying wildfire events, as reflected by its high recall score. However, this came at the cost of a larger number of false alarms. Since performance changed very little across different values of the regularisation parameter, it appears that a simple linear decision boundary captures most of the information available in the dataset.

Decision Tree

The Decision Tree produced reasonable results but showed clear signs of overfitting when the depth was left unrestricted.

Limiting the depth to seven levels improved generalisation performance substantially. Compared to Logistic Regression, the tree made fewer false alarms but missed more wildfire events. While Decision Trees are easy to interpret, they were not the strongest performer for this dataset.

K-Nearest Neighbours

KNN achieved similar performance to the Decision Tree.

One advantage of KNN was its comparatively high precision, meaning that when it predicted a wildfire, it was slightly more likely to be correct. On the other hand, prediction time is slower because KNN stores all training observations and compares new samples against them during inference.

The tuning results suggested that performance continued improving as the value of k increased, indicating that further experimentation might yield minor improvements.

Gaussian Naive Bayes

Gaussian Naive Bayes served as a useful baseline model.

Its performance was slightly lower than the other approaches, which is not surprising given the independence assumption that the algorithm relies on. Several weather variables in the dataset are correlated with one another, meaning that this assumption is unlikely to hold perfectly.

Despite its limitations, Naive Bayes was extremely fast to train and evaluate.

Support Vector Machine

The Support Vector Machine produced the strongest overall results.

It achieved the highest scores for Accuracy, AUC, Recall, F1 Score and MCC, while also missing the fewest wildfire events. Although the improvement over Logistic Regression was relatively small, SVM consistently ranked first across most evaluation metrics.

The main drawback was computational cost. Training the final model required significantly more time than any of the alternative approaches.

Overall Best Model

Based on the evaluation results, the Support Vector Machine (SVM) was selected as the best-performing model for this dataset.

While the improvement over Logistic Regression was modest, SVM consistently achieved the highest scores across the most important metrics, particularly:

Accuracy
AUC
Recall
F1 Score
MCC

The results suggest that wildfire occurrence can be predicted to some extent using weather and environmental variables, but the signal remains relatively weak. Even the best model achieved only around 64% accuracy, indicating that wildfire occurrence is influenced by many additional factors not included in the dataset.

For situations where simplicity and computational efficiency are important, Logistic Regression would still be a strong practical alternative.

Streamlit Application

A Streamlit dashboard was developed to make the trained models easy to explore and evaluate.

The application loads the saved models and preprocessing objects directly from disk and does not perform any retraining on uploaded datasets.

The dashboard includes:

Dataset Upload

Users can upload a CSV file containing all required predictors and the target variable (occured).

Model Selection

Any of the five trained models can be selected from a dropdown menu.

Evaluation Metrics

The dashboard calculates and displays:

Accuracy
AUC
Precision
Recall
F1 Score
MCC

for the selected model.

Confusion Matrix and Classification Report

A confusion matrix heatmap and a detailed classification report are generated automatically after prediction.

Model Comparison

Results from all five models can be compared side by side on the uploaded dataset.

Additional visualisations such as ROC curves are included to support model comparison.

Single Prediction Tool

A simple interface allows users to manually enter feature values and obtain a wildfire probability prediction for a single observation.

Geospatial View

Because the dataset includes latitude and longitude coordinates, the dashboard also provides a geographic view of wildfire predictions. Users can visualise wildfire and non-wildfire locations directly on an interactive map, helping connect model outputs with their spatial distribution.

Limitations

Although the project achieved its objectives, several limitations should be acknowledged.

The final models were trained on a 30,000-row sample rather than the complete dataset to keep training time manageable.

Model evaluation is based on a single train-test split. Different splits may lead to slightly different performance estimates.

KNN and SVM hyperparameter tuning was performed on a smaller subset of the training data due to computational constraints.

Geographic coordinates (lat and lon) were retained as model features. While useful, they may allow models to learn regional patterns rather than purely environmental relationships.

Wind direction was used in its original form and was not transformed into sine and cosine components, which may have provided additional information.

Wildfire occurrence is influenced by many human and environmental factors that are not present in the dataset. Consequently, prediction performance remains moderate even for the best-performing model.
Reproducibility

All experiments were conducted using:

github link: https://github.com/avisiktaM/Prediction-of-Wildfire

Python 3.12+
Scikit-learn 1.9.1
Pandas
NumPy
Streamlit

A fixed random seed (random_state = 42) was used throughout the project to ensure reproducibility.

The complete workflow, including preprocessing, model training, tuning, evaluation and model export, is available in Preprocessing.ipynb.
