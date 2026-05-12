# Employee Job Satisfaction Prediction Using Machine Learning



## Project Overview



This project predicts employee job satisfaction using machine learning models trained on workplace-related features. The project includes exploratory data analysis (EDA), data preprocessing, feature engineering, class imbalance handling, model training, hyperparameter tuning, and explainable AI techniques.



The primary goal of this study is to identify the most important workplace factors influencing employee job satisfaction while comparing the predictive performance of multiple machine learning algorithms.



---



## Project Highlights



- End-to-end machine learning workflow

- Class imbalance handling using Class Weights and SMOTE

- Hyperparameter optimization using RandomizedSearchCV

- Ensemble learning with Random Forest, XGBoost, and LightGBM

- Explainable AI using SHAP

- Streamlit deployment preparation



---



## Problem Statement



Employee job satisfaction plays a critical role in productivity, employee retention, workplace well-being, and organizational success. Predicting job satisfaction can help organizations identify potential workplace issues and make informed decisions to improve employee experience.



This project applies supervised machine learning techniques to predict employee job satisfaction levels based on employee-related and workplace-related factors.



---



## Dataset Description



The dataset contains employee-related information and workplace characteristics associated with job satisfaction levels.



### Dataset Characteristics



- Multiple numerical and categorical features

- Multi-class target variable representing job satisfaction levels

- Presence of class imbalance

- Workplace and well-being related attributes



### Example Features



- Work Environment

- Work-Life Balance

- Workload

- Stress Level

- Sleep Quality

- Salary Information

- Career Growth

- Years at Company



---



## Technologies Used



### Programming Language

- Python



### Libraries and Frameworks

- Pandas

- NumPy

- Scikit-learn

- XGBoost

- LightGBM

- SHAP

- Matplotlib

- Seaborn

- Imbalanced-learn (SMOTE)

- Joblib

- Streamlit



---



## Techniques Used



The project applied several machine learning and data science techniques to improve predictive performance, model robustness, and interpretability.



### Data Preprocessing

- Missing value handling

- Duplicate removal

- Feature encoding

- Boolean conversion

- Outlier handling using IQR

- Feature scaling

- Low variance feature analysis

- Multicollinearity analysis using Correlation and VIF



### Machine Learning Models



The project evaluated multiple supervised machine learning algorithms, including both baseline and advanced ensemble learning models.



#### Baseline Models

- Logistic Regression

- Decision Tree

- Support Vector Machine (SVM)



#### Advanced Ensemble Models

- Random Forest

- XGBoost

- LightGBM



### Class Imbalance Handling

- Class weighting techniques

- SMOTE (Synthetic Minority Oversampling Technique)



### Model Optimization

- Hyperparameter tuning using RandomizedSearchCV

- Cross-validation

- Performance optimization across baseline and advanced models

- Model performance evaluation



### Explainable AI

- SHAP (SHapley Additive exPlanations) for model interpretability, feature importance analysis, and understanding prediction behavior



---



## Evaluation Metrics



The models were evaluated using the following performance metrics:



- Accuracy

- Precision

- Recall

- F1-Score

- ROC-AUC

- Confusion Matrix

- Cross-Validation Scores



---



## Final Model Selection



The final model was selected based on:



- Cross-validation performance

- Test set performance

- Overall model stability

- Generalization capability



From the evaluation results, Random Forest and LightGBM emerged as the top-performing models, both demonstrating strong and closely comparable performance across all evaluation metrics.



However, Random Forest achieved slightly higher accuracy and F1-score, giving it a marginal advantage in predictive capability.



In addition, Random Forest maintained consistent results between cross-validation and test evaluation, confirming its stability and ability to generalize effectively to unseen data.



While LightGBM and XGBoost also delivered strong competitive performance, Random Forest provided the best overall balance among accuracy, F1-score, robustness, and generalization.



Based on these findings, Random Forest was selected as the final model for this classification task.



---



## Explainable AI with SHAP



SHAP (SHapley Additive exPlanations) was applied to improve model interpretability and understand how individual features influenced prediction outcomes.



The analysis revealed that workplace and well-being related factors played a major role in employee job satisfaction prediction.



### Most Influential Features

- Work Environment

- Work-Life Balance

- Workload

- Stress

- Sleep Quality



SHAP analysis improved model transparency and provided meaningful insights into the key drivers of employee satisfaction.



---



## Results


The evaluation results demonstrated that ensemble learning methods consistently outperformed simpler baseline models across multiple performance metrics.


### Key Findings

- Random Forest achieved the strongest overall performance

- LightGBM and XGBoost also demonstrated competitive predictive capability

- Ensemble models improved predictive accuracy, robustness, and generalization

- SMOTE improved minority-class prediction performance

- SHAP analysis enhanced model interpretability and transparency

---
## Visual Results

### Model Accuracy Before and After SMOTE

![Model Accuracy Before and After SMOTE](./images/Model%20Accuracy%20Before%20and%20After%20SMOTE.png)

### Model F1-Score Before and After SMOTE

![Model F1-Score Before and After SMOTE](./images/F1_Score%20Before%20and%20After%20SMOTE.png)

### Random Forest Confusion Matrix After Tuning

![Random Forest Confusion Matrix After Tuning](./images/Random%20Forest%20Confusion%20Matrix%20After%20Tuning.png)

### SHAP Feature Importance Analysis

![SHAP Feature Importance](./images/Shap-Based%20Feature%20Importance%20Analysis.png)

### SHAP Summary Plot

![SHAP Summary Plot](./images/SHAP.Summary_plot(shap_values_avg).png)

---


## Project Structure



```text



employee-job-satisfaction-ml/


├── app/
│   └── app.py

├── data/
│   ├── employee_survey.csv
│   └── employee_survey_cleaned.csv

├── images/
│   ├── Baseline Model Comparison.png
│   ├── Correlation_Matrix_Heatmap.png
│   ├── boxplots_outliers_analysis.png
│   ├── boxplots_outliers_analysis (cont.).png
│   ├── Random Forest Confusion Matrix.png
│   ├── Random Forest Confusion Matrix After Tuning.png
│   ├── SHAP.Summary_plot(shap_values_avg).png
│   ├── Shap-Based Feature Importance Analysis.png
│   ├── streamlit_home.png
│   └── streamlit_prediction.png

├── models/
│   └── best_model.pkl

├── notebooks/
│   └── job_satisfaction_analysis.ipynb

├── README.md
├── requirements.txt
└── .gitignore

```

---

## How to Run the Project



### Clone the Repository



```bash

git clone https://github.com/ThereseK02/employee-job-satisfaction-ml.git

```



### Navigate to the Project Folder



```bash

cd employee-job-satisfaction-ml

```



### Install Dependencies



```bash

pip install -r requirements.txt

```



### Run the Streamlit Application



```bash

streamlit run app/app.py

```



## Deployment



The project is prepared for deployment using Streamlit.



The deployed application will allow users to:



- Input employee-related information

- Generate job satisfaction predictions

- View model outputs interactively



---



## Application Preview



### Streamlit Home Interface



![Streamlit Home](./images/streamlit_home.png)



### Prediction Interface



![Prediction Interface](./images/streamlit_prediction.png)

---

## Future Improvements



Potential future improvements include:



- Full Streamlit deployment

- Real-time prediction dashboard

- Additional explainability visualizations

- Deep learning model experimentation

- Integration with HR analytics systems



---



## Final Conclusion



In this project, I developed a complete machine learning workflow to predict employee job satisfaction by combining exploratory data analysis, data preprocessing, model training, and performance evaluation.



To address class imbalance, I initially applied class weighting techniques in selected models and later refined the approach using SMOTE to improve the representation of minority classes. Multiple machine learning models were trained, including both baseline and advanced ensemble algorithms, and their performance was further improved through hyperparameter tuning.



All models were evaluated using consistent performance metrics, including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. The evaluation results demonstrated that ensemble methods consistently outperformed simpler models, with Random Forest and LightGBM achieving the strongest overall performance.



Based on cross-validation results, test-set performance, and model stability, Random Forest was selected as the final model because it provided the best balance among predictive accuracy, robustness, and generalization capability.



To improve model interpretability, SHAP (SHapley Additive exPlanations) analysis was applied to identify how individual features influenced prediction outcomes. The analysis revealed that workplace and well-being factors — including work environment, work-life balance, workload, stress, and sleep — played a significant role in determining employee job satisfaction.



Overall, this project demonstrates the importance of combining effective class imbalance handling, comprehensive model evaluation, ensemble learning, and explainable AI techniques to build reliable, interpretable, and actionable predictive models.



---



## Author



Therese Kabayanja



Machine Learning Engineer | Data Science | Software Development



