# [WiDS Datathon#2 2024](https://www.kaggle.com/competitions/widsdatathon2024-challenge2/overview)

## Overview

The challenge focused on predicting the metastatic diagnosis period for breast cancer patients. Our approach was centered on robust data imputation, strategic feature engineering, and the use of diverse machine learning models. Through iterative experimentation and careful validation, we developed a solution that balanced accuracy and generalizability, ultimately securing the second place in the competition.

## Data Imputation

We observed that the features with the most missing values were patient_race, payer_type, and bmi. Our primary focus was on imputing these features as accurately as possible to improve model performance.

**Strategy 1**: Consider all the missing values as a different category. This approach helped in preserving the information about missingness but was not sufficient on its own.

**Strategy 2**: Fill the missing values with median (for numerical features) and mode (for categorical features).This method provided a baseline imputation but lacked the sophistication needed for better predictions.

**Strategy 3**: Impute each feature individually

-   **payer_type**: We found that people above the age of 65 are typically covered by MEDICARE ADVANTAGE. Hence, we filled the missing values for patients above 65 with MEDICARE ADVANTAGE, covering around 500 out of 1300 missing values. For the remaining values, we created age bins of [0, 22, 45, 65, 200] and imputed the missing values with the mode of payer_type within each age group.
-   **bmi**: Using the same age bins, we imputed missing bmi values with the median bmi of the respective age groups.
-   **patient_race**: We filled the missing values with the mode of patient_race grouped by patient_zip3.

We found that the third strategy provided the best results.

## Feature Engineering and Feature Selection

### Breast Cancer Diagnosis Description

Since this feature contained textual information, we employed several techniques to convert it into numerical form:

**Strategy 1**: Used TFIDF (Term Frequency-Inverse Document Frequency) representation.
**Strategy 2**: Extracted important terms from the text and created a single binary feature.
**Strategy 3**: Concatenated all features built from the text into a single string feature.

The third strategy yielded the best performance.

### Temperature Features

We observed that the temperature for each month remained consistent across years, so we computed the mean temperature for each month. Using feature importance analysis, we identified `May, June, July, August, and September` as the most important months influencing the target variable.

### ICD Features

Building on insights from previous editions of the competition, we enhanced the breast_cancer_diagnosis_code feature. We created additional features such as `icd_state, icd_payer, icd_bmi, and icd_race` by grouping these variables with ICD codes. These new features provided a richer context for our model.

We experimented with other features like area_value and higher-degree polynomial features, but they did not improve performance.

## Feature Selection

We used feature importance metrics and SHAP values to identify the most relevant features. Additionally, we employed Sequential Feature Selector for some tree-based models to optimize feature selection. The top-performing features were:
`type, binned_age, breast_cancer_diagnosis_code, icd_bmi, disabled, metastatic_cancer_diagnosis_code, breast_cancer_diagnosis_desc, icd_race, icd_payer, icd_state, patient_age`

## Model Building

We experimented with a variety of linear and tree-based models, including:
`Ridge Regression, K-Nearest Neighbors (KNN), Random Forest Regressor, Extra Trees, Gradient Boosting, Hist Gradient Boosting, XGBoost, LightGBM, CatBoost`

To combine the strengths of these models, we employed an ensemble approach. Learning from the previous edition, we ensured that the models had minimal discrepancies between training and cross-validation (CV) scores. This strategy helped us avoid overfitting and ensured better generalization on unseen data.

## Model Selection

We chose our final model based on both local CV performance and public leaderboard results. The best-performing submission on local CV and the best on the public leaderboard were evaluated, and the one that gave us the best score on the private leaderboard was selected.

In conclusion, our meticulous approach to data imputation, feature engineering, model building, and selection was instrumental in achieving second place in the WIDS Datathon 2024 Challenge #2. This experience has been invaluable, providing deep insights into handling complex datasets and building robust predictive models.

[Here](https://www.kaggle.com/code/predator4hack/wids-datathon-challenge-2-final-notebook) is our code. You can also find the code in the current dir in the notebook named wids-datathon-challenge-2-final-notebook.ipnb

## Reference

Big thanks to everyone who participated in the discussion forums and provided us with valuable thoughts during the competition. Here are the resources that were particularly helpful to us:

-   https://www.kaggle.com/competitions/widsdatathon2024-challenge1/discussion/481084
-   https://www.kaggle.com/competitions/widsdatathon2024-challenge1/discussion/469583
-   https://www.icd10data.com/ICD10CM/Codes/C00-D49/C50-C50/C50-/C50.919
-   https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection.html
-   https://www.kaggle.com/competitions/widsdatathon2024-challenge2/discussion/506049
