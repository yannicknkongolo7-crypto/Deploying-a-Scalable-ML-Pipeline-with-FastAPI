# Model Card for Census Income Classifier

## Model Details

This repository contains a RandomForestClassifier trained to predict whether
an individual's annual income is >50K or <=50K using features from the U.S.
Census dataset. The model was trained using scikit-learn's
RandomForestClassifier with 300 estimators and random_state=0.

## Intended Use

This model is intended for educational purposes and demonstration of a
deployment pipeline using FastAPI. It is not intended for use in
high-stakes decision making (e.g., lending, hiring) without further
validation and fairness analysis.

## Training Data

The model was trained on a cleaned version of the U.S. Census income
dataset located at `data/census.csv`. The dataset was split into a
training set (80%) and a test set (20%) using stratified sampling on the
`salary` label.

## Evaluation Data

Evaluation was performed on the held-out test split (20% of the dataset).

## Metrics

The model is evaluated using Precision, Recall, and F1-score. The overall
metrics computed on the test set are:

- Precision: 0.7365
- Recall: 0.6346
- F1-score: 0.6817

Per-slice metrics for several categorical features are written to
`slice_output.txt` in the repository root. These provide per-category
performance (precision/recall/f1) for values such as `education`,
`workclass`, and `sex`.

## Ethical Considerations

Predicting income brackets from demographic data can introduce or
amplify biases. This example does not perform any fairness mitigation or
thorough bias analysis. Do not deploy the model to production for
decision-making without a careful fairness and privacy review.

## Caveats and Recommendations

- The dataset may contain biases that reflect historic and social
  disparities.
- The model's metrics are suitable for a demonstration project, but
  require further tuning and validation for production use.
- Consider calibration, fairness audits, and explainability before any
  real-world deployment.
