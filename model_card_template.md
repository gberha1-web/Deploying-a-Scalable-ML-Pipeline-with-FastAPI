# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a supervised binary classification model trained to predict whether an individual’s income exceeds $50K per year. It uses a Logistic Regression classifier implemented with scikit-learn. The model takes demographic and employment-related features as input and outputs a binary prediction representing income class.

## Intended Use
The intended use of this model is for educational purposes to demonstrate how to build, evaluate, and deploy a machine learning pipeline. The model should not be used to make real-world decisions about individuals, particularly in high-stakes contexts such as hiring, lending, or eligibility determination.

## Training Data
The model was trained on the U.S. Census Income dataset (`census.csv`). The dataset contains demographic attributes such as age, education, workclass, occupation, race, sex, and hours worked per week. The target label is income, categorized as either `<=50K` or `>50K`. Some categorical features include missing values encoded as the string `"?"`, which were treated as valid categories during training.
## Evaluation Data
The evaluation data consists of a held-out test set comprising 20% of the original dataset. The split was stratified by the income label to preserve class distribution between the training and test sets.

## Metrics
The model was evaluated using precision, recall, and F1 score. On the test dataset, the model achieved a precision of 0.7159, a recall of 0.5963, and an F1 score of 0.6507. In addition to overall performance, the model’s performance was evaluated across slices of the data for each categorical feature. Metrics for each slice were computed and saved to `slice_output.txt`.


## Ethical Considerations
The dataset contains sensitive demographic attributes such as sex and race, which introduces the risk of biased predictions. Performance varies across different categorical slices, indicating that the model may behave unevenly for different demographic groups. As a result, this model should not be used in applications where fairness and equity are critical without further bias analysis and mitigation.
## Caveats and Recommendations
The model is trained on historical census data and may not generalize well to current or future populations. The presence of class imbalance in the target variable may also affect performance, particularly recall for the minority class. Future improvements could include better handling of missing values, feature scaling, and experimenting with more advanced models or fairness-aware training techniques.