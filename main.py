import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

### DATA ###

# Load data
data = pd.read_csv('data.csv')

# Data Layout
print(data.head())
print(data.info())
print(data.describe())

# Drop unnecessary columns
data.drop(['Unnamed: 32'], axis=1, inplace=True)

# Convert diagnosis column
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])  # M=1, B=0

# Convert diagnosis to categorical type
data['diagnosis'] = data['diagnosis'].astype('category')

# Plot distribution of diagnosis (Malignant vs Benign)
data['diagnosis'].value_counts().plot(kind='bar')

# Separate target (y) and features (x)
y = data['diagnosis']
x = data.drop(['diagnosis'], axis=1)

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

### TRAIN LOGISTIC MODEL ###
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=43)

# Train a Logistic Regression model
lr = LogisticRegression(max_iter=10000)  # increase max_iter if model doesn't converge
lr.fit(X_train, Y_train)

# Make predictions on the test set
Y_prediction = lr.predict(X_test)

# Classification report for detailed evaluation
print(classification_report(Y_test, Y_prediction))

# Check predictions and print whether the tumor is Malignant or Benign
if Y_prediction[0] == 1:
    print("The tumor is likely Malignant.")
else:
    print("The tumor is likely Benign.")

# More models for comparison

models = {
    "Decision Tree": DecisionTreeClassifier(), "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(Y_test, preds):.2f}")

# Evaluate LR accuracy
accuracy = accuracy_score(Y_test, Y_prediction)
print(f'Logistic Regression Accuracy: {accuracy: .2f}')

### Model Perfomance Comparison ###

# Logistic Regression ROC

y_prob_lr = lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(Y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Train a Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

# Train Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

# Predict probabilities for Decision Tree ROC
y_prob_dt = dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(Y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Predict probablities for Random Forest ROC
y_prob_rf = dt.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf,tpr_rf )

# Cross-Validation Scores
cv_scores = cross_val_score(LogisticRegression(max_iter=10000), X_scaled, y, cv=5)
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.2f}")


### Visualisations ###

# Look for and visualise any missing values
sns.heatmap(data.isnull())

sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()  # Missing Values

# Class Distribution
sns.countplot(x='diagnosis', data=data)
plt.title('Diagnosis Distribution')
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()

# Correlated Heatmap
plt.figure(figsize=(12,10))
correlation = data.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
top_corr_features = correlation.index[abs(correlation["diagnosis"]) > 0.5]
plt.figure(figsize=(10, 8))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="YlGnBu")
plt.title('Top Correlated Features with Diagnosis')
plt.show()  # Top Features

# Model Performance

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# ROC Curve

y_prob = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Feature importance
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

# Feature importance plot
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10])
plt.title("Top 10 Feature Importances from Random Forest")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Merged Plots

plt.figure()

# Logistic Regression
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
         label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# Decision Tree
plt.plot(fpr_dt, tpr_dt, color='green', lw=2,
         label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')

# Random Forest
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2,
         label=f'Random Forest (AUC = {roc_auc_rf: .2f} )')

# Baseline
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid()
plt.show()

#### Export Function for Reuse! ####
