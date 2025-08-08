#  ZION TECH HUB - Breast Cancer Detection Project 

#  File: train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Step 1: Load and Prepare Data ---
df = pd.read_csv("breast_cancer.csv")
df.drop(columns=['id'], errors='ignore', inplace=True)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Step 2: EDA - Boxplot of Selected Features ---
features_to_plot = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[features_to_plot], palette="pastel")
plt.title("Boxplot of Tumor Features")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 3: EDA - Diagnosis Class Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x="diagnosis", data=df)
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.title("Distribution of Diagnosis Classes")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
for index, value in enumerate(df["diagnosis"].value_counts()):
    plt.text(index, value + 5, str(value), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# --- Step 4: EDA - Correlation Heatmap ---
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.show()

# --- Step 5: Model Training and Comparison ---
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

# --- Step 6: Bar Charts of Model Performance ---
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["#2a9d8f", "#264653", "#e76f51", "#f4a261"]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Performance Comparison of ML Models", fontsize=18)

for i, metric in enumerate(metrics):
    row, col = i // 2, i % 2
    ax = axs[row][col]
    bars = ax.bar(results_df["Model"], results_df[metric], color=colors[i])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center', fontweight='bold')
    ax.set_title(f"{metric} by Model", fontsize=14)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df["Model"], rotation=45)

plt.tight_layout()
plt.show()

# --- Step 7: Confusion Matrix for Best Model ---
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix: {best_model_name}")
plt.grid(False)
plt.tight_layout()
plt.show()

# --- Step 8: Hyperparameter Tuning (Grid Search) ---
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# --- Step 9: Visualize Grid Search Results ---
cv_results = pd.DataFrame(grid_search.cv_results_)

pivot_table = cv_results.pivot_table(
    index='param_n_estimators',
    columns='param_max_depth',
    values='mean_test_score'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("GridSearchCV F1-Score Heatmap\n(n_estimators vs max_depth)")
plt.xlabel("Max Depth")
plt.ylabel("N Estimators")
plt.tight_layout()
plt.show()

# --- Step 10: Save Final Best Model ---
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X, y)
os.makedirs("breast_cancer_ml_project/models", exist_ok=True)
joblib.dump(best_rf_model, "breast_cancer_ml_project/models/best_rf_model.pkl")
print(" Final Random Forest model saved!")

# --- Step 11: Summary  ---
print("\n FINDINGS & INSIGHTS:")
print("\n- Tumor characteristics vary widely, especially in radius and area, indicating potential outliers.")
print("- Benign tumors were more common than malignant ones, which can bias model training.")
print("- Random Forest consistently outperformed other models and achieved excellent recall.")
print("- The confusion matrix revealed very few false negatives, which is crucial in healthcare.")
print("- GridSearch helped optimize the model further by tuning depth and number of trees.")
print("- This model can be a powerful tool for early detection and assist clinicians in diagnosis.")
