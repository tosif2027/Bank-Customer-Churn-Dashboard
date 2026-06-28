# ==========================================================
# BANK CUSTOMER CHURN PREDICTION MODEL
# ==========================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("European_Bank.csv")

df.columns = df.columns.str.strip()

# ==========================================================
# DATA CLEANING
# ==========================================================

df.drop_duplicates(inplace=True)

df.drop(
    columns=[
        "Year",
        "CustomerId",
        "Surname"
    ],
    inplace=True
)

# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])

df["Geography"] = encoder.fit_transform(df["Geography"])

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("Exited", axis=1)

y = df["Exited"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=200
    )

}

best_model = None

best_accuracy = 0

# ==========================================================
# TRAINING
# ==========================================================

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    print("=" * 60)

    print(name)

    print("Accuracy :", round(accuracy * 100, 2), "%")

    print()

    print(confusion_matrix(
        y_test,
        prediction
    ))

    print()

    print(classification_report(
        y_test,
        prediction
    ))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    best_model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("=" * 60)

print("Best Accuracy :", round(best_accuracy * 100, 2), "%")

print("Model Saved Successfully")

print("=" * 60)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns

if isinstance(best_model, RandomForestClassifier):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": best_model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig("feature_importance.png")
    plt.close()

# ==========================================================
# ACCURACY COMPARISON
# ==========================================================

accuracy_dict = {}

for name, model in models.items():

    pred = model.predict(X_test)

    accuracy_dict[name] = accuracy_score(
        y_test,
        pred
    )

plt.figure(figsize=(8,5))

sns.barplot(
    x=list(accuracy_dict.keys()),
    y=list(accuracy_dict.values())
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.ylim(0.70,1)

plt.savefig("accuracy_comparison.png")
plt.close()

# ==========================================================
# CONFUSION MATRIX HEATMAP
# ==========================================================

pred = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()

