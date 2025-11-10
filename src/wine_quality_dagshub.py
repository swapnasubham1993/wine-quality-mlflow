import os
os.environ["MLFLOW_ENABLE_LOGGED_MODEL_CREATION"] = "false"
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='swapnasubham1993', repo_name='wine-quality-mlflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/swapnasubham1993/wine-quality-mlflow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Define the params for RF model
max_depth = 8
n_estimators = 9

# Mention Your experiment
mlflow.set_experiment("Wine_Quality_Experiment")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion Matrix.png")

    # Log the confusion matrix as an artifact
    mlflow.log_artifact("Confusion Matrix.png")
    mlflow.log_artifact(__file__)

    # Set Tags
    mlflow.set_tags({"Author": "Anik", "model_type": "RandomForest", "dataset": "Wine"})

    # Log the model
    mlflow.sklearn.log_model(rf, "RandomForestModel", registered_model_name=None)
    
    print(f"Accuracy: {accuracy}")


