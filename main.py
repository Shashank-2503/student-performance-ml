import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_and_preprocess_data(filepath):
    """
    Loads data from a CSV, handles missing values, and encodes categorical features.
    """
    # Load data
    data = pd.read_csv(filepath, sep=";")
    print("Available columns in dataset:")
    print(data.columns)

    # Handle missing values by filling with the median for numeric columns
    data = data.fillna(data.median(numeric_only=True))

    # Encode categorical variables
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Define features and target
    target = "G3"  # Final grade
    X = data.drop(target, axis=1)
    y = data[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.
    """
    print("\nTraining the model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints the accuracy score.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.2f}")


if __name__ == "__main__":
    # Define the path to the dataset
    dataset_path = "student_performance.csv"

    # Run the machine learning pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)