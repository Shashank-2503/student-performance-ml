import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
data = pd.read_csv("student_performance.csv", sep=";")


# Show available columns
print("Available columns in dataset:")
print(data.columns)

# 2. Handle missing values
data = data.fillna(data.median(numeric_only=True))

# 3. Encode categorical variables
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# ---- Adjust this depending on available target ----
if "G3" in data.columns:
    target = "G3"
elif "Grade" in data.columns:
    target = "Grade"
else:
    raise ValueError("Could not find target column (like G3 or Grade) in dataset")

# 4. Define features & target
X = data.drop(target, axis=1)
y = data[target]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc:.2f}")
