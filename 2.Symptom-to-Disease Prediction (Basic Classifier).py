# 1. Install Required Libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Load Dataset
import pandas as pd

# Example CSV structure:
# fever,cough,headache,nausea,disease
# 1,1,0,0,Flu

data = pd.read_csv("symptom_disease.csv")

print(data.head())

# 3. Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode disease labels
le = LabelEncoder()
data['disease_encoded'] = le.fit_transform(data['disease'])

# Features and target
X = data.drop(['disease', 'disease_encoded'], axis=1)
y = data['disease_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training size: {X_train.shape}")

# 4. Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = LogisticRegression(max_iter=1000)

# Train
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# 5. Random Forest Model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# 6. Compare Models
print("Model Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
``

# 7. Predict New Patient
# Example new patient symptoms
# Must match training feature order
new_patient = [[1, 0, 1, 0]]  # example: fever=1, cough=0, headache=1, nausea=0

prediction = rf_model.predict(new_patient)

# Convert back to disease name
predicted_disease = le.inverse_transform(prediction)

print("Predicted Disease:", predicted_disease[0])

# 8. Optional: Feature Importance (Random Forest)
import matplotlib.pyplot as plt
import seaborn as sns

importances = rf_model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Symptom Importance")
plt.show()

