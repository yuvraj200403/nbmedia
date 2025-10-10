import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🚀 Step 1: Load full dataset
df = pd.read_csv("gesture_data.csv")

# 🧹 Step 2: Keep only labels 0, 1, 2 (e.g., Hello, Thanks, I Love You)
df = df[df['label'].isin([0, 1, 2,])]

# 💾 Save cleaned data
df.to_csv("gesture_data_clean.csv", index=False)
print("✅ Cleaned dataset saved as 'gesture_data_clean.csv'")

# 🧠 Step 3: Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# 🎯 Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 🛠️ Step 5: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Step 6: Save the trained model
joblib.dump(model, "gesture_model.pkl")
print("\n✅ Model saved as 'gesture_model.pkl'")

# 📈 Step 7: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔍 Accuracy: {accuracy:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 📌 Optional: show gesture count
print("\n🔢 Final Sample Count per Label:")
print(df['label'].value_counts())
