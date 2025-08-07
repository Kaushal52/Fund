import pandas as pd

file_path = "/Users/kaushalkhatu/Desktop/Mutual fund/MF.csv"

try:
    df = pd.read_csv(file_path)
    print(df.head())  # shows first 5 rows
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")

import pandas as pd

file_path = "/Users/kaushalkhatu/Desktop/Mutual fund/MF.csv"

df = pd.read_csv(file_path)
print("Column names:")
print(df.columns.tolist())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
file_path = "/Users/kaushalkhatu/Desktop/Mutual fund/MF.csv"
df = pd.read_csv(file_path)

# Step 1: Create target variable (Buy if avg returns > 10%)
df['average_return'] = df[['returns_1yr', 'returns_3yr', 'returns_5yr', 'returns_10yr']].mean(axis=1)
df['label'] = df['average_return'].apply(lambda x: 'Buy' if x > 10 else 'Not Buy')

# Step 2: Encode categorical features
le_rating = LabelEncoder()
le_category = LabelEncoder()
le_sub_category = LabelEncoder()

df['rating_encoded'] = le_rating.fit_transform(df['rating'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['sub_category_encoded'] = le_sub_category.fit_transform(df['sub_category'])

# Step 3: Define features and target
features = ['rating_encoded', 'category_encoded', 'sub_category_encoded', 
            'returns_1yr', 'returns_3yr', 'returns_5yr', 'returns_10yr']
X = df[features]
y = df['label']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
   
}

# Step 6: Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))

# Step 7: Feature importance from Random Forest
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
print("\nFeature Importances (Random Forest):")
print(feature_importance_df.sort_values(by="Importance", ascending=False))

from sklearn.metrics import log_loss

# Binary encode labels for log_loss (Buy=1, Not Buy=0)
y_train_bin = y_train.map({'Buy': 1, 'Not Buy': 0})
y_test_bin = y_test.map({'Buy': 1, 'Not Buy': 0})

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]  # Prob of class 'Buy'
    
    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test_bin, probs)

    print(f"\n🔍 Model: {name}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"❌ Log Loss: {loss:.4f}")
    print("📋 Classification Report:")
    print(classification_report(y_test, preds))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# 🔹 Load Data
df = pd.read_csv('/Users/kaushalkhatu/Desktop/Mutual fund/MF.csv')

# 🔹 Drop missing values and reset index
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 🔹 Define target
df['target'] = df['returns_3yr'].apply(lambda x: 'Buy' if x > df['returns_3yr'].mean() else 'Not Buy')

# 🔹 Encode categorical features
le_rating = LabelEncoder()
le_category = LabelEncoder()
le_sub_category = LabelEncoder()

df['rating_encoded'] = le_rating.fit_transform(df['rating'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['sub_category_encoded'] = le_sub_category.fit_transform(df['sub_category'])

# 🔹 Feature & Target
X = df[['rating_encoded', 'category_encoded', 'sub_category_encoded', 'returns_1yr', 'returns_3yr', 'returns_5yr', 'returns_10yr']]
y = df['target']

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 Evaluate accuracy (for checking overall model performance)
accuracy = accuracy_score(y_test, model.predict(X_test))

# 🔹 Take User Input for Scheme
scheme_input = input("\nEnter the scheme name: ").strip()

# 🔹 Find scheme
matched = df[df['scheme_name'].str.lower() == scheme_input.lower()]

if matched.empty:
    print("⚠️ Scheme not found. Please check the name.")
else:
    row = matched.iloc[0]
    
    # Prepare feature for this specific scheme
    input_df = pd.DataFrame([[
        le_rating.transform([row['rating']])[0],
        le_category.transform([row['category']])[0],
        le_sub_category.transform([row['sub_category']])[0],
        row['returns_1yr'],
        row['returns_3yr'],
        row['returns_5yr'],
        row['returns_10yr']
    ]], columns=['rating_encoded', 'category_encoded', 'sub_category_encoded',
                 'returns_1yr', 'returns_3yr', 'returns_5yr', 'returns_10yr'])
    
    # Prediction for the scheme
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of 'Buy'

    # Actual label for the scheme (based on the returns_3yr column)
    input_actual = 'Buy' if row['returns_3yr'] > df['returns_3yr'].mean() else 'Not Buy'
    
    # Calculate log loss for this scheme
    input_log_loss = log_loss([input_actual], [probability], labels=['Buy', 'Not Buy'])

    # Display the results
    print(f"\n🔍 Scheme: {row['scheme_name']}")
    print(f"📈 Prediction: {'✅ BUY' if prediction == 'Buy' else '❌ NOT BUY'}")
    print(f"   10-Year: {row['returns_10yr']}%")

