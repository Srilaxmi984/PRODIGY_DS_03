# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Step 1: Load the dataset
url = "bank.csv"  # Ensure the correct file path
df = pd.read_csv(url, delimiter=';')  # UCI dataset uses semicolon as delimiter

# Step 2: Data Preprocessing
# Checking for missing values
print("Missing values per column:\n", df.isnull().sum())

# Convert target variable 'y' to numerical (0 or 1)
label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])  

# Get actual column names
print("Dataset columns:", df.columns)

# Define categorical columns & ensure they exist in the dataset
categorical_cols = [col for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                                    'contact', 'month', 'day_of_week', 'poutcome'] if col in df.columns]

# Apply label encoding to categorical columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("Categorical columns encoded:", categorical_cols)

# Step 3: Split the data into training and testing sets
X = df.drop('y', axis=1)  # Features
y = df['y']  # Target variable

# Splitting data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True, rounded=True)
plt.show()
