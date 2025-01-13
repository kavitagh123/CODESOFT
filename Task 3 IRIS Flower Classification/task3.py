import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

#Load the Iris dataset
df=pd.read_csv('IRIS.csv')
print(df)

print(df.head())

#Data Cleaning
print(df.isnull().sum())

print(df.info())

print(df.duplicated().sum())

print(df.dropna())

print(df.drop_duplicates(inplace=True))

print(df.describe())

#Data Visuvalization
#Create a sactter plot
plt.figure(figsize=(10,6))
sns.scatterplot(x='sepal_length',y='sepal_width',hue='species',data=df)
plt.title('Sactter PLot of Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepla Width')
plt.show()

#Create a pair plot
sns.pairplot(df,hue='species')
plt.title('Pair plot of Iris Dataset')
plt.show()

# 1. Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# 2. Load the Dataset
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
'''print(df)

# Display first few rows of the dataset
print(df.head())

# Summary statistics of the dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Count of each species
print(df['species'].value_counts())

# 3. Exploratory Data Analysis (EDA)
# Check dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Visualizing the pairplot
sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()

# Compute correlation matrix excluding the 'species' column
correlation_matrix = df.drop('species', axis=1).corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()'''

# 4. Data Preprocessing
# Define features and target
X = df.drop('species', axis=1)
y = df['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)

# 5. Train the Model
# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the Model
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blue', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

