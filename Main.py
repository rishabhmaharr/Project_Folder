# Student Performance Analyzer (SPA)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------
# Load dataset from the same folder
# ----------------------------------------

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'StudentsPerformance.csv')
df = pd.read_csv(csv_path)

# ----------------------------------------
# Exploratory Data Analysis (EDA)
# ----------------------------------------

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['math score'], kde=True, color='skyblue')
plt.title('Math Score Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['reading score'], kde=True, color='lightgreen')
plt.title('Reading Score Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['writing score'], kde=True, color='salmon')
plt.title('Writing Score Distribution')

plt.tight_layout()
plt.show()

# Boxplot: Gender vs Math Score
plt.figure(figsize=(6, 4))
sns.boxplot(x='gender', y='math score', data=df)
plt.title("Gender vs Math Score")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Score Correlation Matrix")
plt.show()

# ----------------------------------------
# Feature Engineering
# ----------------------------------------

# Add average score & result column
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['result'] = df['average_score'].apply(lambda x: 'pass' if x >= 40 else 'fail')

# Backup original data for insights
original_df = df.copy()

# Label Encode categorical features
le = LabelEncoder()
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ----------------------------------------
# Classification Models (Logistic & KNN)
# ----------------------------------------

X = df.drop(['average_score', 'result'], axis=1)
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("\n=== K-Nearest Neighbors ===")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))

# ----------------------------------------
# K-Means Clustering
# ----------------------------------------

# Elbow Method to determine optimal k
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df[['math score', 'reading score', 'writing score']])
    inertia.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df[['math score', 'reading score', 'writing score']])

# Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='math score', y='reading score', hue='cluster', data=df, palette='Set1')
plt.title("K-Means Clustering (3 Clusters)")
plt.show()

# ----------------------------------------
# Bonus Insights
# ----------------------------------------

print("\n=== Bonus Insights ===")

# Test Preparation Course
prep_stats = original_df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean()
print("\nTest Preparation Course Impact:\n", prep_stats)

# Lunch Type
lunch_stats = original_df.groupby('lunch')[['math score', 'reading score', 'writing score']].mean()
print("\nLunch Type Impact:\n", lunch_stats)

# Parental Education
parent_stats = original_df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean()
print("\nParental Education Impact:\n", parent_stats)
