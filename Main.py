# Data Preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Project_Folder\StudentsPerformance (2).csv")
print(df.isnull().sum())
print(df.duplicated().sum())

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

# Exploratory Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt

df[['math score', 'reading score', 'writing score']].hist(bins=20, figsize=(10, 5))
plt.show()

sns.boxplot(x='gender', y='math score', data=df)
plt.title("Math Score by Gender")
plt.show()

print(df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean())

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature Engineering
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['result'] = df['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')
df['result_label'] = df['result'].map({'Pass': 1, 'Fail': 0})

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df.drop(['result', 'result_label', 'average_score'], axis=1)
y = df['result_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("KNN:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Clustering
from sklearn.cluster import KMeans

score_data = df[['math score', 'reading score', 'writing score']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(score_data)

plt.figure(figsize=(8, 5))
sns.scatterplot(x='math score', y='reading score', hue='cluster', data=df, palette='Set1')
plt.title("K-Means Clustering by Scores")
plt.show()

# Bonus Analysis
df.groupby('lunch')['average_score'].mean().plot(kind='bar', title='Avg Score by Lunch Type')
plt.ylabel('Average Score')
plt.show()

df.groupby('parental level of education')['average_score'].mean().plot(kind='bar', title='Avg Score by Parental Education')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.show()