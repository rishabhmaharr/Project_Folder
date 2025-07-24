# 🎓 Student Performance Analyzer (SPA)

A Python-based data analysis tool that explores and models student performance data using **Pandas**, **Seaborn**, and **scikit-learn**. It performs EDA, classification (Logistic Regression & KNN), clustering (K-Means), and provides bonus insights like the impact of lunch, test preparation, and parental education.

---

## 📁 Dataset

The project uses the [StudentsPerformance.csv](StudentsPerformance.csv) dataset, which contains the following columns:

- Gender
- Race/ethnicity
- Parental level of education
- Lunch
- Test preparation course
- Math score
- Reading score
- Writing score

---

## 🔧 Features

- 📊 **EDA** with histograms, boxplots, and correlation heatmaps
- 🤖 **Classification** using:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- 🔍 **Clustering** using K-Means with Elbow Method
- 🎁 **Bonus Insights**:
  - Lunch type vs performance
  - Test preparation vs scores
  - Parental education impact

---

## 🛠 Requirements

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
