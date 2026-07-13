
# Toxic Comments Exploratory Data Analysis (EDA)

## Project Overview

This project performs Exploratory Data Analysis (EDA) on the Jigsaw Toxic Comment Classification dataset. The goal is to understand the dataset before building a machine learning model.

---

## Dataset

- Dataset: Jigsaw Toxic Comment Classification
- Source: Kaggle
- Rows: 159,571
- Columns: 8

---

## Libraries Used

- Python
- Pandas
- Matplotlib
---

## Tasks Performed

- Loaded the dataset
- Checked shape and data types
- Checked missing values
- Checked duplicate values
- Analyzed toxicity labels
- Calculated percentage of each toxicity category
- Created a new feature called `comment_length`
- Compared average comment length across toxicity classes
- Created `toxicity_count`
- Sorted comments by toxicity level
- Performed correlation analysis
- Created a bar graph for visualisation
---

## Key Findings

- The dataset contains no missing values.
- The dataset contains no duplicate rows.
- Approx 9.6% of comments are toxic.
- Only 31 comments belong to all six toxicity categories.
- Most comments are non-toxic.

---

## Project Structure

```
Toxic-Comments-EDA/
requirements.txt
toxic_eda
train.csv.zip
```

---

## Future Work

This EDA project will be used as the foundation for building a Toxic Comment Classifier using TF-IDF and Logistic Regression.

## Skills Demonstrated

- Data loading and inspection
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- GroupBy operations
- Correlation analysis
- Data visualization using Matplotlib
- Python programming with Pandas
