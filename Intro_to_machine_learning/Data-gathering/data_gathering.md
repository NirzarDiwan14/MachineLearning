# Data Gathering for Machine Learning Models

## 1. Data Gathering Methods

### 1.1 Via CSV Files

* Load data from local or cloud-stored CSV files
* Common in small to medium datasets
* Easy to inspect and preprocess

---

### 1.2 Via JSON / SQL

* **JSON**: Used for semi-structured data (logs, APIs)
* **SQL**: Used for structured data stored in databases

**Benefits**

* Reliable storage
* Easy querying
* Scales well

---

### 1.3 Via APIs

* Fetch real-time or live data
* Common for weather, finance, social media, etc.

**Tools**

* Requests (Python)
* RapidAPI (free APIs for data)

---

### 1.4 Via Web Scraping

* Extract data from websites

**Tools**

* BeautifulSoup
* Scrapy
* Selenium

**Use cases**

* Product prices
* Reviews
* News articles

---

## 2. RapidAPI

**What it is**
A platform that provides free and paid APIs for data.

**Why use it**

* Quick access to datasets
* No need to build scrapers
* Easy API integration

---

## 3. Bias–Variance Tradeoff

### Bias

**What it is**
Difference between predicted values and actual values.

**Meaning**

* High bias → model is too simple
* Causes underfitting

---

### Variance

**What it is**
Difference between training accuracy and test accuracy.

**Meaning**

* High variance → model is too complex
* Causes overfitting

---

## 4. Underfitting vs Overfitting

### Underfitting (High Bias, Low Variance)

* Performs poorly on training data
* Performs poorly on test data
* Model is too simple

---

### Overfitting (Low Bias, High Variance)

* Performs very well on training data
* Performs poorly on test data
* Model memorizes training data

---

## 5. Summary

* Data can be collected from CSV, JSON, SQL, APIs, or web scraping.
* RapidAPI provides free APIs for quick data access.
* High bias → underfitting.
* High variance → overfitting.
* Balance bias and variance for a good model.

---

**One-liner:**

> Good ML models start with good data and balanced bias–variance.



# Cross Validation in Machine Learning

## What is Cross Validation?

Cross validation is a technique used to evaluate and tune machine learning models.

**Why it is needed**

* We split data into **training** and **testing** sets.
* During **hyperparameter tuning**, we should NOT touch the test data.
* So we further split the training data into:

  * Training subset
  * Validation subset

By repeating this process and combining results, we select better hyperparameters and build a more reliable model.

---

## Types of Cross Validation Techniques

### 1. Leave-One-Out Cross Validation (LOOCV)

**What it is**
Uses all data points except one for training, and the remaining one for validation.

**How it works**

* For N data points:

  * Train on N−1 points
  * Test on 1 point
* Repeat this N times

**Pros**

* Uses maximum data for training
* Low bias

**Cons**

* Very slow for large datasets
* High variance

**When to use**

* Very small datasets

---

### 2. K-Fold Cross Validation

**What it is**
Splits data into K equal parts (folds).

**How it works**

* Train on K−1 folds
* Validate on 1 fold
* Repeat K times
* Average the results

**Pros**

* Good balance of bias and variance
* Efficient

**Cons**

* Costly for very large datasets

**When to use**

* Most common general-purpose method

---

### 3. Stratified K-Fold

**What it is**
K-Fold that preserves class distribution in each fold.

**Why needed**

* Prevents class imbalance issues

**Pros**

* More reliable for classification
* Better class representation

**When to use**

* Classification problems

---

### 4. Time Series Cross Validation

**What it is**
Used when data has a time order.

**How it works**

* Train on past data
* Validate on future data
* Never shuffle data

**Pros**

* Realistic evaluation
* Avoids data leakage

**When to use**

* Stock prices
* Sales forecasting

---

### 5. Hold-Out Validation

**What it is**
Single split of data into training and validation.

**How it works**

* 70% train
* 30% validate

**Pros**

* Simple and fast

**Cons**

* Unstable results
* Depends on random split

**When to use**

* Very large datasets

---

## Summary

* Use cross validation to tune hyperparameters.
* Never touch test data during tuning.
* K-Fold is the most common method.
* Stratified K-Fold is best for classification.
* Time series CV is used for ordered data.
* Hold-out is simple but less reliable.

---

**One-liner:**

> Cross validation helps tune models safely without leaking test data.

