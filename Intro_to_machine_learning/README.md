# Table of Contents

1. [Types of Machine Learning](#types-of-machine-learning)
2. [Batch (Offline) Learning vs Online Learning](#batch-offline-learning-vs-online-learning)
3. [Instance-Based vs Model-Based Learning](#instance-based-vs-model-based-learning)
4. [Challenges of Machine Learning](#challenges-of-machine-learning)
5. [Applications of Machine Learning in B2B Market](#applications-of-machine-learning-in-b2b-market)
6. [Machine Learning Development Life Cycle (MLDLC)](#machine-learning-development-life-cycle-mldlc)
7. [Key Job Roles in Data & ML](#key-job-roles-in-data--ml)

---

# Types of Machine Learning

## 1. Supervised Learning

**What it is**
Learning from labeled data (input + correct output).

**Goal**
Predict output for new, unseen data.

**Main types**

* **Regression** – output is numerical
  Examples: house price, age prediction

* **Classification** – output is categorical
  Examples: spam detection, gender classification

**Common algorithms**

* Linear Regression
* Logistic Regression
* Decision Trees
* k-NN
* Neural Networks

**When to use**

* You have labeled data
* You want direct predictions

---

## 2. Unsupervised Learning

**What it is**
Learning from unlabeled data to find hidden patterns.

**Goal**
Discover structure or relationships in data.

**Main types**

* **Clustering** – group similar data points
  Examples: customer segmentation

* **Dimensionality Reduction** – reduce number of features
  Examples: PCA, t-SNE

* **Anomaly Detection** – find rare or unusual points
  Examples: fraud detection

* **Association Rule Learning** – find item relationships
  Examples: market basket analysis

**Common algorithms**

* K-Means
* Hierarchical Clustering
* DBSCAN
* PCA

**When to use**

* No labeled data
* Exploratory analysis

---

## 3. Semi-Supervised Learning

**What it is**
Uses a small amount of labeled data with a large amount of unlabeled data.

**Why it matters**
Labeled data is expensive and hard to get.

**How it works**

* Train on labeled data
* Predict labels for unlabeled data
* Retrain using both

**When to use**

* Few labels, lots of raw data
* Labeling cost is high

---

## 4. Reinforcement Learning

**What it is**
An agent learns by interacting with an environment.

**Key elements**

* Agent
* Environment
* Action
* Reward / Penalty
* Policy

**Goal**
Maximize total reward over time.

**Examples**

* Game playing (Chess, Go)
* Robotics
* Self-driving cars

**When to use**

* Sequential decision making
* No fixed dataset

---

## 5. Quick Comparison

| Type            | Data        | Output Known? | Goal                    |
| --------------- | ----------- | ------------- | ----------------------- |
| Supervised      | Labeled     | Yes           | Predict outputs         |
| Unsupervised    | Unlabeled   | No            | Find patterns           |
| Semi-Supervised | Mixed       | Partially     | Improve with few labels |
| Reinforcement   | Interaction | No            | Maximize rewards        |

---

## 6. Summary

* **Supervised**: Learn with labels → predict values or classes.
* **Unsupervised**: Learn without labels → discover structure.
* **Semi-Supervised**: Learn with few labels → boost accuracy.
* **Reinforcement**: Learn by trial and error.

---

**One-liner:**

> Supervised predicts, Unsupervised discovers, Semi-supervised bridges, Reinforcement decides.


# Batch (Offline) Learning vs Online Learning

## 1. Batch (Offline) Learning

**What it is**
You train your model on a fixed dataset, freeze it, and deploy it to production.

**Workflow**

1. Collect historical data
2. Train model offline
3. Deploy model to production
4. Use it for predictions
5. Periodically retrain with new + old data

**Pros**

* Simple to implement
* Stable and predictable behavior
* Easy to debug and reproduce
* Works well when data distribution is stable

**Cons / Problems**

* Model becomes stale as new data arrives
* Cannot learn from new data in real time
* Requires full retraining to update
* High compute cost for frequent retraining
* Downtime or deployment risk during updates

**When to use**

* Data distribution is stable
* Concept drift is low or slow
* Latency requirements are not extreme
* You can afford periodic retraining (e.g., daily, weekly)

---

## 2. Online Machine Learning

**What it is**
The model is updated continuously as new data arrives, usually in small chunks or one sample at a time.

**Workflow**

1. Model starts with initial weights
2. New data arrives in a stream
3. Model updates itself incrementally
4. Model keeps adapting over time

**Pros**

* Adapts to new data in real time
* Handles concept drift well
* No need for full retraining
* Lower memory usage (processes small chunks)

**Cons / Disadvantages**

* More complex to design and monitor
* Harder to debug and reproduce
* Easy to corrupt the model with bad data
* Fewer mature tools compared to batch ML

**When to use**

* Concept drift is present
* Real-time or near-real-time learning is required
* Data arrives as a stream
* Fast adaptation is critical (e.g., fraud detection, recommendations)

---

## 3. Out-of-Core Learning

**What it is**
Used when the dataset is too large to fit into memory. The data is processed in chunks, but training still happens offline.

**Key idea**

* Split data into small batches
* Train incrementally on each batch
* Do NOT update the model in production

**How it differs from online learning**

* Out-of-core: offline + chunked
* Online: live + continuous

**Example tools**

* scikit-learn (`partial_fit`)
* Dask
* Spark MLlib

---

## 4. Tools for Online Learning

* **River** – modern Python library for online ML
* **Vowpal Wabbit** – fast online learning system
* **scikit-learn** – `partial_fit()` for incremental training

---

## 5. When to Use What?

| Scenario                        | Batch Learning | Online Learning |
| ------------------------------- | -------------- | --------------- |
| Stable data                     | ✅              | ❌               |
| Concept drift                   | ❌              | ✅               |
| Real-time adaptation needed     | ❌              | ✅               |
| Simple system                   | ✅              | ❌               |
| Huge dataset (won't fit in RAM) | ⚠️ Out-of-core | ❌               |
| Streamed data                   | ❌              | ✅               |

---

## 6. Simple Architecture Diagrams

### Batch Learning

```
[Historical Data] ---> [Offline Training] ---> [Trained Model]
                                              |
                                              v
                                         [Production]
```

### Online Learning

```
[Data Stream] ---> [Model Update] ---> [Updated Model]
                         |
                         v
                    [Predictions]
```

---

## 7. Summary

* **Batch Learning**: Simple, stable, but becomes stale and costly to retrain.
* **Online Learning**: Adaptive, efficient, but complex and risky if not monitored.
* **Out-of-Core Learning**: Offline training on very large datasets using chunks.

---

## 8. Practical Tips

* Start with batch ML unless you truly need online learning.
* Monitor data drift and model performance.
* Use online ML only when adaptation speed matters.
* Protect online models with validation checks and rollback.

---

**One-liner:**

> Batch ML is easier and safer. Online ML is powerful but dangerous if misused.


# Instance-Based vs Model-Based Learning

## 1. Instance-Based Learning

**What it is**
The model stores training examples and makes predictions by comparing new data points to stored instances.

**How it works**

1. Store all (or many) training samples
2. When a new point arrives, find similar instances
3. Predict based on neighbors

**Common examples**

* k-Nearest Neighbors (k-NN)
* Locally Weighted Regression

**Pros**

* Simple and intuitive
* No real training phase
* Adapts easily to new data
* Works well for small datasets

**Cons / Problems**

* High memory usage
* Slow at prediction time
* Sensitive to noisy data
* Poor scalability

**When to use**

* Small to medium datasets
* Irregular or complex decision boundaries
* Fast prototyping

---

## 2. Model-Based Learning

**What it is**
The algorithm learns a general model (function) from data instead of storing all examples.

**How it works**

1. Train a model on data
2. Learn parameters
3. Discard most training data
4. Use the model for predictions

**Common examples**

* Linear Regression
* Logistic Regression
* Decision Trees
* Neural Networks

**Pros**

* Fast predictions
* Low memory usage
* Scales well to large datasets
* More robust to noise

**Cons / Problems**

* Requires training
* Harder to update incrementally
* Can underfit complex patterns

**When to use**

* Large datasets
* Real-time prediction systems
* Production environments

---

## 3. Key Differences

| Feature         | Instance-Based | Model-Based |
| --------------- | -------------- | ----------- |
| Stores data     | Yes            | No          |
| Training time   | Very low       | High        |
| Prediction time | High           | Low         |
| Memory usage    | High           | Low         |
| Scalability     | Poor           | Good        |

---

## 4. Simple Diagrams

### Instance-Based

```
[Training Data] ---> [Stored Instances]
                          |
                          v
                     [Prediction]
```

### Model-Based

```
[Training Data] ---> [Training] ---> [Model]
                                     |
                                     v
                                [Prediction]
```

---

## 5. Summary

* **Instance-Based**: Stores data, slow predictions, flexible.
* **Model-Based**: Learns a model, fast predictions, scalable.

---

**One-liner:**

> Instance-based learning memorizes. Model-based learning generalizes.

# Challenges of Machine Learning

## 1. Data Collection

**Problem**
Getting enough useful data is hard.

**Issues**

* Data may be expensive to collect
* Privacy and legal restrictions
* Data may come from many sources

---

## 2. Insufficient / Labeled Data

**Problem**
ML models need lots of labeled data.

**Issues**

* Labeling is time-consuming
* Labeling is costly
* Few labels → poor model accuracy

---

## 3. Non-Representative Data

**Problem**
Training data does not match real-world data.

**Issues**

* Biased predictions
* Poor generalization

---

## 4. Poor-Quality Data

**Problem**
Noisy, missing, or wrong data.

**Issues**

* Garbage in → garbage out
* Lower model performance

---

## 5. Irrelevant Features

**Problem**
Using useless or misleading features.

**Issues**

* Confuses the model
* Increases training time
* Reduces accuracy

---

## 6. Overfitting

**Problem**
Model memorizes training data.

**Result**

* High training accuracy
* Low test accuracy

---

## 7. Underfitting

**Problem**
Model is too simple.

**Result**

* Poor training accuracy
* Poor test accuracy

---

## 8. Software Integration

**Problem**
Deploying ML models into real systems.

**Issues**

* Compatibility problems
* Scaling issues
* Monitoring and updates

---

## 9. Offline Learning / Deployment

**Problem**
Models become stale after deployment.

**Issues**

* Needs retraining
* Downtime risks
* High compute cost

---

## 10. Cost Involved

**Problem**
ML is expensive.

**Costs**

* Data collection
* Labeling
* Compute (training + inference)
* Infrastructure

---

## 11. Summary

* Data problems are the biggest challenge.
* Bad data → bad models.
* Deployment and cost matter in production.
* Balance overfitting and underfitting.

---

**One-liner:**

> Most ML failures come from bad data, bad features, or bad deployment.
# Applications of Machine Learning in B2B Market

## 1. Retail (B2B)

**Use cases**

* Demand forecasting
* Inventory optimization
* Dynamic pricing
* Customer segmentation

**Benefits**

* Reduce stockouts and overstock
* Improve supply chain efficiency
* Increase revenue

---

## 2. Sentiment Analysis (Twitter / Social Media)

**What it does**
Analyzes public opinion about brands, products, or services.

**Use cases**

* Brand monitoring
* Customer feedback analysis
* Crisis detection

**Benefits**

* Understand customer mood
* Improve marketing strategy
* Early issue detection

---

## 3. Recommendation Engines

**What it does**
Suggests products or services based on user behavior.

**Use cases**

* Cross-selling and upselling
* Personalized offers
* Content recommendations

**Benefits**

* Increase sales
* Improve customer experience
* Better retention

---

## 4. Manufacturing (e.g., Tesla)

**Use cases**

* Robot automation
* Quality inspection
* Predictive maintenance

**Anomaly Detection**

* Detect defective parts
* Detect abnormal machine behavior

**Benefits**

* Reduce downtime
* Improve product quality
* Lower maintenance costs

---

## 5. Fraud & Anomaly Detection (Industrial)

**What it does**
Finds unusual patterns in data.

**Use cases**

* Equipment fault detection
* Manufacturing defects
* Process monitoring

**Benefits**

* Prevent failures
* Improve safety
* Save costs

---

## 6. Summary

* Retail: forecasting, pricing, inventory.
* Sentiment: brand and customer insights.
* Recommendations: sales growth.
* Manufacturing: automation and defect detection.
* Anomaly detection: fault and fraud prevention.

---

**One-liner:**

> ML in B2B improves efficiency, reduces cost, and enables smarter decisions.


# Machine Learning Development Life Cycle (MLDLC)

## 1. Frame the Problem

**What it means**
Define what you want to solve.

**Key points**

* Business objective
* ML type (regression, classification, etc.)
* Success metrics

---

## 2. Data Gathering

**What it means**
Collect relevant data.

**Sources**

* Databases
* APIs
* Logs
* Sensors

---

## 3. Data Preprocessing

**What it means**
Clean and prepare data.

**Tasks**

* Handle missing values
* Remove duplicates
* Encode categorical data
* Scale features

---

## 4. Exploratory Data Analysis (EDA)

**What it means**
Understand data patterns.

**Tasks**

* Summary statistics
* Visualizations
* Detect outliers

---

## 5. Feature Engineering & Selection

**What it means**
Create and choose useful features.

**Tasks**

* Feature creation
* Feature transformation
* Remove irrelevant features

---

## 6. Model Training & Evaluation

**What it means**
Train and test models.

**Tasks**

* Train algorithms
* Cross-validation
* Performance metrics

---

## 7. Model Deployment

**What it means**
Make the model available in production.

**Methods**

* REST API
* Batch jobs
* Embedded systems

---

## 8. Testing

**What it means**
Validate the deployed model.

**Tasks**

* Functional tests
* Load tests
* Performance tests

---

## 9. Optimization

**What it means**
Improve model over time.

**Tasks**

* Hyperparameter tuning
* Model retraining
* Drift handling

---

## 10. Simple MLDLC Diagram

```
[Frame Problem]
       |
       v
[Gather Data]
       |
       v
[Preprocess Data]
       |
       v
[EDA]
       |
       v
[Feature Engg]
       |
       v
[Train & Evaluate]
       |
       v
[Deploy Model]
       |
       v
[Testing]
       |
       v
[Optimize]
```

---

## 11. Summary

* Start with a clear problem.
* Data quality drives model quality.
* Deployment is as important as training.
* Optimization is continuous.

---

**One-liner:**

> MLDLC turns a business problem into a deployed ML solution.


# Key Job Roles in Data & ML

## 1. Data Engineer

**What they do**
Build and maintain data pipelines and infrastructure.

**How they do it**

* Design ETL pipelines
* Manage databases and data lakes
* Ensure data quality and availability

**Skills required**

* SQL, Python, Spark
* Cloud (AWS, GCP, Azure)
* Data modeling, pipelines

**What companies need**

* Reliable data flow
* Scalable infrastructure
* Clean, usable data

---

## 2. Data Analyst

**What they do**
Analyze data and create reports and dashboards.

**How they do it**

* Query data
* Build visualizations
* Interpret trends

**Skills required**

* SQL, Excel
* Python / R (basic)
* Tableau, Power BI

**What companies need**

* Business insights
* KPI tracking
* Decision support

---

## 3. Data Scientist

**What they do**
Build predictive models and extract insights.

**How they do it**

* EDA and feature engineering
* Train ML models
* Evaluate and improve models

**Skills required**

* Python, ML algorithms
* Statistics, math
* SQL

**What companies need**

* Predictive insights
* Business optimization
* Advanced analytics

---

## 4. ML Engineer

**What they do**
Deploy and maintain ML models in production.

**How they do it**

* Build ML pipelines
* Serve models via APIs
* Monitor model performance

**Skills required**

* Python, ML frameworks
* Docker, Kubernetes
* CI/CD, MLOps

**What companies need**

* Scalable ML systems
* Reliable production models
* Automation

---

## 5. Key Differences

| Role           | Focus Area         | Main Goal           |
| -------------- | ------------------ | ------------------- |
| Data Engineer  | Data pipelines     | Reliable data flow  |
| Data Analyst   | Reports & insights | Business decisions  |
| Data Scientist | Models & insights  | Predictions         |
| ML Engineer    | Production ML      | Scalable ML systems |

---

## 6. Skills Comparison

| Skill  | DE | DA | DS | MLE |
| ------ | -- | -- | -- | --- |
| SQL    | ✅  | ✅  | ✅  | ⚠️  |
| Python | ✅  | ⚠️ | ✅  | ✅   |
| ML     | ❌  | ❌  | ✅  | ✅   |
| Cloud  | ✅  | ⚠️ | ⚠️ | ✅   |
| DevOps | ⚠️ | ❌  | ❌  | ✅   |

---

## 7. Pay (Very Rough, Region-Dependent)

> Pay varies a lot by country, company, and skills.

* **Data Analyst** – Lowest among four
* **Data Engineer** – High
* **Data Scientist** – High
* **ML Engineer** – Highest

**Experience trend**

* Entry-level → Low to medium
* Mid-level → Medium to high
* Senior → High to very high

---

## 8. Summary

* Data Engineers build data systems.
* Data Analysts explain data.
* Data Scientists predict outcomes.
* ML Engineers deploy ML models.

---

**One-liner:**

> Analysts explain, Scientists predict, Engineers build and deploy.

# Classification Metrics in Machine Learning

## 1. Confusion Matrix

A **confusion matrix** is a table that compares **actual** vs **predicted** class labels.

|                     | Predicted Positive  | Predicted Negative  |
| ------------------- | ------------------- | ------------------- |
| **Actual Positive** | TP (True Positive)  | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative)  |

**Definitions**

* **TP** – Correctly predicted positive
* **FP** – Incorrectly predicted positive
* **FN** – Incorrectly predicted negative
* **TN** – Correctly predicted negative

---

## 2. Accuracy

**What it is**

Accuracy measures the overall correctness of the model.

**Formula**

Accuracy = (TP + TN) / (TP + FP + FN + TN)

**When to use**

* Balanced datasets
* When both classes are equally important

**When *not* to use**

* Imbalanced datasets (e.g., fraud detection, rare disease detection)
* When false negatives or false positives are costly

---

## 3. Recall (Sensitivity / True Positive Rate)

**What it is**

Recall measures how many actual positives were correctly identified.

**Formula**

Recall = TP / (TP + FN)

**When to use**

* When missing a positive case is expensive
* Examples:

  * Disease detection (don’t miss sick patients)
  * Fraud detection
  * Spam detection

**When *not* to use**

* When false positives are very costly

---

## 4. Precision (Positive Predictive Value)

**What it is**

Precision measures how many predicted positives are actually positive.

**Formula**

Precision = TP / (TP + FP)

**When to use**

* When false positives are expensive
* Examples:

  * Email spam filters (don’t block important emails)
  * Legal or compliance systems

**When *not* to use**

* When missing a positive is more critical than raising a false alarm

---

## 5. F-Beta Score

When both **precision** and **recall** are important, we use the **F-beta score**.

**What it is**

The F-beta score is a weighted harmonic mean of precision and recall.

**Formula**

Fβ = (1 + β²) × (Precision × Recall) / (Precision + Recall)

**How beta (β) works**

* **β > 1** → Recall is more important than precision
* **β = 1** → Precision and recall are equally important (F1 score)
* **β < 1** → Precision is more important than recall

**When to use**

* Imbalanced datasets
* When you need a single score combining precision and recall

---

## 6. Summary

| Metric    | What it Measures                     | Best Used When                   |
| --------- | ------------------------------------ | -------------------------------- |
| Accuracy  | Overall correctness                  | Balanced datasets                |
| Recall    | Ability to find all positives        | Missing positives is costly      |
| Precision | Correctness of positive predictions  | False positives are costly       |
| F-beta    | Trade-off between precision & recall | Both precision and recall matter |

---

**One-liner:**

> Accuracy for balance, Recall to catch all positives, Precision to avoid false alarms, and F-beta to balance both.

---

## 7. ROC Curve and AUC

### ROC Curve (Receiver Operating Characteristic)

**What it is**

The ROC curve shows the trade-off between:

* **True Positive Rate (TPR)** = Recall = TP / (TP + FN)
* **False Positive Rate (FPR)** = FP / (FP + TN)

at different classification thresholds.

The curve plots:

* X-axis → FPR
* Y-axis → TPR

Each point on the curve corresponds to a different probability threshold.

---

### AUC (Area Under the Curve)

**What it is**

AUC is the area under the ROC curve.

It measures how well the model can distinguish between positive and negative classes.

**Range**

* 1.0 → Perfect classifier
* 0.9–1.0 → Excellent
* 0.8–0.9 → Good
* 0.7–0.8 → Fair
* 0.5 → Random guessing

---

### How ROC Curve is Built

1. Sort predictions by probability (highest to lowest)
2. Start with a very high threshold → almost everything predicted negative
3. Gradually lower the threshold
4. At each step, compute TPR and FPR
5. Plot the points to form the curve

---

### When to Use ROC–AUC

* Binary classification problems
* When class imbalance exists (moderate imbalance)
* When ranking predictions is more important than exact probabilities
* When you want a threshold-independent metric

**Examples**

* Credit scoring
* Medical diagnosis
* Fraud detection

---

### When *Not* to Use ROC–AUC

* Extremely imbalanced datasets (precision–recall curve is better)
* When false positives and false negatives have very different costs
* When you care about a specific operating threshold

---

### ROC vs Precision–Recall Curve

| Metric Aspect    | ROC Curve                          | Precision–Recall Curve    |
| ---------------- | ---------------------------------- | ------------------------- |
| Focus            | TPR vs FPR                         | Precision vs Recall       |
| Best for         | Balanced or mildly imbalanced data | Highly imbalanced data    |
| Interpretability | Good overall discrimination        | Better for rare positives |

---

### Summary

* ROC curve shows TPR vs FPR across thresholds
* AUC measures overall separability of classes
* Higher AUC = better model
* ROC–AUC is threshold-independent

**One-liner:**

> ROC shows trade-offs across thresholds, AUC tells how well the model separates positives from negatives.

---

## 8. Regression Metrics: MAE, MSE, RMSE, R², Adjusted R²

We use these metrics to evaluate **regression models** (e.g., predicting salary from CGPA).

---

## 8.1 Mean Absolute Error (MAE)

**What it is**
MAE measures the average absolute difference between actual and predicted values.

**Formula**
MAE = (1/n) × Σ | yᵢ − ŷᵢ |

**Advantages**

* Easy to understand
* Less sensitive to outliers
* Same unit as target (e.g., salary in ₹ or $)

**Disadvantages**

* Does not penalize large errors strongly
* Not differentiable at zero (for some optimization methods)

**Real-life meaning**

> On average, your prediction is off by MAE units.

**CGPA → Package Example**

If MAE = 0.8 LPA, then:

> On average, the predicted package differs from the actual package by 0.8 LPA.

**Good vs Bad Values**

* Lower MAE = better
* What is "good" depends on business context

---

## 8.2 Mean Squared Error (MSE)

**What it is**
MSE measures the average squared difference between actual and predicted values.

**Formula**
MSE = (1/n) × Σ ( yᵢ − ŷᵢ )²

**Advantages**

* Penalizes large errors heavily
* Smooth and differentiable
* Common loss function in ML

**Disadvantages**

* Sensitive to outliers
* Units are squared (hard to interpret)

**Real-life meaning**

> Large mistakes hurt more than small ones.

**CGPA → Package Example**

If MSE = 4, then:

> Large prediction errors are being penalized heavily.

**Good vs Bad Values**

* Lower MSE = better
* Not directly interpretable due to squared units

---

## 8.3 Root Mean Squared Error (RMSE)

**What it is**
RMSE is the square root of MSE.

**Formula**
RMSE = √MSE

**Advantages**

* Same unit as target
* Penalizes large errors
* More interpretable than MSE

**Disadvantages**

* Still sensitive to outliers
* Can exaggerate impact of rare big errors

**Real-life meaning**

> Typical prediction error size.

**CGPA → Package Example**

If RMSE = 1.5 LPA, then:

> Typical error in salary prediction is about 1.5 LPA.

**Good vs Bad Values**

* Lower RMSE = better
* Compare RMSE to target range (e.g., packages from 3–25 LPA)

---

## 8.4 R² Score (Coefficient of Determination)

**What it is**
R² measures how much variance in the target variable is explained by the model.

**Formula**
R² = 1 − ( Σ ( yᵢ − ŷᵢ )² / Σ ( yᵢ − ȳ )² )

**Range**

* 1.0 → Perfect model
* 0.0 → No better than predicting the mean
* < 0 → Worse than predicting the mean

**Advantages**

* Easy to interpret
* Scale-independent
* Common benchmark metric

**Disadvantages**

* Increases when more features are added (even useless ones)
* Does not indicate overfitting
* Can be misleading for nonlinear data

**Real-life meaning**

> Percentage of variability explained by the model.

**CGPA → Package Example**

If R² = 0.72, then:

> 72% of salary variation is explained by CGPA.

**Good vs Bad Values**

* > 0.7 → Good
* 0.5–0.7 → Moderate
* < 0.5 → Weak

---

## 8.5 Adjusted R² Score

**What it is**
Adjusted R² penalizes adding irrelevant features.

**Formula**
Adjusted R² = 1 − [ (1 − R²) × (n − 1) / (n − p − 1) ]

Where:

* n = number of data points
* p = number of features

**Advantages**

* Penalizes useless variables
* Better for model comparison

**Disadvantages**

* Slightly harder to interpret
* Still does not guarantee good generalization

**Real-life meaning**

> How well the model explains data after considering feature count.

**CGPA → Package Example**

If R² = 0.80 and Adjusted R² = 0.65, then:

> Many features added are not useful.

---

## 9. Common Problems in Regression Metrics

**Outliers**

* MAE → less affected
* MSE / RMSE → highly affected

**Overfitting**

* High R² on training data
* Low R² on test data

**Scale sensitivity**

* MAE, MSE, RMSE depend on target scale
* R² is scale-independent

---

## 10. Metric Comparison Table

| Metric | Unit           | Sensitive to Outliers | Interpretable | Best Use Case        |
| ------ | -------------- | --------------------- | ------------- | -------------------- |
| MAE    | Same as target | Low                   | High          | Stable error measure |
| MSE    | Squared        | High                  | Low           | Model training       |
| RMSE   | Same as target | High                  | High          | Penalize big errors  |
| R²     | None           | Medium                | High          | Explain variance     |
| Adj R² | None           | Medium                | High          | Compare models       |

---

## 11. Summary

* MAE → Average error size
* MSE → Penalizes large errors
* RMSE → Interpretable MSE
* R² → Explained variance
* Adjusted R² → R² with penalty

**One-liner:**

> MAE shows average mistake, RMSE shows typical big mistakes, R² shows how much the model explains, and Adjusted R² tells if extra features really help.
