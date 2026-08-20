# 🏦 Predict Bank Term Deposit

<p align="center">
  <strong>Machine Learning for Predicting Customer Term Deposit Subscription</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Data%20Science-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=flat-square&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Random%20Forest-Best%20Model-success?style=flat-square" />
</p>

<br>

> **Can we identify which bank customers are most likely to subscribe to a term deposit?**

This project applies **Machine Learning** to predict whether a bank client will subscribe to a term deposit based on Portuguese bank telemarketing data.

The objective is to help financial institutions focus marketing efforts on high-potential customers, improve campaign efficiency, and reduce unnecessary outreach.

---

<!-- ========================================= -->

<!--               PROJECT OVERVIEW             -->

<!-- ========================================= -->

## 🎯 Project Overview

Traditional marketing campaigns often contact a large number of customers without knowing who is genuinely interested.

This project transforms historical customer and campaign data into a predictive model capable of estimating the likelihood that a client will subscribe to a **term deposit**.

### The Prediction Task

```text
                    BANK CUSTOMER DATA
                           │
                           ▼
                ┌─────────────────────┐
                │   DATA PROCESSING   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  FEATURE SELECTION  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ MACHINE LEARNING    │
                │                     │
                │ Logistic Regression │
                │ Decision Tree       │
                │ Random Forest       │
                └──────────┬──────────┘
                           │
                           ▼
                   PREDICTION
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         SUBSCRIBE                  NOT SUBSCRIBE
```

---

<!-- ========================================= -->

<!--                   DATASET                  -->

<!-- ========================================= -->

## 📊 Dataset

The project uses the **Portuguese Bank Marketing Dataset**.

| Information       | Details              |
| ----------------- | -------------------- |
| 📁 Dataset        | Bank Marketing       |
| 📊 Records        | **45,211**           |
| 🔢 Input Features | **16**               |
| 🎯 Target         | `y`                  |
| 📌 Prediction     | Deposit Subscription |

The target variable:

```text
y = yes → Client subscribes to a term deposit
y = no  → Client does not subscribe
```

### 🔍 Example Features

The dataset contains customer, financial, and campaign-related attributes:

```text
👤 Customer Profile
├── Age
├── Job
├── Marital Status
└── Education

💰 Financial Information
├── Balance
├── Housing Loan
└── Personal Loan

📞 Marketing Campaign
├── Contact Type
├── Duration
├── Campaign
├── Pdays
└── Previous Outcome
```

---

<!-- ========================================= -->

<!--                BUSINESS VALUE              -->

<!-- ========================================= -->

## 💼 Business Value

The model can support a more targeted marketing strategy.

### 🎯 Identify Potential Customers

Focus on customers with a higher probability of subscribing.

### 💰 Reduce Marketing Costs

Avoid spending resources on low-probability prospects.

### 📈 Improve Conversion Potential

Use historical data to support more informed campaign decisions.

---

<!-- ========================================= -->

<!--                  WORKFLOW                  -->

<!-- ========================================= -->

## ⚙️ Machine Learning Workflow

```text
Raw Bank Marketing Data
          │
          ▼
┌─────────────────────────┐
│   Data Understanding    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Data Preprocessing    │
│                         │
│ • Missing Value Check   │
│ • Feature Encoding      │
│ • Data Preparation      │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Feature Selection     │
│                         │
│ • ANOVA                 │
│ • Mutual Information    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Imbalanced Data Handling│
│       Resampling        │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Model Development     │
│                         │
│ Logistic Regression     │
│ Decision Tree           │
│ Random Forest           │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Model Evaluation      │
│                         │
│ Accuracy                │
│ Precision               │
│ Recall                  │
│ F1-Score                │
└────────────┬────────────┘
             ▼
       Final Prediction
```

---

<!-- ========================================= -->

<!--                 PREPROCESSING              -->

<!-- ========================================= -->

## 🧹 Data Preprocessing

Several preprocessing steps were performed before training the models:

* 🔎 Missing value inspection
* 🔄 Categorical feature encoding
* 🎯 Feature selection using **ANOVA**
* 📊 Feature selection using **Mutual Information**
* ⚖️ Imbalanced data handling using resampling

The goal was to prepare a more meaningful feature set for the classification models.

---

<!-- ========================================= -->

<!--                MODEL COMPARISON            -->

<!-- ========================================= -->

## 🤖 Model Comparison

Three Machine Learning algorithms were evaluated.

| Model                |   Accuracy | Performance        |
| -------------------- | ---------: | ------------------ |
| Logistic Regression  |     77.48% | Baseline Model     |
| Decision Tree        |     88.41% | Strong Performance |
| 🌲 **Random Forest** | **92.35%** | 🏆 **Best Model**  |

> Random Forest achieved the highest accuracy among the evaluated models.

---

<!-- ========================================= -->

<!--                  BEST MODEL                -->

<!-- ========================================= -->

## 🏆 Best Performing Model

<p align="center">

### 🌲 RANDOM FOREST

**92.35% Accuracy**

</p>

Random Forest demonstrated the strongest performance after model evaluation and hyperparameter tuning.

```text
Customer Information
        │
        ▼
   Random Forest
        │
        ▼
Probability of Subscription
        │
   ┌────┴────┐
   ▼         ▼
 YES         NO
Subscribe   Not Subscribe
```

---

<!-- ========================================= -->

<!--              FEATURE IMPORTANCE            -->

<!-- ========================================= -->

## 🔑 Important Predictive Features

Several features showed a strong influence on the prediction:

| Feature       | Description                      |
| ------------- | -------------------------------- |
| ⏱️ `duration` | Duration of the last contact     |
| 📅 `pdays`    | Days since the previous contact  |
| 📈 `poutcome` | Outcome of the previous campaign |
| 💰 `balance`  | Customer account balance         |
| 📞 `contact`  | Contact communication type       |

These variables provide useful signals for estimating the probability of subscription.

---

<!-- ========================================= -->

<!--                 PREDICTION                 -->

<!-- ========================================= -->

## 🔮 Example Prediction

Example customer input:

```json
{
  "duration": 90,
  "pdays": 3,
  "outcome": "success",
  "month": "June",
  "contact": "cellular"
}
```

The model processes customer information and produces an estimated prediction:

```text
Input Customer Data
        │
        ▼
Machine Learning Model
        │
        ▼
Subscription Probability
        │
        ▼
┌───────────────────────┐
│  YES / NO Prediction  │
└───────────────────────┘
```

---

<!-- ========================================= -->

<!--                PROJECT STRUCTURE           -->

<!-- ========================================= -->

## 📂 Repository Structure

```text
Predict_Deposit_Bank/
│
├── 📓 DS_kel3.ipynb
├── 📓 Deposit_Bank.ipynb
├── 📓 Deposit_Bank_Selection_Feature.ipynb
│
├── 🐍 Deposit_Bank_ALL_in.py
├── 🐍 deposit.py
├── 🐍 training_data.py
│
├── 🚀 run_streamlit.py
├── 🧪 test_streamlit.py
├── 🧪 test_streamlit_2.py
│
├── 📊 bank-full.csv
├── 📊 bank-full-1.csv
├── 📊 bank-full_original.csv
│
├── 📄 Explanatioin_project_Predict Bank Deposit.pdf
│
└── 📖 README.md
```

---

<!-- ========================================= -->

<!--                 TECH STACK                 -->

<!-- ========================================= -->

## 🛠️ Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=python" height="50"/>

</p>

| Category            | Tools                                             |
| ------------------- | ------------------------------------------------- |
| 🐍 Programming      | Python                                            |
| 📊 Data Processing  | Pandas, NumPy                                     |
| 🤖 Machine Learning | Scikit-learn                                      |
| 📈 Models           | Logistic Regression, Decision Tree, Random Forest |
| 📓 Experimentation  | Jupyter Notebook                                  |
| 🚀 Application      | Streamlit                                         |

---

<!-- ========================================= -->

<!--                GETTING STARTED             -->

<!-- ========================================= -->

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/arbihidayatullah/Predict_Deposit_Bank.git
```

```bash
cd Predict_Deposit_Bank
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

### 3. Run the Project

You can explore the Machine Learning workflow through the notebooks:

```text
Deposit_Bank.ipynb
```

or run the Python implementation:

```bash
python Deposit_Bank_ALL_in.py
```

For the Streamlit application:

```bash
streamlit run run_streamlit.py
```

> Adjust the command if the Streamlit entry point or project dependencies differ from your local environment.

---

<!-- ========================================= -->

<!--                   RESULTS                  -->

<!-- ========================================= -->

## 📈 Key Takeaways

* 📊 The project analyzes **45,211 bank marketing records**.
* 🤖 Three Machine Learning models were compared.
* 🌲 **Random Forest achieved the strongest performance** with approximately **92.35% accuracy**.
* 🎯 Feature selection and preprocessing were applied before model training.
* 💼 The prediction can support more targeted bank marketing strategies.

---

<!-- ========================================= -->

<!--                   AUTHOR                   -->

<!-- ========================================= -->

## 👨‍💻 Author

**Arbi Hidayatullah**

AI • Machine Learning • Computer Vision • Data

<p>
  <a href="https://github.com/arbihidayatullah">
    GitHub
  </a>
  &nbsp; • &nbsp;
  <a href="https://www.linkedin.com/in/arbi-hidayatullah/">
    LinkedIn
  </a>
</p>

---

<p align="center">
  <strong>Transforming data into meaningful predictions.</strong>
</p>

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>
