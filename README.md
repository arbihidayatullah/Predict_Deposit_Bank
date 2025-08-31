Predict Deposit Bank

This project aims to predict whether a bank client will subscribe to a term deposit based on telemarketing data. Using machine learning, the model helps financial institutions optimize marketing strategies, reduce costs, and increase deposit subscriptions.

📊 Dataset

Source: Portuguese Bank Marketing Dataset (Kaggle)

Rows: 45,211

Features: 16 input features + 1 target label

Target: y → whether the client subscribes to a deposit (yes/no)

Key features include: age, job, marital status, education, balance, loan, housing, contact type, campaign, previous outcome, duration, and more
.

🎯 Business Purpose

Identify potential clients → determine which customers are more likely to subscribe to a term deposit.

Increase marketing efficiency → reduce costs by focusing on high-probability clients.

Boost bank revenue → by improving term deposit conversion rates
.

⚙️ Preprocessing

Missing value check

Feature encoding (job, marital, education, contact, month, etc.)

Feature selection using ANOVA & Mutual Information

Handling imbalanced data (resampling)

🤖 Models

Several models were evaluated:

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.7748	0.76–0.79	0.76–0.79	0.78
Decision Tree	0.8841	0.87–0.90	0.87–0.90	0.88
Random Forest	0.9235	0.89–0.95	0.89–0.95	0.92

After hyperparameter tuning, Random Forest remained the best-performing model with ~92% accuracy.

🔑 Feature Importance

Important features influencing predictions include:

Duration of the last contact

Pdays (days since last contact)

Poutcome (outcome of previous campaign)

Balance

Contact type (cellular/telephone)

🚀 Deployment

The model is deployed to predict new client data. Example input features:

{
  "duration": 90,
  "pdays": 3,
  "outcome": "success",
  "month": "June",
  "contact": "cellular"
}


Output → Probability that client will subscribe to a term deposit.

🛠️ Tech Stack

Python (pandas, numpy, scikit-learn)

Machine Learning (Logistic Regression, Decision Tree, Random Forest)

Model Evaluation (Accuracy, Precision, Recall, F1-score)

📌 How to Run
# Clone repo
git clone https://github.com/arbihidayatullah/Predict_Deposit_Bank.git
cd Predict_Deposit_Bank

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook or Python scripts
jupyter notebook

📖 Contribution

Contributions are welcome! Feel free to fork this repository, create a branch, and submit a pull request.

👨‍💻 Author

Arbi Hidayatullah
Computer Vision & Machine Learning Enthusiast
