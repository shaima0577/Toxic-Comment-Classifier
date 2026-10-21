# Toxic Comment Classifier

A Machine Learning project that classifies text comments as **Toxic** or **Non-Toxic** using Natural Language Processing (NLP).

## Project Overview

This project explores text classification using machine learning techniques. It analyzes written comments and predicts whether they may contain toxic language.

The project includes:
- A Jupyter Notebook for data analysis, model training, and evaluation.
- A Python GUI application for classifying text comments.
- A trained Logistic Regression model used by the application.

## Models Compared

The training notebook compares the following machine learning algorithms:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest Classifier

The final Logistic Regression model is saved and used in the GUI application.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Jupyter Notebook
- Tkinter
- Matplotlib
- Seaborn
- Joblib

## Project Files

- `project.ipynb` — Notebook for data analysis, model training, and evaluation.
- `toxic_app.py` — GUI application for classifying comments.
- `toxicity_model.pkl` — Trained machine learning model.
- `requirements.txt` — List of Python dependencies.

## Dataset and Learning Resource

The project was developed using data associated with this Kaggle notebook:

[Exercise: Identifying Bias in AI](https://www.kaggle.com/code/shaima463/exercise-identifying-bias-in-ai)

The notebook is a learning resource related to identifying bias in AI. This project focuses on toxic comment classification; it does not claim to measure or eliminate bias.

## How to Run

1. Install Python.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `toxicity_model.pkl` is in the same folder as `toxic_app.py`.
4. Run the application:

```bash
python toxic_app.py
```

## Limitations

- Model predictions may be incorrect.
- Toxicity classification does not fully capture context, intent, or differences in language.
- The application should be tested in the target environment before use.

## Author

Shaimaa Abdullah Mohammed Al-Ganim
