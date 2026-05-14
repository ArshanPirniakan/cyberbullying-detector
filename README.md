# Cyberbullying Detector

A machine learning tool that detects toxic and harmful content in text using TF-IDF and Logistic Regression. Trained on the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), it outputs a binary toxicity verdict alongside confidence scores for six specific harm categories.

---

## Features

- **Binary classification** — flags text as toxic or non-toxic with a confidence score
- **Multi-label detection** — scores across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **Text preprocessing** — lowercasing, URL removal, lemmatization, stopword filtering
- **TF-IDF + Logistic Regression pipeline** — fast, interpretable, and lightweight
- **Interactive CLI mode** — test the model in real time from the terminal

---

## Installation

```bash
git clone https://github.com/your-username/cyberbullying-detector.git
cd cyberbullying-detector
pip install -r requirements.txt
```

**Dependencies:** `scikit-learn`, `nltk`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## Usage

### Train

```bash
python cyberbullying_detector.py train path/to/train.csv
```

Expects a CSV with a `comment_text` column and label columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

Outputs:
- `model.pkl` — saved model
- `confusion_matrix.png` — binary classifier confusion matrix
- `metrics.png` — accuracy, precision, recall, F1 bar chart

### Predict — single input

```bash
python cyberbullying_detector.py predict "Your text here"
```

### Predict — interactive mode

```bash
python cyberbullying_detector.py predict
```

---

## Example Output

```
Input : you are an idiot and I hate you
Toxic  : YES
Score  : 91.4%

Category Scores:
  toxic           88.2%
  severe_toxic    12.1%
  obscene         34.7%
  threat           5.3%
  insult          81.6%
  identity_hate    3.9%

Detected: toxic, insult
```

---

## Model Details

| Component | Detail |
|---|---|
| Vectorizer | TF-IDF, unigrams + bigrams, 50k features |
| Classifier | Logistic Regression (C=5, balanced class weight) |
| Train/test split | 80/20, stratified |
| Per-label models | One binary classifier per category |

---

## Dataset

This project uses the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset from Kaggle. Download `train.csv` and pass its path to the train command.

---

## License

MIT
