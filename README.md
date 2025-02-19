Overview

This project implements a Naive Bayes Classifier to perform sentiment analysis on text data. The model processes tokenized text, vectorizes it, and applies probability-based classification with Laplace smoothing to improve generalization.

Features

Tokenization & Vectorization: Converts text into numerical representations.

Bernoulli Naive Bayes Model: Implements probabilistic classification.

Learning Curve Analysis: Evaluates model performance by varying training data size.

Performance Evaluation: Computes precision, recall, and F1-score.

Data Import & Export: Reads and writes vectors and labels for easy dataset handling.

├── NaiveBayes/

│   ├── LearningCurveGenerator.java  # Generates learning curves

│   ├── NaiveBayesClassifier.java    # Implements Naive Bayes training & prediction

│   ├── Run.java                     # Main entry point for training & evaluation

├── Utils/

│   ├── EvaluationMetrics.java       # Computes precision, recall, and F1-score

│   ├── VectorImporter.java          # Imports vectors and labels from files

How It Works

Train the Model: The classifier learns word probabilities from labeled text data.

Predict Sentiment: Given a new text sample, the model predicts whether it's positive or negative.

Evaluate Performance: Computes precision, recall, and F1-score to measure accuracy.

Generate Learning Curve: Analyzes how model accuracy improves with more training data.

Input Data

Store training and test dataset vectors in .txt files.

Update file paths in Run.java.

Output Files

learning_curve.csv: Learning curve data.

train_vectors.txt, train_labels.txt: Training dataset.

test_vectors.txt, test_labels.txt: Testing dataset.
