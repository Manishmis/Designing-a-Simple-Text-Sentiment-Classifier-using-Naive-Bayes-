"""
Utility functions for data processing and evaluation
"""

import random
from sklearn.metrics import accuracy_score, f1_score

def load_data(filename):
    """
    Load data from TSV file
    
    Args:
        filename: Path to data file
        
    Returns:
        texts: List of text strings
        labels: List of class labels
    """
    texts = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label, text = parts
                texts.append(text)
                labels.append(label)
    
    return texts, labels

def create_folds(texts, labels, k=5):
    """
    Create k folds for cross-validation
    
    Args:
        texts: List of text strings
        labels: List of class labels
        k: Number of folds
        
    Returns:
        folds: List of k folds, each containing (texts, labels)
    """
    # Shuffle data while maintaining text-label correspondence
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts_shuffled, labels_shuffled = zip(*combined)
    
    fold_size = len(texts) // k
    folds = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(texts)
        
        fold_texts = texts_shuffled[start_idx:end_idx]
        fold_labels = labels_shuffled[start_idx:end_idx]
        
        folds.append((fold_texts, fold_labels))
    
    return folds

def evaluate_predictions(true_labels, predicted_labels):
    """
    Calculate accuracy and F1-score
    
    Args:
        true_labels: List of true class labels
        predicted_labels: List of predicted class labels
        
    Returns:
        accuracy: Accuracy score
        f1: F1-score (macro average)
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    return accuracy, f1

def print_results(multinomial_scores, bernoulli_scores):
    """
    Print formatted results for both models
    
    Args:
        multinomial_scores: List of (accuracy, f1) for Multinomial NB
        bernoulli_scores: List of (accuracy, f1) for Bernoulli NB
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    # Multinomial Naive Bayes results
    print("\nMULTINOMIAL NAIVE BAYES:")
    print("-" * 30)
    for i, (acc, f1) in enumerate(multinomial_scores, 1):
        print(f"Fold {i}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")
    
    avg_acc = sum(acc for acc, f1 in multinomial_scores) / len(multinomial_scores)
    avg_f1 = sum(f1 for acc, f1 in multinomial_scores) / len(multinomial_scores)
    print(f"\nAverage: Accuracy = {avg_acc:.4f}, F1-Score = {avg_f1:.4f}")
    
    # Bernoulli Naive Bayes results
    print("\nBERNOULLI NAIVE BAYES:")
    print("-" * 30)
    for i, (acc, f1) in enumerate(bernoulli_scores, 1):
        print(f"Fold {i}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")
    
    avg_acc = sum(acc for acc, f1 in bernoulli_scores) / len(bernoulli_scores)
    avg_f1 = sum(f1 for acc, f1 in bernoulli_scores) / len(bernoulli_scores)
    print(f"\nAverage: Accuracy = {avg_acc:.4f}, F1-Score = {avg_f1:.4f}")
    print("="*60)
