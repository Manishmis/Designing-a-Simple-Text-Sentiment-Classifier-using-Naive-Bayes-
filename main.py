"""
Main script for sentiment analysis using Naive Bayes
Performs 5-fold cross-validation on both Multinomial and Bernoulli models
"""

from naive_bayes import MultinomialNaiveBayes, BernoulliNaiveBayes
from utils import load_data, create_folds, evaluate_predictions, print_results

def main():
    """Main function to run the sentiment analysis experiment"""
    
    # Load data
    print("Loading data...")
    try:
        texts, labels = load_data('data.tsv')
        print(f"Loaded {len(texts)} examples")
    except FileNotFoundError:
        print("Error: data.tsv file not found!")
        print("Please ensure the data file is in the correct format and location.")
        return
    
    # Sample data if needed (500-1000 examples as per instructions)
    if len(texts) > 1000:
        print("Sampling 1000 examples...")
        combined = list(zip(texts, labels))
        import random
        random.seed(42)  # For reproducibility
        sampled = random.sample(combined, 1000)
        texts, labels = zip(*sampled)
        texts, labels = list(texts), list(labels)
    
    print(f"Using {len(texts)} examples for training and evaluation")
    
    # Create 5 folds
    print("Creating 5 folds for cross-validation...")
    folds = create_folds(texts, labels, k=5)
    
    # Initialize models
    multinomial_model = MultinomialNaiveBayes(alpha=1.0)
    bernoulli_model = BernoulliNaiveBayes(alpha=1.0)
    
    # Store results
    multinomial_scores = []
    bernoulli_scores = []
    
    # Perform 5-fold cross-validation
    print("\nStarting 5-fold cross-validation...")
    for fold_idx, (test_texts, test_labels) in enumerate(folds, 1):
        print(f"\nProcessing Fold {fold_idx}/5...")
        
        # Prepare training data (all folds except current)
        train_texts = []
        train_labels = []
        
        for i, (fold_texts, fold_labels) in enumerate(folds):
            if i != fold_idx - 1:  # Exclude current test fold
                train_texts.extend(fold_texts)
                train_labels.extend(fold_labels)
        
        # Train and evaluate Multinomial Naive Bayes
        print("  Training Multinomial Naive Bayes...")
        multinomial_model.fit(train_texts, train_labels)
        multinomial_preds = multinomial_model.predict(test_texts)
        multinomial_acc, multinomial_f1 = evaluate_predictions(test_labels, multinomial_preds)
        multinomial_scores.append((multinomial_acc, multinomial_f1))
        
        # Train and evaluate Bernoulli Naive Bayes
        print("  Training Bernoulli Naive Bayes...")
        bernoulli_model.fit(train_texts, train_labels)
        bernoulli_preds = bernoulli_model.predict(test_texts)
        bernoulli_acc, bernoulli_f1 = evaluate_predictions(test_labels, bernoulli_preds)
        bernoulli_scores.append((bernoulli_acc, bernoulli_f1))
        
        print(f"  Fold {fold_idx} completed")
    
    # Print final results
    print_results(multinomial_scores, bernoulli_scores)

if __name__ == "__main__":
    main()
