"""
Naive Bayes Classifier Implementation
Multinomial and Bernoulli Naive Bayes for Sentiment Analysis
"""

import math
import collections

class NaiveBayes:
    """Base class for Naive Bayes classifiers"""
    
    def __init__(self, alpha=1.0):
        """
        Initialize Naive Bayes classifier
        
        Args:
            alpha: Smoothing parameter (Laplace smoothing)
        """
        self.alpha = alpha
        self.class_probs = {}
        self.feature_probs = {}
        self.vocab = set()
        self.classes = set()
        self.total_docs = 0
        
    def preprocess_text(self, text):
        """
        Preprocess text: lowercase, split into words, basic cleaning
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase and split into words
        tokens = text.lower().split()
        # Remove non-alphanumeric characters and short words
        processed_tokens = []
        for token in tokens:
            # Remove punctuation and keep only alphanumeric
            clean_token = ''.join(char for char in token if char.isalnum())
            if len(clean_token) > 1:  # Keep only words with length > 1
                processed_tokens.append(clean_token)
        return processed_tokens
    
    def calculate_prior_probabilities(self, labels):
        """
        Calculate prior probabilities for each class
        
        Args:
            labels: List of class labels
        """
        class_counts = collections.Counter(labels)
        self.total_docs = len(labels)
        self.classes = set(labels)
        
        for class_label in self.classes:
            self.class_probs[class_label] = class_counts[class_label] / self.total_docs
    
    def predict(self, texts):
        """
        Predict class labels for given texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of predicted class labels
        """
        predictions = []
        for text in texts:
            predictions.append(self._predict_single(text))
        return predictions
    
    def _predict_single(self, text):
        """
        Predict class for a single text (to be implemented by subclasses)
        """
        raise NotImplementedError("Subclasses must implement this method")


class MultinomialNaiveBayes(NaiveBayes):
    """Multinomial Naive Bayes classifier based on word frequency counts"""
    
    def __init__(self, alpha=1.0):
        super().__init__(alpha)
        self.class_word_counts = {}
        self.class_total_words = {}
    
    def fit(self, texts, labels):
        """
        Train the Multinomial Naive Bayes model
        
        Args:
            texts: List of training text strings
            labels: List of corresponding class labels
        """
        # Calculate prior probabilities
        self.calculate_prior_probabilities(labels)
        
        # Initialize counters for each class
        for class_label in self.classes:
            self.class_word_counts[class_label] = collections.Counter()
            self.class_total_words[class_label] = 0
        
        # Count words in each class
        for text, label in zip(texts, labels):
            tokens = self.preprocess_text(text)
            self.class_word_counts[label].update(tokens)
            self.class_total_words[label] += len(tokens)
            self.vocab.update(tokens)
        
        # Calculate feature probabilities with Laplace smoothing
        vocab_size = len(self.vocab)
        self.feature_probs = {}
        
        for class_label in self.classes:
            self.feature_probs[class_label] = {}
            total_words_class = self.class_total_words[class_label]
            
            for word in self.vocab:
                word_count = self.class_word_counts[class_label].get(word, 0)
                # Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
                prob = (word_count + self.alpha) / (total_words_class + self.alpha * vocab_size)
                self.feature_probs[class_label][word] = prob
    
    def _predict_single(self, text):
        """
        Predict class for a single text using Multinomial Naive Bayes
        
        Args:
            text: Input text string
            
        Returns:
            Predicted class label
        """
        tokens = self.preprocess_text(text)
        
        best_class = None
        best_score = float('-inf')
        
        for class_label in self.classes:
            # Start with log of prior probability
            score = math.log(self.class_probs[class_label])
            
            # Add log probabilities for each word
            for token in tokens:
                if token in self.feature_probs[class_label]:
                    score += math.log(self.feature_probs[class_label][token])
                else:
                    # Handle unseen words with minimum probability
                    vocab_size = len(self.vocab)
                    total_words_class = self.class_total_words[class_label]
                    min_prob = self.alpha / (total_words_class + self.alpha * vocab_size)
                    score += math.log(min_prob)
            
            if score > best_score:
                best_score = score
                best_class = class_label
        
        return best_class


class BernoulliNaiveBayes(NaiveBayes):
    """Bernoulli Naive Bayes classifier based on binary word presence"""
    
    def __init__(self, alpha=1.0):
        super().__init__(alpha)
        self.class_doc_counts = {}
        self.class_word_presence = {}
    
    def fit(self, texts, labels):
        """
        Train the Bernoulli Naive Bayes model
        
        Args:
            texts: List of training text strings
            labels: List of corresponding class labels
        """
        # Calculate prior probabilities
        self.calculate_prior_probabilities(labels)
        
        # Initialize counters for each class
        for class_label in self.classes:
            self.class_doc_counts[class_label] = 0
            self.class_word_presence[class_label] = collections.Counter()
        
        # Build vocabulary and count document presence
        for text, label in zip(texts, labels):
            tokens = set(self.preprocess_text(text))  # Use set for binary presence
            self.class_doc_counts[label] += 1
            
            # Update word presence counts
            for token in tokens:
                self.class_word_presence[label][token] += 1
                self.vocab.add(token)
        
        # Calculate feature probabilities with Laplace smoothing
        self.feature_probs = {}
        
        for class_label in self.classes:
            self.feature_probs[class_label] = {}
            doc_count_class = self.class_doc_counts[class_label]
            
            for word in self.vocab:
                presence_count = self.class_word_presence[class_label].get(word, 0)
                # Laplace smoothing: (presence_count + alpha) / (doc_count_class + 2 * alpha)
                prob = (presence_count + self.alpha) / (doc_count_class + 2 * self.alpha)
                self.feature_probs[class_label][word] = prob
    
    def _predict_single(self, text):
        """
        Predict class for a single text using Bernoulli Naive Bayes
        
        Args:
            text: Input text string
            
        Returns:
            Predicted class label
        """
        tokens = set(self.preprocess_text(text))
        
        best_class = None
        best_score = float('-inf')
        
        for class_label in self.classes:
            # Start with log of prior probability
            score = math.log(self.class_probs[class_label])
            
            # Handle all words in vocabulary
            for word in self.vocab:
                if word in tokens:
                    # Word is present in document
                    prob_present = self.feature_probs[class_label][word]
                    score += math.log(prob_present)
                else:
                    # Word is absent from document
                    prob_absent = 1 - self.feature_probs[class_label][word]
                    score += math.log(prob_absent)
            
            if score > best_score:
                best_score = score
                best_class = class_label
        
        return best_class
