# Statistical Hangman Solver

A sophisticated implementation of a Hangman game solver using statistical analysis and machine learning techniques.

## Overview

This project implements a statistical solver for the Hangman game that uses various features and weighted probabilities to make optimal letter guesses.
It is trained and evaluated on separate train and test wordsets.

The solver achieves high win (~58%) rates by combining multiple strategies:

- N-gram frequency analysis (1 to 5-grams)
- Position-based letter frequencies
- Prefix/suffix pattern recognition
- Vowel/consonant ratio analysis
- Word decomposition for complex cases

## Components

### FeatureExtractor

The `FeatureExtractor` class analyzes a training corpus of words to build statistical models of:
- Letter frequencies by position
- N-gram probabilities
- Common affixes (prefixes/suffixes)
- Vowel/consonant patterns
- Word structure patterns

### HangmanSolver

The `HangmanSolver` class uses the extracted features to make intelligent guesses by:
- Combining multiple probability distributions with learned weights
- Adapting strategy based on word length and game state
- Using special "atomizing" rules for high-confidence predictions
- Maintaining consonant/vowel balance

### Optimization

The solver includes an Optuna-based hyperparameter optimization system that:
- Tunes the weights for different features
- Uses grid search around known good values
- Implements early pruning for poor performers
- Supports parallel optimization


## Performance

The solver typically achieves:
- ~58% win rate on the test set given a sufficiently large training set.

## Requirements

- Python 3.7+
- NumPy
- NLTK
- scikit-learn
- Optuna (for optimization)
- tqdm (for progress bars)

## License

MIT License
