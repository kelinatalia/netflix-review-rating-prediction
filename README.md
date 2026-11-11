# Netflix Review Rating Prediction

Predicts a Netflix Play Store review's star rating (1-5) from its text alone, comparing classic NLP features against pretrained transformer features, with a focus on handling severe class imbalance.

## The problem

Given only the text of a Play Store review, can a model recover the star rating the user gave? Ratings are heavily skewed toward 5 stars, so the real challenge isn't accuracy, it's correctly identifying the minority ratings (1-4) without the model collapsing into "always predict 5."

## Data

~1,000 English-language reviews of the Netflix Android app, scraped via `google-play-scraper`, later supplemented with additional minority-class reviews (ratings 2-4) to address class imbalance.

## Pipeline

1. **EDA** — rating distribution, review length, writing style, stopword effects, dominant words and n-grams per rating, topic modeling, and cases where review text doesn't match its star rating.
2. **Preprocessing** — cleaning, emoji-to-text conversion, contraction/slang expansion, lemmatization vs. stemming, stopword removal.
3. **Text representation** — TF-IDF vs. Word2Vec (trained from scratch on the corpus).
4. **Modeling** — SVM and Random Forest, grid-searched over both representations (4 combinations), evaluated with macro F1 (chosen over accuracy, since always predicting 5 scores ~68% accuracy on its own).
5. **Imbalanced data treatment** — class weighting, random oversampling, and SMOTE, compared against the untreated baseline.
6. **Scaling up** — scraping and incorporating additional real minority-class reviews, then swapping in pretrained transformer features (sentence embeddings, RoBERTa sentiment probabilities, BERT star-rating probabilities) in place of TF-IDF/Word2Vec.

## Results

| Approach | Test macro F1 |
|---|---|
| Baseline (always predict 5) | 0.162 |
| Best TF-IDF/Word2Vec model (no treatment) | 0.330 |
| Best TF-IDF/Word2Vec model (with imbalance treatment) | 0.391 |
| Best TF-IDF/Word2Vec + extra minority data | 0.359 |
| **Best overall: SVM on pretrained transformer features + extra data + class weighting** | **0.529** |

Imbalance treatment consistently helped classic representations (class weighting, oversampling, and SMOTE all beat the untreated baseline), but the largest gain came from switching to pretrained transformer features altogether. The persistent difficulty is separating neighboring ratings (2, 3, 4); with the test set collapsed to 3 classes (negative/neutral/positive), the same best model reaches 0.695 macro F1. All scores carry roughly ±0.09 uncertainty from bootstrap resampling, given the small test set.
