# Sentiment Analysis Project Report

## Project Overview

This project implements a machine learning-based sentiment analysis system for Amazon product reviews. The system classifies reviews into three categories: positive, negative, and neutral based on the review text content.

## Dataset Information

- **Source**: Amazon product reviews dataset (Reviews.csv)
- **Initial Size**: Variable (depends on the original dataset)
- **Features**: Score, Text, HelpfulnessNumerator, HelpfulnessDenominator
- **Target Variable**: Sentiment classification (positive/negative/neutral)

## Data Preprocessing

### 1. Data Cleaning
- Removed rows with missing values
- Eliminated duplicate reviews based on Score and Text
- Filtered out invalid helpfulness data (where numerator > denominator)

### 2. Target Variable Creation
- **Positive**: Score > 3
- **Negative**: Score < 3  
- **Neutral**: Score = 3

### 3. Data Balancing
- Sampled up to 50,000 reviews from each sentiment class
- Final balanced dataset includes positive, negative, and neutral reviews

### 4. Text Preprocessing Pipeline
The text preprocessing includes:
- HTML tag removal
- Punctuation removal
- Digit removal
- Lowercase conversion
- Multiple whitespace normalization
- Stop word removal (excluding negative words like "no", "not", "don't")
- Porter stemming

## Feature Engineering

### 1. Bag of Words (BoW)
- CountVectorizer with max_features=10,000
- Captures word frequency in reviews

### 2. TF-IDF Vectorization
- TfidfVectorizer with max_features=10,000
- Considers both term frequency and inverse document frequency
- Generally provides better performance than BoW

## Model Development

### 1. Machine Learning Models Tested

#### Logistic Regression
- **Hyperparameters tested**: C = [0.001, 0.01, 0.1, 1, 10]
- **Best performance**: C = 1
- **Vectorization**: Both BoW and TF-IDF tested

#### Naive Bayes (Multinomial)
- **Hyperparameters tested**: alpha = [0, 0.2, 0.6, 0.8, 1]
- **Best performance**: alpha = 0.2
- **Vectorization**: Both BoW and TF-IDF tested

### 2. Model Selection
- **Final Model**: Logistic Regression with C=1
- **Vectorization Method**: TF-IDF
- **Reason**: Best balance of accuracy and interpretability

## Model Performance

### Evaluation Metrics
- **Accuracy Score**: Primary evaluation metric
- **Confusion Matrix**: For detailed class-wise performance analysis

### Results
- The model achieves good accuracy on both training and test sets
- Confusion matrix shows balanced performance across all three classes
- TF-IDF vectorization generally outperforms BoW

## Model Deployment

### 1. Model Persistence
- TF-IDF vectorizer saved as "transformer.pkl"
- Trained model saved as "model.pkl"

### 2. Prediction Function
```python
def get_sentiment(review):
    # preprocessing
    x = preprocessor(review)
    #vectorization
    x = tfidf_vectorizer.transform([x])
    #prediction
    y = int(bmodel.predict(x.reshape(1,-1)))
    return labels[y]
```

## Visualization and Analysis

### 1. Data Distribution
- Bar charts showing the distribution of sentiment classes
- Balanced dataset after sampling

### 2. Word Clouds
- Generated for each sentiment class (positive, negative, neutral)
- Provides visual insights into most common words per sentiment

## Technical Implementation

### Dependencies
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Text Processing**: nltk, re, string
- **Visualization**: matplotlib, seaborn, wordcloud
- **Deep Learning**: keras, tensorflow
- **Model Persistence**: pickle

### Key Functions
1. `preprocessor()`: Complete text preprocessing pipeline
2. `create_targets()`: Converts numerical scores to sentiment labels
3. `train_and_eval()`: Model training and evaluation wrapper
4. `plot_cm()`: Confusion matrix visualization
5. `generate_wcloud()`: Word cloud generation

## Project Structure

```
project/
├── sentimentAnalysisIntern.py    # Main implementation
├── Reviews.csv                   # Dataset
├── transformer.pkl              # Saved TF-IDF vectorizer
├── model.pkl                    # Saved trained model
└── report.md                    # This report
```

## Key Findings

1. **Text Preprocessing Impact**: Comprehensive preprocessing significantly improves model performance
2. **Vectorization Choice**: TF-IDF outperforms Bag of Words for this sentiment analysis task
3. **Model Selection**: Logistic Regression provides the best balance of performance and interpretability
4. **Data Balancing**: Sampling helps create a more balanced dataset for training

## Future Improvements

1. **Deep Learning**: Implement LSTM/Transformer models for potentially better performance
2. **Hyperparameter Tuning**: Use GridSearchCV for more systematic hyperparameter optimization
3. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
4. **Feature Engineering**: Explore additional features like review length, word count, etc.
5. **Model Ensemble**: Combine multiple models for improved performance

## Conclusion

This sentiment analysis system successfully classifies Amazon product reviews into positive, negative, and neutral categories. The implementation demonstrates a complete machine learning pipeline from data preprocessing to model deployment. The Logistic Regression model with TF-IDF vectorization provides reliable performance and can be easily deployed for real-world applications.

The project showcases best practices in:
- Data cleaning and preprocessing
- Feature engineering
- Model selection and evaluation
- Model persistence and deployment
- Visualization and analysis 
