# Pakistani Social Media Content Detective

This project extends the supplied social-media dataset with a lightweight machine-learning workflow for sentiment analysis.

## What it does

- Loads the comment dataset from `comments.csv`
- Cleans the text and keeps labeled sentiment rows (`Positive`, `Neutral`, `Negative`)
- Trains a TF-IDF + logistic-regression model to classify sentiment
- Saves the trained model and evaluation metrics
- Lets you run quick sentiment predictions from the command line
- Provides an interactive Streamlit dashboard for exploring post engagement and comment labels

## How to run

1. Install dependencies:
   `py -3 -m pip install -r requirements.txt`
2. Launch the dashboard:
   `py -3 -m streamlit run app.py`
3. Train or refresh the sentiment model:
   `py -3 train_sentiment_model.py`
4. Predict a comment from the command line:
   `py -3 predict_sentiment.py "Beautiful collection, loved the styling!"`

## Output files

- `artifacts/comment_sentiment_model.joblib`: trained classifier
- `artifacts/sentiment_metrics.json`: validation metrics such as accuracy and classification report
- `app.py`: Streamlit dashboard; its prediction panel uses the trained classifier

## Notes

The dashboard's recorded interactions are raw likes + comments counts, not engagement rates. Sentiment labels are provisional annotations and should not be treated as verified ground truth. The model is tuned for the provided dataset and is intended as a practical machine-learning extension of the analytical project, not a production-grade social listening engine.
