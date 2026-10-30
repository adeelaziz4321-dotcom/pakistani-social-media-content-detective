# Pakistani Social Media Content Detective

**Course project:** Big Data Analytics  
**GitHub repository:** [Add repository URL after creating the GitHub repository]

## Project overview

This project explores what types of social-media content appear to attract engagement for Pakistani fashion and retail brands. It combines exploratory analysis of posts and audience comments with a machine-learning sentiment classifier and an interactive Streamlit dashboard.

The project addresses the business question: **What type of social-media content gets Pakistani audiences to engage?** The analysis is exploratory: it identifies patterns worth testing, rather than claiming that any format or creative choice causes higher engagement.

## Data and methods

The analysis uses the supplied `posts.csv` and `comments.csv` datasets, with `evidence.csv` and `recommendations.csv` providing supporting examples and suggestions. The post dataset contains 159 posts; the comment dataset contains 2,118 collected comments.

Post performance is summarized using available likes and comment counts. Where both measures are present, recorded interactions are calculated as likes plus comments. The dashboard compares the median of these counts by post format, shows platform coverage, and lets users filter results by brand, platform, and format. These counts are not reach-adjusted engagement rates.

For sentiment analysis, comments with Positive, Neutral, or Negative manual labels are cleaned and used to train a TF-IDF text representation with a logistic-regression classifier. A stratified train/test split is used to evaluate the model. The saved evaluation reports **89.2% accuracy** and a **0.844 macro F1 score** on the held-out test set. These scores describe performance on this dataset and do not guarantee accurate predictions on other data.

## Key findings

Reels are the most common format in the dataset (134 of 159 posts). Among posts with both likes and comment counts, Reels have a median of 3,371 recorded interactions. The sample contains only 11 carousels, 7 videos, and 4 images, so comparisons for those formats are uncertain. Instagram accounts for 146 posts, leaving too few observations on other platforms for a dependable platform ranking.

Of the collected comments, 664 have a Positive label, 477 Neutral, 154 Negative, and 822 Uncertain; one is excluded. Common supplied topics include appreciation and price inquiries. Sentiment labels are provisional annotations, and the large uncertain group limits conclusions about audience opinion.

## Dashboard and recommendation

The Streamlit dashboard provides interactive filters, charts for sentiment and topics, format and platform comparisons, and a table of top posts by recorded interactions. It also offers an optional sentiment prediction using the trained model.

The findings suggest testing a short, creator-led collection Reel with a clear product call to action against a comparable product-focused version. To evaluate the test fairly, collect reach, impressions, likes, comments, shares, saves, clicks, and purchases, then compare engagement per reach and business outcomes. Raw likes alone cannot establish a winning strategy.

## Limitations

The data is a collected sample, platform coverage is uneven, and some post metrics are missing. Reach and impressions are unavailable for most posts, so raw interactions cannot be interpreted as engagement rates. Comment labels may be uncertain or inconsistent, and the dataset does not provide reliable audience demographics. These constraints make the results directional rather than conclusive.

## Running the project

Install the requirements and launch the dashboard from the project folder:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

Train or refresh the classifier with `py -3 train_sentiment_model.py`. The trained model and evaluation metrics are saved under `artifacts/`.
