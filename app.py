from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    posts_path = BASE_DIR / "posts.csv"
    comments_path = BASE_DIR / "comments.csv"
    if not posts_path.exists():
        raise FileNotFoundError(f"Required dataset not found: {posts_path}")
    if not comments_path.exists():
        raise FileNotFoundError(f"Required dataset not found: {comments_path}")

    posts = pd.read_csv(posts_path, encoding="utf-8-sig")
    comments = pd.read_csv(comments_path, encoding="utf-8-sig")

    required_post_columns = {
        "post_id",
        "brand",
        "platform",
        "post_type",
        "likes",
        "comments_count",
        "caption",
    }
    required_comment_columns = {"post_id", "manual_sentiment", "topic"}
    missing_posts = required_post_columns.difference(posts.columns)
    missing_comments = required_comment_columns.difference(comments.columns)
    if missing_posts or missing_comments:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing from posts.csv: {sorted(missing_posts)}; "
            f"missing from comments.csv: {sorted(missing_comments)}."
        )

    for column in ("likes", "comments_count"):
        posts[column] = pd.to_numeric(posts[column], errors="coerce")
    posts["recorded_interactions"] = posts[
        ["likes", "comments_count"]
    ].sum(axis=1, min_count=2)
    return posts, comments


def display_number(value: float | int) -> str:
    return f"{value:,.0f}" if pd.notna(value) else "N/A"


st.set_page_config(
    page_title="Pakistani Social Media Content Detective",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f3f8f6;
        border: 1px solid #dce9e4;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pakistani Social Media Content Detective")
st.caption(
    "Explore post engagement and audience comment labels across Pakistani fashion "
    "and retail brands."
)

posts, comments = load_data()

with st.sidebar:
    st.header("Filter the dataset")
    brands = ["All"] + sorted(posts["brand"].dropna().astype(str).unique().tolist())
    platforms = ["All"] + sorted(posts["platform"].dropna().astype(str).unique().tolist())
    post_types = ["All"] + sorted(posts["post_type"].dropna().astype(str).unique().tolist())
    selected_brand = st.selectbox("Brand", brands)
    selected_platform = st.selectbox("Platform", platforms)
    selected_post_type = st.selectbox("Post format", post_types)

filtered_posts = posts.copy()
if selected_brand != "All":
    filtered_posts = filtered_posts[filtered_posts["brand"] == selected_brand]
if selected_platform != "All":
    filtered_posts = filtered_posts[filtered_posts["platform"] == selected_platform]
if selected_post_type != "All":
    filtered_posts = filtered_posts[filtered_posts["post_type"] == selected_post_type]
filtered_comments = comments[comments["post_id"].isin(filtered_posts["post_id"])]

valid_interactions = filtered_posts["recorded_interactions"].dropna()
median_interactions = valid_interactions.median() if not valid_interactions.empty else float("nan")
known_sentiments = filtered_comments[
    filtered_comments["manual_sentiment"].isin(SENTIMENT_LABELS)
]
positive_count = int((known_sentiments["manual_sentiment"] == "Positive").sum())
positive_share = (
    positive_count / len(known_sentiments) * 100 if not known_sentiments.empty else float("nan")
)

metric_columns = st.columns(4)
metric_columns[0].metric("Posts", f"{len(filtered_posts):,}")
metric_columns[1].metric(
    "Median recorded interactions",
    display_number(median_interactions),
)
metric_columns[2].metric("Related comments", f"{len(filtered_comments):,}")
metric_columns[3].metric(
    "Positive share of labeled comments",
    f"{positive_share:.1f}%" if pd.notna(positive_share) else "N/A",
)
st.caption(
    "Recorded interactions are likes + comments where both values are available; "
    "they are raw counts, not engagement rates. Positive share uses only comments "
    "labeled Positive, Neutral, or Negative."
)

st.subheader("Audience comment labels")
sentiment_column, topic_column = st.columns(2)
with sentiment_column:
    st.markdown("**Sentiment labels**")
    sentiment_counts = (
        known_sentiments["manual_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_LABELS, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Comments")
    )
    if known_sentiments.empty:
        st.info("No positive, neutral, or negative labels match these filters.")
    else:
        st.bar_chart(sentiment_counts, x="Sentiment", y="Comments")
    omitted_count = len(filtered_comments) - len(known_sentiments)
    st.caption(
        f"{omitted_count:,} uncertain, excluded, or unlabeled comments are omitted. "
        "Existing sentiment labels are provisional annotations, not verified ground truth."
    )

with topic_column:
    st.markdown("**Most common comment topics**")
    topics = (
        filtered_comments["topic"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .rename_axis("Topic")
        .reset_index(name="Comments")
    )
    if topics.empty:
        st.info("No topic labels match these filters.")
    else:
        st.bar_chart(topics, x="Topic", y="Comments")

st.subheader("Post performance")
performance_column, platform_column = st.columns(2)
with performance_column:
    st.markdown("**Median recorded interactions by post format**")
    format_summary = (
        filtered_posts.groupby("post_type", dropna=False)
        .agg(
            posts=("post_id", "size"),
            posts_with_both_metrics=("recorded_interactions", "count"),
            median_interactions=("recorded_interactions", "median"),
        )
        .reset_index()
        .rename(
            columns={
                "post_type": "Post format",
                "posts": "Posts",
                "posts_with_both_metrics": "Posts with both metrics",
                "median_interactions": "Median interactions",
            }
        )
    )
    if format_summary["Median interactions"].notna().any():
        st.bar_chart(
            format_summary.dropna(subset=["Median interactions"]),
            x="Post format",
            y="Median interactions",
        )
    else:
        st.info("No posts with both likes and comment counts match these filters.")
    st.dataframe(format_summary, hide_index=True, width="stretch")

with platform_column:
    st.markdown("**Posts by platform**")
    platform_counts = (
        filtered_posts["platform"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Platform")
        .reset_index(name="Posts")
    )
    if platform_counts.empty:
        st.info("No posts match these filters.")
    else:
        st.bar_chart(platform_counts, x="Platform", y="Posts")
    st.caption("Platform coverage is uneven, so small platform samples are not comparable.")

st.subheader("Top posts by recorded interactions")
top_posts = (
    filtered_posts.dropna(subset=["recorded_interactions"])
    .sort_values("recorded_interactions", ascending=False)
    .head(10)
    .copy()
)
if top_posts.empty:
    st.info("No posts with both likes and comment counts match these filters.")
else:
    top_posts["caption"] = top_posts["caption"].fillna("").astype(str).str.slice(0, 140)
    top_posts = top_posts.rename(
        columns={
            "brand": "Brand",
            "platform": "Platform",
            "post_type": "Post format",
            "likes": "Likes",
            "comments_count": "Comments",
            "recorded_interactions": "Recorded interactions",
            "caption": "Caption",
        }
    )
    st.dataframe(
        top_posts[
            [
                "Brand",
                "Platform",
                "Post format",
                "Likes",
                "Comments",
                "Recorded interactions",
                "Caption",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

with st.expander("Try a sentiment prediction"):
    st.write(
        "Enter a comment to classify it with the project's trained sentiment model. "
        "Predictions are exploratory and may be unreliable for mixed-language text."
    )
    comment_text = st.text_area("Comment text", key="prediction_comment")
    if st.button("Predict sentiment"):
        if not comment_text.strip():
            st.warning("Enter a comment before requesting a prediction.")
        else:
            from predict_sentiment import predict_text

            st.success(f"Predicted sentiment: {predict_text(comment_text.strip())}")
