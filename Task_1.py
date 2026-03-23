""" Student Name: [Eaint Taryar Linlat] """
# task 1 - sentiment analysis and topic modelling on fake news data

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.stats import mannwhitneyu, chi2_contingency

for pkg in ["vader_lexicon", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

# figure out where this script lives so we can save outputs there
script_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(script_dir, "fakenews.csv"),
                 usecols=["text", "label"], low_memory=False)

# some labels got messed up because of commas in the text column
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df.dropna(subset=["text", "label"], inplace=True)
df["label"] = df["label"].astype(int)

print(f"Dataset: {len(df)} rows")
print(df["label"].value_counts().to_string())
print()


# sentiment analysis with VADER
print("--- Sentiment Analysis ---\n")

sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

grouped = df.groupby("label")["sentiment"].agg(["mean", "median", "std", "count"])
grouped.index = ["Real (0)", "Fake (1)"]
print(grouped.to_string())

real = df.loc[df["label"] == 0, "sentiment"]
fake = df.loc[df["label"] == 1, "sentiment"]
u, p = mannwhitneyu(real, fake, alternative="two-sided")

print(f"\nMann-Whitney U = {u:,.0f}, p = {p:.4e}")
if p < 0.05:
    direction = "higher" if fake.mean() > real.mean() else "lower"
    print(f"Fake news sentiment is significantly {direction} than real news (p < 0.05).")
else:
    print("No significant difference in sentiment between the two groups.")



# topic modelling with LDA
print("--- Topic Modelling ---\n")

stop = set(stopwords.words("english"))
lem = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [lem.lemmatize(w) for w in text.split()
             if w not in stop and len(w) > 2]
    return " ".join(words)

df["clean"] = df["text"].apply(clean_text)

vectorizer = CountVectorizer(max_features=5000, min_df=5, max_df=0.90)
dtm = vectorizer.fit_transform(df["clean"])
vocab = vectorizer.get_feature_names_out()

# try k=2 to k=10 and pick the best one
ks = range(2, 11)
perp = []
ll = []

print("Fitting LDA for k =", end=" ")
for k in ks:
    print(k, end=" ", flush=True)
    try:
        model = LatentDirichletAllocation(
            n_components=k, random_state=42, max_iter=30,
            learning_method="batch", n_jobs=-1,
            doc_topic_prior=1.0/k, topic_word_prior=0.01
        )
        model.fit(dtm)
        perp.append(model.perplexity(dtm))
        ll.append(model.score(dtm))
    except Exception as e:
        print(f"Error for k={k}: {e}", end=" ")
        perp.append(float('inf'))
        ll.append(float('-inf'))
print()

results = pd.DataFrame({"k": list(ks), "Perplexity": perp, "Log-Likelihood": ll})
print("\n" + results.to_string(index=False))

# elbow method to pick k
diffs = np.diff(perp)
ratios = diffs[1:] / diffs[:-1]
elbow_idx = np.argmin(np.abs(ratios)) + 2
chosen_k = list(ks)[elbow_idx]

print(f"\nLowest perplexity at k = {int(results.loc[results['Perplexity'].idxmin(), 'k'])}")
print(f"Elbow suggests k = {chosen_k}")
print(f"Using k = {chosen_k}\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(ks), perp, "o-", color="#1976D2", label="Perplexity")
ax.axvline(chosen_k, ls="--", color="grey", alpha=0.7, label=f"k = {chosen_k}")
ax.set_xlabel("Number of Topics (k)")
ax.set_ylabel("Perplexity")
ax.set_title("LDA Perplexity vs Number of Topics")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "lda_elbow.png"), dpi=150)
plt.close(fig)
print("Saved: lda_elbow.png")

# final model
print(f"Fitting final model with k = {chosen_k}")
final_lda = LatentDirichletAllocation(
    n_components=chosen_k, random_state=42, max_iter=50,
    learning_method="batch", n_jobs=-1,
    doc_topic_prior=1.0/chosen_k, topic_word_prior=0.01
)
final_lda.fit(dtm)

n_words = 15
print(f"\nTop {n_words} words per topic:")
for i, comp in enumerate(final_lda.components_):
    top = comp.argsort()[-n_words:][::-1]
    words = [vocab[j] for j in top]
    print(f"  Topic {i}: {', '.join(words)}")

# dominant topic per document
topic_dist = final_lda.transform(dtm)
df["topic"] = topic_dist.argmax(axis=1)

ct = pd.crosstab(df["topic"], df["label"], margins=True)
ct.columns = ["Real (0)", "Fake (1)", "Total"]
ct.index = [f"Topic {i}" if i != "All" else "Total" for i in ct.index]
print("\nCross-tabulation:")
print(ct.to_string())

# chi-squared
ct_raw = pd.crosstab(df["topic"], df["label"])
chi2, p_chi, dof, _ = chi2_contingency(ct_raw)
print(f"\nChi-squared = {chi2:.2f}, df = {dof}, p = {p_chi:.4e}")
if p_chi < 0.05:
    print("Topics and labels are significantly associated.")
else:
    print("No significant association between topics and labels.")

proportions = pd.crosstab(df["topic"], df["label"], normalize="index")
proportions.columns = ["Real (0)", "Fake (1)"]
proportions.index = [f"Topic {i}" for i in proportions.index]
print("\nProportions (real/fake per topic):")
print(proportions.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
proportions.plot(kind="bar", stacked=True, color=["#4CAF50", "#F44336"], ax=ax, width=0.6)
ax.set_ylabel("Proportion")
ax.set_xlabel("Dominant Topic")
ax.set_title("Topic Distribution by Label")
ax.legend(title="Label")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "topic_label_distribution.png"), dpi=150)
plt.close(fig)
print("\nSaved: topic_label_distribution.png")

print("\nDone.")