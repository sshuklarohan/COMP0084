import re
from collections import defaultdict
import matplotlib.pyplot as plt

DATA = "cw-data/passage-collection.txt"
FIGURE_1_TITLE = "Figure 1"
FIGURE_2_TITLE = "Figure 2"
FIGURE_3_TITLE = "Figure 3"
# From NLTK's list of stop words
STOP_WORDS = set(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    ]
)

def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    tokens = text.split()
    return tokens

def remove_stop_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOP_WORDS]

def count_occurences(
    words: list[str],
) -> dict[str, int]:
    word_count = defaultdict(int)
    for word in words:
        word_count[word] += 1
    return word_count

def normalized_frequency(
    word_count: dict[str, int],
) -> dict[str, float]:
    total_words = sum(word_count.values())
    norm_freq = {word: count / total_words for word, count in word_count.items()}
    return norm_freq

def zipf_distribution(N: int, s: float = 1.0) -> list[float]:
    normalizer = sum((1 / (i**s)) for i in range(1, N + 1))
    return [(1 / (k**s)) / normalizer for k in range(1, N + 1)]

def plot_freq_rank(counts: dict[str, int], title: str, is_log_scale: bool = False):
    sorted_freqs = sorted(counts.values(), reverse=True)
    ranks = range(1, len(sorted_freqs) + 1)
    zipf_probs = zipf_distribution(len(sorted_freqs), s=1)
    plt.figure(figsize=(8, 6))
    plt.plot(ranks, sorted_freqs, label="Normalised frequency")
    plt.plot(
    ranks,
    zipf_probs,
    linestyle="--",
    label="Zipf (s = 1)",
    )   
    if is_log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Probability of occurrence")
    plt.title(title)
    plt.legend()
    plt.show()


def task1():
    with open(DATA, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokens = preprocess_text(raw_text)
    word_count = count_occurences(tokens)
    norm_freq = normalized_frequency(word_count)
    vocab_size = len(word_count)
    print(f"Vocabulary Size: {vocab_size}")

    plot_freq_rank(norm_freq, FIGURE_1_TITLE, is_log_scale=False)

    plot_freq_rank(norm_freq, FIGURE_2_TITLE, is_log_scale=True)

    tokens_without_stopwords = remove_stop_words(tokens)
    word_count_no_stop = count_occurences(tokens_without_stopwords)
    norm_freq_no_stop = normalized_frequency(word_count_no_stop)
    vocab_size_no_stop = len(word_count_no_stop)

    print(f"Vocabulary Size after stop-word removal: {vocab_size_no_stop}")
    plot_freq_rank(norm_freq_no_stop, FIGURE_3_TITLE, is_log_scale=True)
    #PLOTS SHOULD SAVE AS THE ZOOMABLE FORMAT TALKED ABOUT IN THE SLIDES


if __name__ == "__main__":
    task1()


"""
Although removing stop words increases the normalised frequency of the remaining terms due to probability 
mass redistribution, this increase is substantially smaller than what Zipf’s law predicts for the highest ranks. Zipf’s law, as defined by Eq. (1) with 
𝑠=1s=1, assumes that the top-ranked terms account for a large proportion of the total probability mass. 
After stop-word removal, the most frequent remaining content words still have relatively modest probabilities,
causing the Zipfian distribution to overestimate their expected frequencies. This effect is further amplified 
by rank reassignment, whereby lower-frequency terms occupy smaller rank indices without acquiring the 
corresponding increase in probability assumed by the Zipf model. Consequently, the empirical distribution 
decays more steeply, leading to a larger divergence from the ideal Zipfian distribution.
"""
