"""Retrieval benchmark: 24 passages, 24 keyword queries, 24 paraphrase queries,
6 off-topic queries.

Hand-written for this project, so treat it as a stress test of one property
(does retrieval survive rewording?) rather than a general benchmark.

Rules the data follows, enforced by check_overlap() below:
- a keyword query shares at least two content words with its target passage;
- a paraphrase query shares no content words with its target passage
  (compared on the first five letters of each word, stopwords removed).
"""
import re

PASSAGES = {
    "p01": "Precision measures how many of the items a classifier flags as positive are actually positive. It falls when the model raises many false alarms.",
    "p02": "Recall measures how many of the truly positive cases a classifier manages to find. A fraud model with low recall lets most fraudulent transactions through unnoticed.",
    "p03": "ROC-AUC summarises ranking quality across every possible threshold, so it can look strong even when the default cut-off produces poor decisions for the business.",
    "p04": "Holding out a test set that is never used for tuning keeps the final performance estimate unbiased. Reusing it to pick hyperparameters leaks information and inflates scores.",
    "p05": "Class imbalance occurs when one label is far rarer than the others. Accuracy then becomes misleading, because always predicting the majority class already scores highly.",
    "p06": "Data drift happens when the distribution of input features in production moves away from the training data, for example after a pricing change alters customer behaviour.",
    "p07": "Concept drift means the relationship between inputs and the target changes over time, so a model can degrade even when the inputs look the same as before.",
    "p08": "A model registry stores each trained model with its version, metrics and training data reference, which makes rollbacks and audits straightforward.",
    "p09": "Shadow deployment runs a new model alongside the current one on live traffic without serving its predictions, so teams can compare behaviour safely before switching.",
    "p10": "Containers package an application with its dependencies so it runs the same way on a laptop, a CI runner and a cloud server.",
    "p11": "Continuous integration runs the test suite automatically on every commit, catching regressions before they reach the main branch.",
    "p12": "Health checks let an orchestrator detect a crashed or frozen service and restart it automatically.",
    "p13": "Autoscaling adds or removes server instances based on load, so a service can absorb traffic spikes without paying for idle capacity at night.",
    "p14": "Chunking splits long documents into smaller passages before indexing. Chunks that are too large dilute relevance; chunks that are too small lose surrounding context.",
    "p15": "Dense retrieval encodes queries and passages as vectors and finds neighbours by cosine similarity, which lets it match paraphrases that share no keywords.",
    "p16": "Sparse lexical retrieval such as BM25 or TF-IDF scores documents by overlapping terms. It is fast and precise for exact names, codes and rare words.",
    "p17": "Reciprocal rank fusion merges several ranked lists by summing the inverse of each item's rank, so a passage that several retrievers place near the top rises overall.",
    "p18": "A re-ranker reads the query and each candidate passage together and rescores them, trading extra latency for better ordering of the top results.",
    "p19": "Hallucination is when a language model states something fluent but unsupported by its sources. Requiring citations to retrieved passages makes such claims easier to catch.",
    "p20": "Prompt injection hides instructions inside retrieved content so that the model follows the attacker instead of the user.",
    "p21": "Access control in enterprise search must be enforced at retrieval time; otherwise the assistant can quote documents the user is not permitted to open.",
    "p22": "Personal data such as names and email addresses should be removed or masked before text is logged or used for training.",
    "p23": "Caching repeated queries cuts response time and API spend, but stale cache entries can return outdated answers after the underlying documents change.",
    "p24": "Quantising model weights to eight-bit integers shrinks memory use and speeds up inference on CPUs, at the cost of a small drop in accuracy.",
}

KEYWORD_QUERIES = [
    ("What does precision measure for a classifier?", "p01"),
    ("Why does low recall matter for a fraud model?", "p02"),
    ("Can ROC-AUC look strong while the default threshold gives poor decisions?", "p03"),
    ("Why keep a held-out test set separate from hyperparameter tuning?", "p04"),
    ("Why is accuracy misleading under class imbalance?", "p05"),
    ("What is data drift in production input features?", "p06"),
    ("What is concept drift?", "p07"),
    ("What does a model registry store?", "p08"),
    ("How does shadow deployment compare a new model on live traffic?", "p09"),
    ("Why package an application in containers with its dependencies?", "p10"),
    ("What does continuous integration run on every commit?", "p11"),
    ("What do health checks let an orchestrator do?", "p12"),
    ("How does autoscaling handle traffic spikes?", "p13"),
    ("What happens when chunks are too large or too small?", "p14"),
    ("How does dense retrieval match paraphrases?", "p15"),
    ("When is TF-IDF lexical retrieval precise?", "p16"),
    ("How does reciprocal rank fusion merge ranked lists?", "p17"),
    ("What does a re-ranker trade for better ordering?", "p18"),
    ("How do citations help catch hallucination?", "p19"),
    ("What is prompt injection in retrieved content?", "p20"),
    ("Where must access control be enforced in enterprise search?", "p21"),
    ("Should personal data be masked before logging?", "p22"),
    ("Can caching repeated queries return stale answers?", "p23"),
    ("Does quantising weights to eight-bit integers speed up inference?", "p24"),
]

PARAPHRASE_QUERIES = [
    ("Of everything my detector marks as a hit, what share is genuinely correct?", "p01"),
    ("How many real cheating payments slip past the system without being caught?", "p02"),
    ("Can a single number covering all cutoffs hide that our chosen operating point makes bad calls?", "p03"),
    ("Why should the exam data stay untouched while I adjust settings?", "p04"),
    ("Why can blindly guessing the common answer look impressive when an outcome is very scarce?", "p05"),
    ("Shoppers began acting differently after a discount campaign, and now the incoming values no longer resemble what the system learned from.", "p06"),
    ("The link from signals to the outcome has shifted, although the signals themselves appear unchanged.", "p07"),
    ("Where can we look up which past predictor build we shipped, with its scores, so we can revert?", "p08"),
    ("How can we trial a candidate system on real requests while users never see its output?", "p09"),
    ("How do I make my software behave identically on my own computer and on a rented host?", "p10"),
    ("What verifies each code change by itself so breakages are spotted prior to merging?", "p11"),
    ("How does the platform notice a hung process and bring it back up?", "p12"),
    ("How do we cope with sudden surges in visitors while keeping bills low overnight?", "p13"),
    ("How big should the pieces be when I cut up lengthy files prior to storing them?", "p14"),
    ("Which search approach can locate text that says the same thing in different words?", "p15"),
    ("Which search style excels at matching specific part numbers and uncommon identifiers?", "p16"),
    ("How can I combine the orderings from two different search systems into a single result?", "p17"),
    ("Is a slower second scoring pass worth it to sort the best few hits more accurately?", "p18"),
    ("How can I tell when the chatbot is confidently making things up?", "p19"),
    ("Can a web page trick the assistant into obeying commands hidden in its text?", "p20"),
    ("Could the chatbot reveal a confidential file to an employee who shouldn't be allowed to see it?", "p21"),
    ("Should we strip customer identities out before writing records to logs?", "p22"),
    ("Why might reusing earlier results give out-of-date replies after the knowledge base is edited?", "p23"),
    ("Can I make a network lighter and quicker on ordinary processors by lowering numeric precision?", "p24"),
]

OFF_TOPIC_QUERIES = [
    "What is a good recipe for banana bread?",
    "Who won the 1966 football World Cup?",
    "How do I repot an orchid?",
    "What is the capital of Australia?",
    "How long should I boil an egg?",
    "Which planets have rings?",
]

_STOP = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "that", "this", "these",
    "those", "than", "then", "so", "if", "when", "which", "who", "what", "where", "why", "how", "can",
    "could", "will", "would", "should", "may", "might", "has", "have", "had", "do", "does", "did",
    "i", "we", "my", "our", "us", "you", "they", "them", "their", "not", "no", "into", "too", "very",
    "many", "much", "most", "after", "before", "while", "now", "own", "such", "each", "every", "all", "some", "any", "one", "same", "s", "t",
}


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def check_overlap() -> None:
    """Fail loudly if a query breaks the keyword/paraphrase rules above."""
    for q, pid in KEYWORD_QUERIES:
        shared = _stems(q) & _stems(PASSAGES[pid])
        assert len(shared) >= 2, f"keyword query shares too little with {pid}: {q!r} {shared}"
    for q, pid in PARAPHRASE_QUERIES:
        shared = _stems(q) & _stems(PASSAGES[pid])
        assert not shared, f"paraphrase query shares words with {pid}: {q!r} {shared}"
