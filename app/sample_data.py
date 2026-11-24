"""Small built-in corpus so the demo works with one click.

The passages are original text written for this project. Two of them
deliberately disagree about chunk size so the contradiction critic has
something to find.
"""

SAMPLE_DOCS = [
    {
        "source": "rag-risks-overview.txt",
        "text": (
            "Retrieval-augmented generation grounds a language model in retrieved "
            "documents. The main risks of adopting RAG for enterprise search are "
            "stale indexes, poor chunking that loses context, and access-control "
            "leaks when documents are indexed without permission metadata. "
            "Teams that skip evaluation usually discover these problems only after "
            "users lose trust in the answers."
        ),
    },
    {
        "source": "rag-evaluation-guide.txt",
        "text": (
            "Evaluation of a RAG system should measure retrieval recall and citation "
            "accuracy separately from answer fluency. A fluent answer that cites a "
            "passage which does not support it is worse than a plain answer, because "
            "it looks trustworthy. A useful first check is that every citation marker "
            "in the answer maps to a passage that was actually retrieved."
        ),
    },
    {
        "source": "chunking-notes.txt",
        "text": (
            "Chunking strategy changes retrieval quality. Smaller chunks of around "
            "200 words improve precision because each passage covers one idea. "
            "Chunk overlap of 10 to 20 percent reduces the chance that an answer is "
            "split across a boundary. Very large chunks dilute the match score and "
            "pull in unrelated text."
        ),
    },
    {
        "source": "chunking-counterpoint.txt",
        "text": (
            "Chunking strategy changes retrieval quality. Smaller chunks do not "
            "improve precision on long technical documents, because they lose the "
            "surrounding context that explains each idea. Larger chunks of around "
            "800 words keep that context and do not dilute the match score when the "
            "document is well structured."
        ),
    },
]

SAMPLE_QUESTION = "What are the main risks of adopting RAG for enterprise search?"

# (question, source file that should appear in the retrieved evidence)
EVAL_QUESTIONS = [
    (SAMPLE_QUESTION, "rag-risks-overview.txt"),
    ("How should a RAG system be evaluated?", "rag-evaluation-guide.txt"),
    ("Why does retrieval recall matter separately from answer fluency?", "rag-evaluation-guide.txt"),
    ("What goes wrong when documents are indexed without permission metadata?", "rag-risks-overview.txt"),
    ("Do smaller chunks improve retrieval precision?", "chunking-notes.txt"),
    ("How much chunk overlap reduces answers split across a boundary?", "chunking-notes.txt"),
    ("Do larger chunks keep more context in technical documents?", "chunking-counterpoint.txt"),
    ("Why does stale index data hurt enterprise search?", "rag-risks-overview.txt"),
]
