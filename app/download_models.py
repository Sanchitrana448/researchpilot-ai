"""Download both ONNX models into MODEL_CACHE ahead of time.

Run during the build (`python -m app.download_models`) so a cold start does
not wait on a ~160 MB download from the Hugging Face hub.
"""
from . import embeddings, reranker

if __name__ == "__main__":
    embeddings.warm_up()
    reranker.warm_up()
    print(f"models ready in {embeddings.MODEL_CACHE}")
