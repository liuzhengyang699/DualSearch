import argparse
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from dual_search.search.text_retrieval import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BGE_RERANKER_MODEL,
        DEFAULT_BM25_NAME,
        HybridRetriever,
        HybridRetrieverConfig,
    )
except ModuleNotFoundError:
    from text_retrieval import (
        DEFAULT_BGE_M3_MODEL,
        DEFAULT_BGE_RERANKER_MODEL,
        DEFAULT_BM25_NAME,
        HybridRetriever,
        HybridRetrieverConfig,
    )


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or config.retrieval_topk
    if request.return_scores:
        results, scores = retriever.batch_search(
            request.queries,
            topk=topk,
            return_score=True,
        )
        response = []
        for docs, item_scores in zip(results, scores):
            response.append(
                [
                    {"document": doc, "score": score}
                    for doc, score in zip(docs, item_scores)
                ]
            )
        return {"result": response}

    results = retriever.batch_search(request.queries, topk=topk, return_score=False)
    return {"result": results}


def main():
    parser = argparse.ArgumentParser(description="Launch EVQA BGE-M3 hybrid text retriever.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to bge_m3_Flat.index.")
    parser.add_argument("--bm25_path", type=str, default=None, help="Path to bm25_rank_bm25.pkl.")
    parser.add_argument("--corpus_path", type=str, required=True, help="EVQA text corpus JSONL.")
    parser.add_argument("--retriever_model", type=str, default=DEFAULT_BGE_M3_MODEL)
    parser.add_argument("--reranker_model", type=str, default=DEFAULT_BGE_RERANKER_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--dense_topk", type=int, default=50)
    parser.add_argument("--bm25_topk", type=int, default=50)
    parser.add_argument("--hybrid_topk", type=int, default=100)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16, help="BGE-M3 query encoding batch size.")
    parser.add_argument("--reranker_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--reranker_max_length", type=int, default=1024)
    parser.add_argument("--no_fp16", action="store_false", dest="use_fp16")
    parser.set_defaults(use_fp16=True)
    parser.add_argument("--faiss_gpu", action="store_true", default=False)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    index_dir = args.index_path.rsplit("/", 1)[0] if "/" in args.index_path else "."
    bm25_path = args.bm25_path or f"{index_dir}/{DEFAULT_BM25_NAME}"

    global config, retriever
    config = HybridRetrieverConfig(
        index_path=args.index_path,
        bm25_path=bm25_path,
        corpus_path=args.corpus_path,
        dense_model_path=args.retriever_model,
        reranker_model_path=args.reranker_model,
        device=args.device,
        faiss_gpu=args.faiss_gpu,
        dense_topk=args.dense_topk,
        bm25_topk=args.bm25_topk,
        hybrid_topk=args.hybrid_topk,
        rrf_k=args.rrf_k,
        retrieval_topk=args.topk,
        dense_batch_size=args.batch_size,
        reranker_batch_size=args.reranker_batch_size,
        dense_max_length=args.max_length,
        reranker_max_length=args.reranker_max_length,
        use_fp16=args.use_fp16,
    )
    retriever = HybridRetriever(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
