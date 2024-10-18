import logging
from collections import defaultdict
from time import time
from typing import Dict, Tuple
import torch
from datasets import Features, Value, load_dataset
from sentence_transformers import SentenceTransformer
from mteb.evaluation.evaluators import RetrievalEvaluator

from mteb.tasks import AbsTaskRetrieval
from mteb.tasks.Retrieval import calculate_length  

from .utils import load_sentence_transformer_model as load_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# search function adapted from the search function in MTEB's RetrievalEvaluator class
def search(
    model,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    prompt_name: str,
    top_k: int = 1000,
    
) -> dict[str, dict[str, float]]:
    # Encode queries with prompt_name
    logger.info("Encoding Queries with prompt_name.")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = model.encode(
        query_texts,
        prompt_name=prompt_name,
    )

    # Encode corpus without prompt_name
    logger.info("Encoding Corpus without prompt_name.")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_embeddings = model.encode(
        corpus_texts
    )

    # Convert embeddings to tensors if they aren't already
    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.tensor(query_embeddings)
    if not isinstance(corpus_embeddings, torch.Tensor):
        corpus_embeddings = torch.tensor(corpus_embeddings)

    # Normalize embeddings to unit vectors
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)

    # Compute cosine similarities
    logger.info("Computing Cosine Similarities.")
    cos_scores = torch.mm(query_embeddings, corpus_embeddings.t())

    # Get top_k results for each query
    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
        cos_scores, top_k, dim=1, largest=True, sorted=True
    )

    # Prepare the results
    results = {}
    for query_idx, qid in enumerate(query_ids):
        scores = cos_scores_top_k_values[query_idx].cpu().tolist()
        indices = cos_scores_top_k_idx[query_idx].cpu().tolist()
        results[qid] = {corpus_ids[idx]: score for idx, score in zip(indices, scores)}

    return results



class CSVDataLoader:
    def __init__(
        self,
        corpus_file,
        query_file,
        qrels_file,
        streaming: bool = False,
        keep_in_memory: bool = False,
    ):
        self.corpus_file = corpus_file
        self.query_file = query_file
        self.qrels_file = qrels_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

        self.corpus = {}
        self.queries = {}
        self.qrels = {}

    
    def load(
        self, split="test"
    ) -> Tuple[Dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
       
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self._load_qrels(split)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])

        return self.corpus, self.queries, self.qrels
        

    def _load_corpus(self):
        corpus_ds = load_dataset(
            "csv",
            data_files=self.corpus_file,
            streaming=self.streaming,
            keep_in_memory=self.keep_in_memory,
        )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        queries_ds = load_dataset(
            "csv",
            data_files=self.query_file,
            streaming=self.streaming,
            keep_in_memory=self.keep_in_memory,
        )
        queries_ds = next(iter(queries_ds.values()))  # get first split
        queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds

        
    def _load_qrels(self, split):
        qrels_ds = load_dataset(
            "csv",
            data_files=self.qrels_file,
            delimiter=",",
            keep_in_memory=self.keep_in_memory,
        )
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds



class RetrievalTask(AbsTaskRetrieval):
    def __init__(
        self,
        data_path,
        model_name,
        model=None, model_loaded=False,
        prompt_name=None,
        **kwargs
    ):
        
        if not model_loaded:
            self.model = load_model(model_name)
        else:
            self.model = model


        self.corpus_file = f'{data_path}/corpus.csv'
        self.query_file = f'{data_path}/queries.csv'
        self.qrels_file = f'{data_path}/qrels.csv'

        self.prompt_name = prompt_name 
       
        super().__init__(**kwargs)

        logger.info("Retrieval Task Initialized ...")

        self.load_data()
        
        self.scores = self.evaluate(self.model)
        logger.info(f"Scores: {self.scores}")
    
        

    def load_data(self, split="test", **kwargs):
            if self.data_loaded:
                return
            self.corpus, self.queries, self.relevant_docs = {}, {}, {}
            corpus, queries, qrels = CSVDataLoader(  
                corpus_file=self.corpus_file,
                query_file=self.query_file,
                qrels_file=self.qrels_file,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # Conversion from DataSet
            queries = {query["id"]: query["text"] for query in queries}
            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )
    
            self.data_loaded = True

            query_len , corpus_len = calculate_length(self.queries["test"], self.corpus["test"])
            
        
            logger.info(f'query length average: {query_len:.2f}')
            logger.info(f'corpus length average: {corpus_len:.2f}')
        

    def evaluate(
        self,
        model,
        split: str = "test"
    ):
        
        retriever = RetrievalEvaluator(retriever=model)
        
        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries, relevant_docs, hf_subset
            )
        logger.info(f"Scores: {scores}")
        return scores
    

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
    ):
        start_time = time()
        # results = retriever(corpus, queries)

        if self.prompt_name:
            # use custom search function
            results = search(
                model=self.model,
                corpus=corpus,
                queries=queries,
                prompt_name=self.prompt_name,
                top_k=max(retriever.k_values)
            )
        else:
            # Use the default retriever
            results = retriever(corpus, queries)
        end_time = time()
        logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")


        ndcg, _map, recall, precision, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs.items()
            },
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs_mrr.items()
            },
        }
        
        return scores

