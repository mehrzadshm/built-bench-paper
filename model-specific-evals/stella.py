import logging
import tqdm
import time
import os
import numpy as np
import torch
import sklearn.cluster
from sklearn import metrics
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from mteb.evaluation.evaluators.RetrievalEvaluator import (
    DRESModel, 
    model_encode,
    RetrievalEvaluator
)
from mteb.evaluation.evaluators import RerankingEvaluator
from mteb.encoder_interface import EncoderWithQueryCorpusEncode
from evaluation.RetrievalEvaluator import RetrievalTask
from evaluation.utils import load_sentence_transformer_model as load_model


# Logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler('stella_rerank_p2p.log', mode='a')  
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




split = 'test'
PATH_TO_DATASET = "data"
parent_dir = os.getcwd()

model_names = [
    'dunzhang/stella_en_400M_v5',
    # 'dunzhang/stella_en_1.5B_v5'
]


# Built-in prompts available from model's config_sentence_transformers.json :
# {
#   "prompts": {
#     "s2p_query": "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",
#     "s2s_query": "Instruct: Retrieve semantically similar text.\nQuery: "
#   }
# }
# We include None as a prompt to compare scores with and without prompting


QUERY_PROMPT_NAMES = [
    # None,
    's2p_query',
    's2s_query',
    
]


# --------- Evaluation for Clustering Tasks ---------

CLUSTERING_TASKS = [
    'clustering-s2s',
    'clustering-p2p',
]

v_measures = []

for model_name in model_names:
    logger.info(f"\n\nStarting Clustering Evaluation for Model: {model_name} ...")
    
    model = load_model(model_name)

    for task in CLUSTERING_TASKS:
        data_path = os.path.join(parent_dir, PATH_TO_DATASET, task)
        logger.info(f"\nStarting Clustering Task: {task}")
        dataset = load_from_disk(data_path)
        
        for prompt_name in QUERY_PROMPT_NAMES:
            logger.info(f"Corpus embedding started with prompt: {prompt_name} ...")
            tic = time.time()
            for i, cluster_set in enumerate(tqdm.tqdm(dataset[split], desc="Clustering")):
                sentences=cluster_set["sentences"] 
                labels=cluster_set["labels"]

                corpus_embeddings = model.encode(sentences, prompt_name=prompt_name, batch_size=32, normalize_embeddings=True)

                logger.info(f"Fitting Mini-Batch K-Means model for subset {i}...")
                clustering_model = sklearn.cluster.MiniBatchKMeans(
                    n_clusters=len(set(labels)),
                    batch_size=32,
                    n_init="auto",
                )
                clustering_model.fit(corpus_embeddings)
                cluster_assignment = clustering_model.labels_

                v_measure_score = metrics.cluster.v_measure_score(labels, cluster_assignment)
                v_measures.append(v_measure_score)

                if v_measure_score > 0.9:  # checking if v-measure is too high
                    logger.info(f'Potential data quality issue in subset {i}: v-measure > 0.9')
                elif v_measure_score < 1/len(set(cluster_set["labels"])):  # checking if v-measure is worse than random clustering
                    logger.info(f'Potential data quality issue in subset {i}: v-measure < 1/|labels|')

                v_mean = round(np.mean(v_measures), 4)
                v_std = round(np.std(v_measures), 4)
                scores = {
                "v_measure": v_mean, 
                "v_measure_std": v_std, 
                "v_measures": [round(v,4) for v in v_measures]
                }

            logger.info(f"Clustering evaluation finished in {time.time() - tic:.2f} seconds")
            logger.info(f"\nScores: {scores}")

    del model
    torch.cuda.empty_cache()



# Custom prompt for clustering tasks
prompt = "Given the description of a built product, identify the category of the product."

for model_name in model_names:
    logger.info(f"\n\nStarting Clustering Evaluation for Model: {model_name} ...")
    
    model = load_model(model_name)

    for task in CLUSTERING_TASKS:
        data_path = os.path.join(parent_dir, PATH_TO_DATASET, task)
        logger.info(f"\nStarting Clustering Task: {task}")
        dataset = load_from_disk(data_path)
        
        logger.info(f"Corpus embedding with custom prompt ...")
        tic = time.time()
        for i, cluster_set in enumerate(tqdm.tqdm(dataset[split], desc="Clustering")):
            sentences=cluster_set["sentences"] 
            labels=cluster_set["labels"]

            corpus_embeddings = model.encode(sentences, prompt=prompt, batch_size=32, normalize_embeddings=True)

            logger.info(f"Fitting Mini-Batch K-Means model for subset {i}...")
            clustering_model = sklearn.cluster.MiniBatchKMeans(
                n_clusters=len(set(labels)),
                batch_size=32,
                n_init="auto",
            )
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_

            v_measure_score = metrics.cluster.v_measure_score(labels, cluster_assignment)
            v_measures.append(v_measure_score)

            if v_measure_score > 0.9:  # checking if v-measure is too high
                logger.info(f'Potential data quality issue in subset {i}: v-measure > 0.9')
            elif v_measure_score < 1/len(set(cluster_set["labels"])):  # checking if v-measure is worse than random clustering
                logger.info(f'Potential data quality issue in subset {i}: v-measure < 1/|labels|')

            v_mean = round(np.mean(v_measures), 4)
            v_std = round(np.std(v_measures), 4)
            scores = {
            "v_measure": v_mean, 
            "v_measure_std": v_std, 
            "v_measures": [round(v,4) for v in v_measures]
            }

        logger.info(f"Clustering evaluation finished in {time.time() - tic:.2f} seconds")
        logger.info(f"\nScores: {scores}")

 




# --------- Evaluation for Retrieval Tasks ---------

RETRIEVAL_TASKS = [
    'retrieval-s2p',
    'retrieval-p2p',
]


class CustomDRESModel(DRESModel):
    def encode_queries(self, queries, *, prompt_name, batch_size, **kwargs):
        return model_encode(
            queries,
            model=self.model,
            prompt_name=prompt_name,  # Pass prompt_name for queries
            batch_size=batch_size,
            **kwargs,
        )

    def encode_corpus(self, corpus, prompt_name, batch_size, **kwargs):
        # Remove 'request_qid' from kwargs
        kwargs.pop('request_qid', None)
        # Ignore prompt_name for corpus encoding
        sentences = [
            (doc.get('title', '') + ' ' + doc['text']).strip() for doc in corpus
        ]
        return model_encode(
            sentences,
            model=self.model,
            prompt_name=None,  # Set prompt_name to None for corpus
            batch_size=batch_size,
            **kwargs,
        )


for model_name in model_names:
    logger.info(f"\n\nStarting Evaluation for Model: {model_name} ...")
    
    model = load_model(model_name)
        
    
    for task in RETRIEVAL_TASKS:
        data_path = os.path.join(parent_dir, PATH_TO_DATASET, task)
        logger.info(f"\nStarting Retrieval Task: {task}")
        task = RetrievalTask(data_path, model_name, model_loaded=True)

        corpus = task.corpus[split]
        queries = task.queries[split]
        qrels = task.relevant_docs[split]

        for prompt_name in QUERY_PROMPT_NAMES:
            logger.info(f"\nStarting Evaluation for Prompt: {prompt_name}")
            
            evaluator = RetrievalEvaluator(
            retriever=CustomDRESModel(model),
            task_name=prompt_name, 
            encode_kwargs={
                'batch_size': 32,
                'show_progress_bar': True
                }
            )

            # Perform retrieval
            results = evaluator(corpus, queries)
            scores = RetrievalEvaluator.evaluate(qrels, results, k_values=[1, 5, 10, 20, 50, 100])
            
            logger.info(f"\nScores for Prompt: {prompt_name}: {scores}")

    



# --------- Evaluation for Reranking Tasks ---------

RERANKING_TASKS = [
    'reranking-s2p',
    'reranking-p2p',
]


class CustomRerankingModel(EncoderWithQueryCorpusEncode):
    def __init__(self, model):
        self.model = model

    def encode_queries(self, queries, *, prompt_name, **kwargs):
        logger.info(f"Custom Encoding Queries with prompt_name: {prompt_name}")
        return model_encode(
            queries,
            model=self.model,
            prompt_name=prompt_name,
            **kwargs,
        )

    def encode_corpus(self, corpus, *, prompt_name=None, **kwargs):
        logger.info(f"Custom Encoding Corpus with prompt_name: {prompt_name}")
        return model_encode(
            corpus,
            model=self.model,
            prompt_name=None,
            **kwargs,
        )
    

class RerankingTask(RerankingEvaluator):
    def __init__(self, 
                 data_path, 
                 model_name, 
                 model=None, 
                 model_loaded=False,
                 task_name=None, 
                 split="test", 
                 **kwargs):
        logger.info(f"Starting Reranking Task ...")

        self.split = split

        self.samples = self.load_data(data_path, split)

        # Initialize the base model
        if not model_loaded:
            base_model = load_model(model_name)
        else:
            base_model = model

        # Wrap the base model with CustomRerankingModel
        self.model = CustomRerankingModel(base_model)

        # Initialize the evaluator
        super().__init__(
            self.samples,
            task_name=task_name,
            **kwargs
        )

        # Compute scores
        self.scores = self.compute_scores()


    def load_data(self, data_path, split):
        dataset = load_from_disk(data_path)
        samples = [sample for sample in dataset[split]]

        # Extract queries, positives, and negatives
        query = [sample["query"] for sample in samples]
        positive = [sample["positive"] for sample in samples]
        negative = [sample["negative"] for sample in samples]

        # Compute statistics for logging
        num_samples = len(query)
        num_positive = sum(len(p) for p in positive)
        num_negative = sum(len(n) for n in negative)
        unique_positive = set(item for sublist in positive for item in sublist)
        unique_negative = set(item for sublist in negative for item in sublist)
        avg_query_len = sum(len(q) for q in query) / len(query)
        avg_positive_len = sum(len(p) for p in unique_positive) / len(unique_positive)
        avg_negative_len = sum(len(n) for n in unique_negative) / len(unique_negative)

        logger.info(
            f"Total queries: {num_samples}; total/unique positives: {num_positive}/{len(unique_positive)}; "
            f"total/unique negatives: {num_negative}/{len(unique_negative)}"
        )
        logger.info(
            f"Average Lengths: [Query: {avg_query_len:.2f}, Positive: {avg_positive_len:.2f}, Negative: {avg_negative_len:.2f}]"
        )
        logger.info(f"Example Query: {query[0]}")
        logger.info(f"Example Positives: {positive[0][:3]}")
        logger.info(f"Example Negatives: {negative[0][:3]}")

        return samples

    def compute_scores(self):
        tic = time.time()
        # Pass the custom model to the evaluator
        scores = self(self.model)
        logger.info(f"Scores computed in {time.time()-tic:.2f} seconds.")
        logger.info(f"Scores: {scores}")
        return scores
    


for model_name in model_names:
    logger.info(f"\n\nStarting Reranking Evaluation for Model: {model_name} ...")
    
    # model = SentenceTransformer(model_name, device='cpu',trust_remote_code=True)
    model = load_model(model_name)

    for task in RERANKING_TASKS:
        data_path = os.path.join(parent_dir, PATH_TO_DATASET, task)
        logger.info(f"\nStarting Reranking Task: {task}")
        
        for prompt_name in QUERY_PROMPT_NAMES:
            reranking_task = RerankingTask(
                data_path=data_path,
                model_name=model_name,
                model=model,
                model_loaded=True,
                task_name=prompt_name,
                encode_kwargs={
                    'batch_size': 32,
                    'show_progress_bar': True
                    }
                )
            logger.info(f"\nScores for Prompt: {prompt_name}: {reranking_task.scores}")

    del model
    torch.cuda.empty_cache()

