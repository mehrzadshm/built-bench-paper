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
        logging.FileHandler('scores.log', mode='a')  
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



split = 'test'
PATH_TO_DATASET = "data"
parent_dir = os.getcwd()


# Model
model_name = 'intfloat/multilingual-e5-large-instruct'
model = SentenceTransformer(model_name, device='cuda', trust_remote_code=True)

logger.info("\n"*3)
logger.info(f"Evaluating with instruction-based prompt for model {model_name}")


TASK_INSTRUCT= {
    "clustering": "Given the description of a built product, identify the category of the product similar to classification systems such as IFC or Uniclass.",
    "retrieval": "Retrieve relevant Uniclass built product descriptions that are semantically similar to the provided IFC product description",
    "reranking": "Retrieve relevant Uniclass built product descriptions that are semantically similar to the provided IFC product description"
}




# --------- Evaluation for Clustering Tasks ---------

CLUSTERING_TASKS = [
    'clustering-s2s',
    'clustering-p2p'
]
   
QUERY_PROMPT = "Instruct: "+TASK_INSTRUCT['clustering']+"\nQuery: "


for task in CLUSTERING_TASKS:
    logger.info(f"\n\nStarting {task} task Eval for Model: {model_name} with instruction prompt:\n{QUERY_PROMPT}")

    dataset = load_from_disk(f'{PATH_TO_DATASET}/{task}')

    v_measures = []

    tic = time.time()
    for i, cluster_set in enumerate(tqdm.tqdm(dataset[split], desc="Clustering")):
        
        sentences=cluster_set["sentences"] 
        labels=cluster_set["labels"]

        logger.info("Corpus embedding started ...")

        corpus_embeddings = model.encode(
            sentences, 
            prompt=QUERY_PROMPT, # encoding with task instruction prompt
            batch_size=32, normalize_embeddings=True
        )

        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(labels)),
            batch_size=32,
            n_init="auto",
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        v_measure_score = metrics.cluster.v_measure_score(labels, cluster_assignment)

        v_measures.append(v_measure_score)

        v_mean = round(np.mean(v_measures), 4)
        v_std = round(np.std(v_measures), 4)

        scores = {
        "v_measure": v_mean, 
        "v_measure_std": v_std, 
        "v_measures": [round(v,4) for v in v_measures]
        }

    logger.info(f"Clustering evaluation finished in {time.time() - tic:.2f} seconds;\nScores: {scores}")




# --------- Evaluation for Retrieval Tasks ---------

class CustomDRESModel(DRESModel):
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def encode_queries(self, queries, *, batch_size, **kwargs):
        logger.info(f"\nEncoding of QUERIES with Prompt: {self.prompt}")
        return self.model.encode(
            queries,
            prompt=self.prompt, 
            batch_size=batch_size,
            **kwargs,
        )

    def encode_corpus(self, corpus, batch_size, **kwargs):
        logger.info(f"\nEncoding of CORPUS with no Prompt.")
        # Remove 'request_qid' from kwargs
        kwargs.pop('request_qid', None)
        # Ignore prompt_name for corpus encoding
        sentences = [
            (doc.get('title', '') + ' ' + doc['text']).strip() for doc in corpus
        ]
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            **kwargs,
        )



RETRIEVAL_TASKS = [
    'retrieval-s2p', 
    'retrieval-p2p'
]

QUERY_PROMPT = "Instruct: "+TASK_INSTRUCT['retrieval']+"\nQuery: "


for task in RETRIEVAL_TASKS:
    data_path = os.path.join(parent_dir, PATH_TO_DATASET, task)
    logger.info(f"\n\nStarting {task} task for Model: {model_name} with instruction prompt:\n{QUERY_PROMPT}")

    task = RetrievalTask(data_path, model_name, model_loaded=True)

    corpus = task.corpus[split]
    queries = task.queries[split]
    qrels = task.relevant_docs[split]
    
    evaluator = RetrievalEvaluator(
        retriever=CustomDRESModel(model=model, prompt=QUERY_PROMPT),
        encode_kwargs={
            'batch_size': 32,
            'show_progress_bar': True
            }
    )
    # Perform retrieval
    results = evaluator(corpus, queries)
    scores = RetrievalEvaluator.evaluate(qrels, results, k_values=[1, 5, 10, 20, 50, 100])
    
    logger.info(f"\nScores:  {scores}")



# --------- Evaluation for Reranking Tasks ---------

class CustomRerankingModel(EncoderWithQueryCorpusEncode):
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def encode_queries(self, queries, *, prompt_name=None, **kwargs):
        logger.info(f"\nEncoding of QUERIES with Prompt: {self.prompt}")
        return self.model.encode(
            queries,
            prompt=self.prompt, # encoding with task instruction prompt
            **kwargs,
        )

    def encode_corpus(self, corpus, *, prompt_name=None, **kwargs):
        logger.info(f"Custom Encoding Corpus with no prompt")
        return self.model.encode(
            corpus,
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
            self.model = load_model(model_name)
        else:
            self.model = model

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



RERANKING_TASKS = [
    'reranking-s2p',
    'reranking-p2p',
]

QUERY_PROMPT = "Instruct: "+TASK_INSTRUCT['reranking']+"\nQuery: "


model = CustomRerankingModel(model=model, prompt=QUERY_PROMPT)


for task in RERANKING_TASKS:
    data_path = os.path.join(parent_dir, PATH_TO_DATASET, task)
    logger.info(f"\n\nStarting Reranking {task} task for Model: {model_name} with instruction prompt:\n{QUERY_PROMPT}")

    
    reranking_task = RerankingTask(
        data_path=data_path,
        model_name=model_name,
        model=model,
        model_loaded=True,
        encode_kwargs={
            'batch_size': 32,
            'show_progress_bar': True
            }
        )
    logger.info(f"\nScores: {reranking_task.scores}")

del model
torch.cuda.empty_cache()
