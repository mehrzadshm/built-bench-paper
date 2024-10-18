import os
import csv
import logging
from dotenv import load_dotenv

# from mteb.models.openai_models import OpenAIWrapper

from evaluation.ClusteringEvaluator import load_dataset, compute_scores as compute_clustering_scores
from evaluation.RetrievalEvaluator import RetrievalTask
from evaluation.RerankingEvaluator import RerankigTask

from mteb.requires_package import requires_package
from typing import Any
import numpy as np



class OpenAIWrapper:
    def __init__(self, model_name: str, embed_dim: int | None = None, **kwargs) -> None:
        requires_package(self, "openai", "Openai text embedding")
        from openai import OpenAI

        self._client = OpenAI()
        self._model_name = model_name
        self._embed_dim = embed_dim

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        requires_package(self, "openai", "Openai text embedding")
        from openai import NotGiven

        if self._model_name == "text-embedding-ada-002" and self._embed_dim is not None:
            logger.warning(
                "Reducing embedding size available only for text-embedding-3-* models"
            )

        max_batch_size = 2048
        sublists = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        all_embeddings = []

        for sublist in sublists:
            response = self._client.embeddings.create(
                input=sublist,
                model=self._model_name,
                encoding_format="float",
                dimensions=self._embed_dim or NotGiven(),
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)
    
    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])


# Logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler('openai_scores.log', mode='a')  
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


load_dotenv()
logger.info(f"Loading environment variables ...")
logger.info(f"Starting OpenAI models eval ...")


PATH_TO_DATASET = "data"
RESULTS_PATH = "results"
FILE_NAME = "main_scores"

logger.info(f"\n\nEvaluation OpenAI models ...")


model_names = [
    "text-embedding-3-large",
    # "text-embedding-3-small"  
]


TASKS = [
    'clustering-s2s',
    'clustering-p2p',
    'retrieval-s2p',
    'retrieval-p2p',
    'reranking-s2p',
    'reranking-p2p'
]



MAIN_METRICS = {
    'clustering': 'v_measure',
    'retrieval': 'ndcg_at_10',
    'reranking':  'map'
}


# check if results folder exists
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
    logger.info(f"Results folder created at: ./{RESULTS_PATH}")
else:
    logger.info(f"Results folder already exists.")

# check if file exists
if os.path.exists(f'{RESULTS_PATH}/{FILE_NAME}.csv'):
    FILE_EXISTS = True
    logger.info(f"Results file already exists; Overwriting scores ...")



# ---------------------- Evaluation ----------------------
# logging scores to csv file

with open(f'{RESULTS_PATH}/{FILE_NAME}.csv', mode='a') as file:
    writer = csv.writer(file)
    writer.writerow(["model", "params (mil)", "embdedding dim"] + TASKS  + ["avg"])
    
    for model_name in model_names:
        logger.info(f"\n\nStarting Evaluation for Model: {model_name} ...")
        
        try:
            model = OpenAIWrapper(model_name=model_name)
            logger.info(f"OpenAI model {model_name} loaded successfully using MTEB wrapper.")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}. Error: {e}")
            continue  # skip to next model 
        
        scores = []
        for task in TASKS:
            try:
                if task.startswith('clustering'):
                    logger.info(f"\nStarting Clustering Task: {task}")
                    dataset = load_dataset(f'{PATH_TO_DATASET}/{task}')
                    scores.append(compute_clustering_scores(model_name, dataset, model=model, model_loaded=True)[MAIN_METRICS['clustering']])
                elif task.startswith('retrieval'):
                    logger.info(f"\nStarting Retrieval Task: {task}")
                    task = RetrievalTask(data_path=f'{PATH_TO_DATASET}/{task}',model_name=model_name, model=model, model_loaded=True)
                    scores.append(task.scores['default'][MAIN_METRICS['retrieval']])
                elif task.startswith('reranking'):
                    logger.info(f"\nStarting Reranking Task: {task}")
                    task = RerankigTask(data_path=f'{PATH_TO_DATASET}/{task}', model_name=model_name, model=model, model_loaded=True)
                    scores.append(task.scores[MAIN_METRICS['reranking']])
                else:
                    logger.error(f"Task: {task} not found.")
            except Exception as e:
                logger.error(f"Failed to compute score for task {task} with model {model_name}. Error: {e}")
                scores.append(None) 

        if all(score is None for score in scores):
            logger.warning(f"All tasks failed for model {model_name}, skipping CSV logging.")
            continue # skip to next task 
        
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            avg_score = round(sum(valid_scores) / len(valid_scores), 4)
        else:
            avg_score = None 
        
        scores = [round(score, 4) if score is not None else 'N/A' for score in scores]
        logger.info(f"\nScores: {scores} | AVG: {avg_score}")
        
        writer.writerow([model_name, "nan", "nan"] + scores + [avg_score])
        logger.info(f"Scores logged to CSV.")
