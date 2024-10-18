import os
import csv
import logging
import argparse

from evaluation.ClusteringEvaluator import load_dataset, compute_scores as compute_clustering_scores
from evaluation.RetrievalEvaluator import RetrievalTask
from evaluation.RerankingEvaluator import RerankigTask
from evaluation.utils import (
    load_sentence_transformer_model as load_model,
    _get_model_info as get_model_info
)

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



PATH_TO_DATASET = "data"
RESULTS_PATH = "results"
FILE_NAME = "main_scores"


model_names = [
    # 'Alibaba-NLP/gte-Qwen2-7B-instruct',
    # 'Alibaba-NLP/gte-base-en-v1.5',
    # 'Alibaba-NLP/gte-large-en-v1.5',
    # 'BAAI/bge-base-en-v1.5',
    # 'BAAI/bge-large-en-v1.5',
    # 'BAAI/bge-large-zh-v1.5',
    # 'Salesforce/SFR-Embedding-2_R',
    # 'WhereIsAI/UAE-Large-V1',
    # 'avsolatorio/GIST-Embedding-v0',
    # 'avsolatorio/GIST-large-Embedding-v0',
    # 'dunzhang/stella_en_1.5B_v5',
    # 'dunzhang/stella_en_400M_v5',
    # 'intfloat/e5-base-v2',
    # 'intfloat/e5-large-v2',
    # 'intfloat/multilingual-e5-large-instruct',
    # 'intfloat/multilingual-e5-small',
    # 'mixedbread-ai/mxbai-embed-large-v1',
    # 'nvidia/NV-Embed-v2',
    # 'sentence-transformers/all-MiniLM-L12-v2',
    # 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    # 'thenlper/gte-base',
    # 'thenlper/gte-large',
    # 'thenlper/gte-small'
]


TASKS = [
    # 'clustering-s2s',
    # 'clustering-p2p',
    # 'retrieval-s2p',
    # 'retrieval-p2p',
    # 'reranking-s2p',
    # 'reranking-p2p'
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
            model = load_model(model_name)
            model_info = get_model_info(model, model_name)
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
        
        writer.writerow([model_name, model_info['num_parameters_mil'], model_info['embedding_dimension']] + scores + [avg_score])
        logger.info(f"Scores logged to CSV.")





# # ----------------- Clustering Tasks -----------------

# dataset_names = [
#     "clustering-s2s",
#     # "clustering-p2p"
# ]


# for dataset_name in dataset_names:
#     dataset = load_dataset(f'{PATH_TO_DATASET}/{dataset_name}')
#     for model_name in model_names:
#         scores = compute_clustering_scores(model_name, dataset)


# # ----------------- Retrieval Tasks -----------------

# task = RetrievalTask(
#     data_path="data/retrieval-p2p",
#     model_name="thenlper/gte-small",
#     split="test"
# )
# scores = task.scores
# logger.info(f"Scores: {scores}")


# # # ----------------- Reranking Tasks -----------------
# # data should be Dataset object (on disk)
# data_path="data/rerank-s2p"
# logger.info(f"Loading Reranking Task: {data_path}")

# task = RerankigTask(
#     data_path=data_path,
#     model_name= "thenlper/gte-small",
#     split='test',
#     mrr_at_k=10
# )

# logger.info(f"Scores: {task.scores}")



# if __name__ == "__main__":

#     args = get_args()
#     main(args)