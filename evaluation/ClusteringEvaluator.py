import time
import logging
import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from mteb.evaluation.evaluators import ClusteringEvaluator

from .utils import load_sentence_transformer_model as load_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_dataset(dataset_path, split="test"):
    dataset = load_from_disk(dataset_path)
    _compute_dataset_statistics(dataset, split)

    if isinstance(dataset[split][0]['sentences'], list):
        logger.info(f"Example sentences: {dataset[split][0]['sentences'][:3]}")
        logger.info(f"Example labels: {dataset[split][0]['labels'][:3]}")
    elif isinstance(dataset[split][0]['sentences'], str):
        logger.info(f"Example sentences: {dataset[split][0]['sentences']}")
        logger.info(f"Example labels: {dataset[split][0]['labels']}")
    
    return dataset


# # compute_scores adapted from mteb (https://github.com/embeddings-benchmark/mteb/)
def compute_scores(
    model_name,
    dataset,
    model=None, model_loaded=False,
    split="test"
):
    scoring_logs = {'model_name': model_name}
    logger.info(f"Clustering evaluation started ...")

    
    if not model_loaded:
        model = load_model(model_name)
    else:
        model = model

    
    v_measures = []
    
    tic = time.time()
    for i, cluster_set in enumerate(tqdm.tqdm(dataset[split], desc="Clustering")):

        # logger.info(f"Processing subset {i} for {set(cluster_set['labels'])} unique labels")

        evaluator = ClusteringEvaluator(
            cluster_set["sentences"], 
            cluster_set["labels"],  
        )
        metrics = evaluator(model)
        v_measure_score = metrics["v_measure"]
        v_measures.append(v_measure_score)

        if v_measure_score > 0.9:  # checking if v-measure is too high
            logger.info(f'Data quality issue in subset {i}: v-measure > 0.9')
        elif v_measure_score < 1/len(set(cluster_set["labels"])):  # checking if v-measure is worse than random clustering
            logger.info(f'Data quality issue in subset {i}: v-measure < 1/|labels|')

  
    v_mean = round(np.mean(v_measures), 4)
    v_std = round(np.std(v_measures), 4)
    
    scores = {
        "v_measure": v_mean, 
        "v_measure_std": v_std, 
        "v_measures": [round(v,4) for v in v_measures]
    }

    scoring_logs.update(scores)
    logger.info(f"Clustering scores: {scoring_logs}")
    logger.info(f"Clustering evaluation finished in {time.time() - tic:.2f} seconds")

    # Offload the model from GPU memory
    del model
    torch.cuda.empty_cache()

    return scores


def _compute_dataset_statistics(dataset, split="test"):
    num_samples_list = []
    unique_labels_list = []
    avg_char_length_list = []

    unique_sentences_set, unique_labels_set = set(), set()
    
    for cluster_set in tqdm.tqdm(dataset[split], desc="Clustering"):
        sentences = cluster_set["sentences"]
        labels = cluster_set["labels"]
        
        # dataset statistics
        num_samples = len(sentences)
        num_samples_list.append(num_samples)
        unique_labels_list.append(len(set(labels)))
        avg_char_length = sum(len(sentence) for sentence in sentences) / num_samples if num_samples > 0 else 0
        avg_char_length_list.append(avg_char_length)

        unique_sentences_set.update(sentences)
        unique_labels_set.update(labels)

    avg_chars = sum(avg_char_length_list) / len(avg_char_length_list) if len(avg_char_length_list) > 0 else 0

    subset_statistics = {
        "num_samples": num_samples_list,
        "unique_labels": unique_labels_list,
        "avg_char_length": [round(length, 2) for length in avg_char_length_list]
    }

    statistics = {
        "num_samples": len(unique_sentences_set),
        "unique_labels": len(unique_labels_set),
        "avg_char_length": round(avg_chars, 2),
        "subset_statistics": subset_statistics 
    }
    
    logger.info(f"Dataset statistics: {statistics}")
    return statistics
