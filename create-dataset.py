import time
import json
import argparse
import logging
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from mteb.evaluation.evaluators import ClusteringEvaluator

from evaluation.ClusteringEvaluator import _compute_dataset_statistics
from evaluation.utils import (
    semantic_diversity_sampler as diversity_sampler,
    similarity_based_negative_sampling as negative_sampler,
    disjoint_random_sampling as random_sampler,
    load_sentence_transformer_model as load_model
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
   logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(), 
            logging.FileHandler('logs.log', mode='a')  
        ]
    )
   

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for dataset creation.")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='dataset_config.json', 
        help='Path to the config JSON file for dataset creation.'
    )
    parser.add_argument("--task_type", type=str, help='clustering, retrieval, reranking')
    parser.add_argument("--dataset_category", type=str, help='s2s or p2p')
    parser.add_argument("--output_path", type=str, default="data")
    
    return parser.parse_args()


def create_clustering_dataset(
    subsets, 
    dataset_category, 
    output_path, 
    embedding_model_name, 
    seed
):
    tic = time.time()

    df = pd.DataFrame(columns=["sentences", "labels"])
    
    embedding_model = SentenceTransformer(embedding_model_name)
    
    for subset in subsets:

        logger.info(f"Processing subset for labels: {subset['labels']}")

        corpus_df = pd.read_csv(subset['df_path'])  
        labels = subset['labels']
        num_samples = subset['num_samples']
        num_sets = subset['num_sets']
        text_column_name = subset['text_column_name']
        label_column_name = subset['label_column_name']
        
        subset_df = create_clustering_subsets(
            corpus_df, labels, num_samples, num_sets,
            text_column_name, label_column_name,
            embedding_model, seed
        )
        
        df = pd.concat([df, subset_df], ignore_index=True)

    dataset = DatasetDict({"test": Dataset.from_pandas(df)})
    output_file_name = f'clustering-{dataset_category}'
    logger.info(f"Dataset statistics: {_compute_dataset_statistics(dataset, 'test')}")
    dataset.save_to_disk(f'{output_path}/{output_file_name}')

    toc = time.time()
    logger.info(f"Clustering dataset created in {toc - tic:.2f}s.")
    logger.info(f"Clustering dataset and saved to {output_path}/{output_file_name}")

    pass



def create_clustering_subsets(
    df, 
    labels, 
    num_samples, 
    num_sets, 
    text_column_name,
    label_column_name, 
    embedding_model,
    seed
):
    sampled_data = {}
    hits = 0
    dataset_df = pd.DataFrame(columns=["sentences", "labels"])

    if isinstance(labels, dict):
        label_mapping = labels
        labels = list(label_mapping.keys())
    else:
        label_mapping = {label: label for label in labels}

    for label in labels:
        df_filtered = df[df[label_column_name] == label]
        if df_filtered.empty:
            raise ValueError(f"Label '{label}' not found in the dataset.")
        df_extracted = pd.DataFrame({
            'sentences': df_filtered[text_column_name],
            'labels': df_filtered[label_column_name]
        })
        mapped_label = label_mapping[label]
        sampled_data.update({mapped_label: diversity_sampler(df_extracted, num_samples, num_sets, embedding_model, seed)})

    for i in range(num_sets):
        sentences, labels = [], []
        for mapped_label in sampled_data.keys():
            new_sentences = sampled_data[mapped_label][i]
            if len(new_sentences):
                sentences.extend(new_sentences)
                labels.extend([mapped_label] * len(new_sentences))
            else:
                new_sentences = sampled_data[mapped_label][0]
                sentences.extend(new_sentences)
                labels.extend([mapped_label] * len(new_sentences))
        
            # Check clustering quality with upper and lower tresholds
            clustering_evaluator = ClusteringEvaluator(sentences, labels)
            upper_score = clustering_evaluator(embedding_model)["v_measure"]  # uses same baseline model used for diversity sampling
        
            lower_treshold_embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            lower_score = clustering_evaluator(lower_treshold_embedding_model)["v_measure"]
        
            shuffled_data = pd.DataFrame({"sentences": [], "labels": []})

            if upper_score < 0.8 and lower_score > 1/len(set(labels)):
                paired_data = list(zip(sentences, labels))
                indices = list(range(len(paired_data)))
                random.shuffle(indices)

                shuffled_sentences = [sentences[i] for i in indices]
                shuffled_labels = [labels[i] for i in indices]

                shuffled_data = pd.DataFrame({"sentences": [list(shuffled_sentences)], "labels": [list(shuffled_labels)]})
                logger.info(f"Subset {i} shuffled; num_sentences: {len(sentences)}; unique labels: {set(labels)}; v-measure: [upper: {upper_score:.4f}, lower:{lower_score:.4f}].")
                
            else:
                hits += 1
                logger.info(f"Subset {i} skipped; v-measure: [upper: {upper_score:.4f}, lower:{lower_score:.4f}].")
        dataset_df = pd.concat([dataset_df, shuffled_data], ignore_index=True)
    logger.info(f"{hits} subsets were skipped due to unacceptable clustering quality treshold.")
        
    return dataset_df 



# ----------------- Reranking Functions -----------------

def create_reranking_dataset(
    config,
    embedding_model_name, 
    data_path='data',
    seed=42,
    neg_sampling_method='similarity',
    split='test'
):
    """
    Create reranking datset as Dataset object with the form:
    [{'query': str, 'positive': [str], 'negative': [str]}]

    Parameters:
    -----------
    TODO: Add parameters description

    """
    corpus_text_col_name=config['corpus_text_col_name']
    query_col_name=config['query_col_name']
    num_positive_samples=config['num_positive_samples']
    pos_neg_ratio=config['pos_neg_ratio']
    max_num_sets=config['max_num_sets']
    neg_sampling_method=config['neg_sampling_method']
    
    
    tic = time.time()
    model = load_model(embedding_model_name)
    
    df_queries = pd.read_csv(f'{data_path}/ifc_entities.csv')
    df_corpus = pd.read_csv(f'{data_path}/uniclass_entities.csv')
    df_qrels = pd.read_csv(f'{data_path}/qrels.csv')

    rel_counts = df_qrels['query-id'].value_counts()

    reranking_samples = []
    
    for item in rel_counts.items():
        idx_count = item[1]
        if idx_count > num_positive_samples:
            target_idx = item[0]

            num_pos_samples = num_positive_samples
            num_neg_samples = num_positive_samples * pos_neg_ratio
            num_sets = min(max_num_sets, idx_count//num_positive_samples)

            corpus_ids = df_qrels[df_qrels['query-id'] == target_idx]['corpus-id'].values
            df_positives = df_corpus[df_corpus['entity_id'].isin(corpus_ids)]
            df_negatives = df_corpus[-df_corpus['entity_id'].isin(corpus_ids)]

            query = df_queries[df_queries["entity_id"] == target_idx][query_col_name].values[0]
            logger.info(f'positive/negative sampling for query: "{query}";  num_sets: {num_sets}, num_pos_samples: {num_pos_samples}, num_neg_samples: {num_neg_samples}')

            positive_sets = diversity_sampler(df_positives, num_pos_samples, num_sets, model, seed=seed, text_column_name=corpus_text_col_name)

            negative_samples_pool = df_negatives[corpus_text_col_name].tolist()

            if neg_sampling_method == 'similarity':
                negative_sets = negative_sampler(
                    query=query,
                    samples=negative_samples_pool,
                    model=model,
                    num_samples=num_neg_samples,
                    num_sets=num_sets
                )
            elif neg_sampling_method == 'random':
                negative_sets = random_sampler(negative_samples_pool, num_neg_samples, num_sets, seed=seed)
            else:
                raise ValueError(f"Invalid negative sampling method: {neg_sampling_method}")

            for sample_set in range(num_sets):
                reranking_samples.append({'query': query, 'positive': positive_sets[sample_set], 'negative': negative_sets[sample_set]})

    
    logger.info(f"Reranking dataset with {len(reranking_samples)} rows created in {time.time() - tic:.2f}s.")

    dataset = DatasetDict({split: Dataset.from_list(reranking_samples)})

    return dataset




def main(args):
    setup_logging()

    with open(args.config_path, 'r') as file:
        config = json.load(file)
    
    DEFAULT_BASELINE_EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    DEFAULT_SEED = 42
    baseline_embedding_model_name = config.get('baseline_embedding_model', DEFAULT_BASELINE_EMBEDDING_MODEL)
    seed = config.get('seed', DEFAULT_SEED)

    if args.task_type is None or args.dataset_category is None:
        logger.error("--task_type and --dataset_category are required arguments.")
        return
    
    elif args.task_type == 'clustering':
        logger.info(f"Creating clustering datasets using baseline model: {baseline_embedding_model_name} and seed: {seed}")
        create_clustering_dataset(
            config['clustering'][f'{args.dataset_category}_subsets'], 
            args.dataset_category, args.output_path,
            baseline_embedding_model_name, seed
        )
    
    elif args.task_type == 'retrieval':
        logger.error(f"Task type {args.task_type} is not supported.")
    
    elif args.task_type == 'reranking':
        dataset = create_reranking_dataset(
            config=config['reranking'][f'{args.dataset_category}_subsets'],
            embedding_model_name=baseline_embedding_model_name, seed=seed,
            data_path=args.output_path, split='test'
        )
        dataset.save_to_disk(f'{args.output_path}/rerank-{args.dataset_category}')
        logger.info(f"Reranking dataset saved to {args.output_path}/rerank-{args.dataset_category}")
    
    else:
        logger.error(f"Task type {args.task_type} is not supported.")


if __name__ == "__main__":

    args = get_args()
    main(args)
    