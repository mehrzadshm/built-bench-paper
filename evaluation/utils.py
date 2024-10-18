import random
import logging
import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_sentence_transformer_model(model_name):
    """Loads a SentenceTransformer model;
    If not enough GPU memory, loads model to cpu.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )
        logger.info(f"Successfully loaded model: {_get_model_info(model, model_name)}")
    except torch.cuda.OutOfMemoryError as e:
        logger.warning(f"CUDA out of memory: {e}. Falling back to CPU.")
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )
        logger.info(f"Successfully loaded model: {_get_model_info(model, model_name)}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

    return model


def _get_model_info(model, model_name):
    """Returns model information"""
    model_info = {
        "model_name": model_name,
        "device": model.device,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "num_parameters_mil": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000,0)
    }
    return model_info



def semantic_diversity_sampler(
    df,
    num_samples,
    num_sets,
    embedding_model,
    seed, 
    text_column_name='sentences'
):
    random.seed(seed)
    np.random.seed(seed)

    embeddings = embedding_model.encode(df[text_column_name].tolist())
    
    similarity_matrix = cosine_similarity(embeddings)

    all_selected_sets = []
    available_indices = np.arange(len(embeddings))  # Use numpy array for better performance

    for _ in range(num_sets):
        selected_indices = []

        if available_indices.size:
            first_index = np.random.choice(available_indices)  # pick first sentence randomly
            selected_indices.append(first_index)
            available_indices = np.delete(available_indices, np.where(available_indices == first_index))  # Remove chosen index

        while len(selected_indices) < num_samples and available_indices.size:
            last_selected_embedding = embeddings[selected_indices[-1]]
            similarities = similarity_matrix[selected_indices[-1], available_indices]
            next_index = available_indices[np.argmin(similarities)]  # Select least similar sentence
            selected_indices.append(next_index)
            available_indices = np.delete(available_indices, np.where(available_indices == next_index))

        selected_sentences = df.iloc[selected_indices][text_column_name].tolist()
        all_selected_sets.append(selected_sentences)

    return all_selected_sets



def disjoint_random_sampling(
        samples: List[str],
        num_samples: int,
        num_sets: int,
        seed=42
    ) -> List[List[str]]:
    """
    Random sampling while ensuring maximum use of available samples; creating disjoint sets.
    """
    random.seed(seed)
    all_samples = samples[:]
    used_samples = set()

    results = []

    for _ in range(num_sets):
        # Reset If all samples already used 
        if len(used_samples) == len(all_samples):
            used_samples = set()

        
        available_samples = [s for s in all_samples if s not in used_samples]  # to choose samples not used yet

        if len(available_samples) < num_samples:
            selected_samples = random.sample(available_samples, len(available_samples))
        else:
            selected_samples = random.sample(available_samples, num_samples)

        results.append(selected_samples)

        used_samples.update(selected_samples)

    return results




def similarity_based_negative_sampling(
        query: str,
        samples: List[str],
        model,
        num_samples: int,
        num_sets: int
    ) -> List[List[str]]:
    """
    Negative sampling based on the highest cosine similarity between the query and available samples.
    """
    # Get embedding for the query
    query_embedding = model.encode([query])[0]

    all_samples = samples[:]
    used_samples = set()

    results = []

    # Precompute embeddings for all samples
    sample_embeddings = model.encode(all_samples)

    for _ in range(num_sets):
        # Reset If all samples already used
        if len(used_samples) == len(all_samples):
            used_samples = set()

        # Filter out used samples
        available_samples = [s for s in all_samples if s not in used_samples]

        if not available_samples:
            break

        # Get embeddings for available samples
        available_sample_embeddings = [sample_embeddings[all_samples.index(s)] for s in available_samples]

        # Compute cosine similarities between the query and available samples
        similarities = cosine_similarity([query_embedding], available_sample_embeddings)[0]

        # Sort the available samples based on similarity scores (highest first)
        sorted_indices = np.argsort(-similarities)
        sorted_samples = [available_samples[i] for i in sorted_indices]

        # Select top `num_samples` most similar samples
        selected_samples = sorted_samples[:min(num_samples, len(sorted_samples))]

        results.append(selected_samples)

        # Mark the selected samples as used
        used_samples.update(selected_samples)

    return results
