import logging
import time
from datasets import load_dataset, load_from_disk
from mteb.evaluation.evaluators import RerankingEvaluator

from .utils import load_sentence_transformer_model as load_model



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



# TODO: Modify load_data to support load from disk
# TODO: Correct positive/negative num_samples and avg lenght


class RerankigTask(RerankingEvaluator):
    def __init__(self, data_path, model_name, model=None, model_loaded=False, split="test", **kwargs):
        logger.info(f"Starting Reranking Task ...")
        
        self.split = split
        
        self.samples = self.load_data(data_path, split)
        
        super().__init__(self.samples, **kwargs)
        
        if not model_loaded:
            self.model = load_model(model_name)
        else:
            self.model = model
        
        self.scores = self.compute_scores()


    def load_data(self, data_path, split):
        
        dataset = load_from_disk(data_path)
        samples = [sample for sample in dataset[split]]

        query = dataset[split]["query"]
        positive = dataset[split]["positive"]
        negative = dataset[split]["negative"]

        num_samples=len(query)
        num_positive=sum([len(p) for p in positive])
        num_negative=sum([len(n) for n in negative])
        unique_positive=set([item for sublist in positive for item in sublist])
        unique_negative=set([item for sublist in negative for item in sublist])
        avg_query_len=sum([len(q) for q in query]) / len(query)
        avg_positive_len=sum([len(p) for p in unique_positive]) / len(unique_positive)
        avg_negative_len=sum([len(n) for n in unique_negative]) / len(unique_negative)

        logger.info(f"Total queries: {num_samples}; total/unique positives: {num_positive}/{len(unique_positive)}; total/unique negatives: {num_negative}/{len(unique_negative)} ")
        logger.info(f"Average Lengths: [Query : {avg_query_len:.2f}, Positive : {avg_positive_len:.2f}, Negative : {avg_negative_len:.2f}]")
        logger.info(f"Example Query: {query[0]}")
        logger.info(f"Example Positives: {positive[0][:3]}")
        logger.info(f"Example Negatives: {negative[0][:3]}")

        return samples
    
    def compute_scores(self):
        tic = time.time()
        scores = self(self.model)
        logger.info(f"Scores computed in {time.time()-tic:.2f} seconds.")
        logger.info(f"Scores: {scores}")
        return scores

