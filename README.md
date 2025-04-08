# Built-Bench-Paper
[![arXiv](https://img.shields.io/badge/arXiv-2411.12056-red)](https://arxiv.org/abs/2411.12056)
[![MTEB Benchmark](https://img.shields.io/badge/MTEB-Integrated-green)](https://github.com/embeddings-benchmark/mteb)
[![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-Datasets-orange)](https://huggingface.co/datasets/mehrzad-shahin/)



Resources &amp; scripts for the paper: **["Benchmarking Pre-trained Text Embedding Models in Aligning Built Asset Information"](https://arxiv.org/abs/2411.12056)**

## Overview
**BuiltBench** is a benchmark designed to evaluate pre-trained text embedding models in the **Built Asset Information Management** domain. 

It is now officially **integrated into MTEB** and available for use in model evaluations.


## Updates
- 🏆 **Now part of MTEB's official repo**
    - Integrated into **MTEB v1.34.15** → [Release details](https://github.com/embeddings-benchmark/mteb/releases/tag/1.34.15)
- 📂 BuiltBench **Datasets are publicly available on Hugging Face**  
  - See dataset links below.
    - [BuiltBench-clustering-p2p](https://huggingface.co/datasets/mehrzad-shahin/builtbench-clustering-p2p)
    - [BuiltBench-clustering-s2s](https://huggingface.co/datasets/mehrzad-shahin/builtbench-clustering-s2s)
    - [BuiltBench-retrieval-s2p](https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-retrieval)
    - [BuiltBench-reranking-s2p](https://huggingface.co/datasets/mehrzad-shahin/BuiltBench-reranking)


## 🔧 Installation & Usage
**Install MTEB and Dependencies (v1.34.15 or later)**

- Run all benchmark tasks:
    ```python
    import mteb

    # Load BuiltBench
    benchmark = mteb.get_benchmark("BuiltBench(eng)")

    # Load model (compatible with MTEB or SentenceTransformer)
    model = ...  

    # Run evaluation
    evaluation = mteb.MTEB(tasks=benchmark.tasks)
    evaluation.run(model)
    ```
- Run tasks individually:
    ```python
    import mteb

    # Load model (compatible with MTEB or SentenceTransformer)
    model = ... 

    """ 
    Names of the current tasks included in the benchamrk: 
     - BuiltBenchClusteringP2P
     - BuiltBenchClusteringS2S
     - BuiltBenchRetrieval
     - BuiltBenchReranking
    """

    tasks = mteb.get_tasks(tasks=["BuiltBenchClusteringP2P"]) 
        
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model)
    ```


# How to cite
```
@article{shahinmoghadam2024benchmarking,
  title={Benchmarking pre-trained text embedding models in aligning built asset information},
  author={Shahinmoghadam, Mehrzad and Motamedi, Ali},
  journal={arXiv preprint arXiv:2411.12056},
  year={2024}
}
```
