from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.generate_goldens_from_docs(
    document_paths=['/home/tagbin/Projects/LLM-Evaluation/datasets/raw_data/1.PLCP_Summary_QA.txt']
)