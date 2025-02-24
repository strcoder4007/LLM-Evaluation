from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.generate_goldens_from_docs(
    document_paths=['datasets/raw_data/doc.pdf'],
    max_goldens_per_document=10
)