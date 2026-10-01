import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration
class Config:
    def __init__(self):
        self.n_clusters = 5
        self.n_meta_features = 10
        self.classifier_type = 'mlp_classifier'
        self.profile_type = 'distribution'
        self.cluster_algorithm = 'kmeans'
        self.labeling_strategy = 'clustering'
        self.similarity = 'clustering'
        self.fasttext_dim = 50          # reduced FastText embedding size
        self.fasttext_model_path = None  # overridden by run_config.yaml
        self.top_k_related = 3          # number of related attrs for vicinity/FastText
