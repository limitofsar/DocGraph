from pathlib import Path
import dill
from src.classifier.catboost_pipeline_class import CatBoostPipeline

def load_catboost_pipeline() -> CatBoostPipeline:
    clf_file = Path(__file__).parent / 'model' / 'catboost_pipeline.pkl'
    with open(clf_file, 'rb') as f:
        return dill.load(f)

clf: CatBoostPipeline = load_catboost_pipeline()
