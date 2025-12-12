"""
Main execution script
"""

import os
import kagglehub
from src.data_preparation import NewsDatasetPreparator
from src.model_training import HeadlineGeneratorTrainer
from src.model_evaluation import ModelEvaluator


def main():
    os.environ["WANDB_DISABLED"] = "true"
    
    print("="*70)
    print("TURKISH NEWS HEADLINE GENERATOR")
    print("="*70)
    
    # Step 1: Data
    print("\n[STEP 1] DATASET PREPARATION")
    path = kagglehub.dataset_download("furkanozbay/turkish-news-dataset")
    xls_path = os.path.join(path, "news.xls")
    
    preparator = NewsDatasetPreparator(min_samples=500, max_samples=2000)
    dataset_path = "data/processed/turkish_news_sft.jsonl"
    preparator.prepare_full_pipeline(xls_path, dataset_path)
    
    # Step 2: Train
    print("\n[STEP 2] MODEL TRAINING")
    trainer = HeadlineGeneratorTrainer()
    model, tokenizer = trainer.load_model()
    dataset = trainer.prepare_training_data(dataset_path)
    trainer.train(dataset, output_dir="outputs", max_steps=800)
    
    model_save_path = "models/llama32_turkish_news_ft"
    trainer.save_model(model_save_path)
    
    # Step 3: Evaluate
    print("\n[STEP 3] EVALUATION")
    test_articles = [
        "Adana'da kaldirimda yururken ruhsatsiz tabancayla havaya ates actigi goruntuler guvenlik kamerasina yansıyan Hamza Y. tutuklandi.",
        "Artan girdi maliyetleriyle zor gunler geciren besicilerin borclarini odeyememesi hayvanlarin satilmasina neden oluyor.",
        "Sarkici Gullu'nun kizi Tugyan Ulkem Gulter gozaltina alindi.",
        "Midyat ilcesinde otomobilde silahla vurulmus halde bulunan otel muhasebecisi topraga verildi.",
        "Universite ogrencilerinin barinma sorunu artan kira fiyatlari nedeniyle derinlesiyor.",
    ]
    
    evaluator = ModelEvaluator()
    base_model, base_tok = evaluator.load_base_model()
    ft_model, ft_tok = evaluator.load_finetuned_model(model_save_path)
    
    results = evaluator.compare_models(base_model, base_tok, ft_model, ft_tok, test_articles)
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()
