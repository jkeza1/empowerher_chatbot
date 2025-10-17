# scripts/train_pipeline.py
import sys
import os
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from model_training import TransformerChatbotTrainer, HyperparameterExperiment
from evaluation import ChatbotEvaluator
import pandas as pd

def main():
    print("🚀 Starting AI Training Pipeline...")
    
    # 1. Load and preprocess data
    print("📊 Step 1: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('dataset/womenchatbot.csv')
    processed_df = preprocessor.preprocess_dataset(df)
    conversations = preprocessor.prepare_conversation_format(processed_df)
    
    # 2. Train model with hyperparameter tuning
    print("🤖 Step 2: Training model with hyperparameter search...")
    trainer = TransformerChatbotTrainer("microsoft/DialoGPT-small")
    
    if trainer.load_model_and_tokenizer():
        dataset = trainer.prepare_dataset(conversations)
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        
        # Hyperparameter experiments
        experiment_configs = [
            {
                'output_dir': './models/experiment_1',
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'learning_rate': 5e-5,
                'warmup_steps': 100,
            },
            {
                'output_dir': './models/experiment_2', 
                'num_train_epochs': 5,
                'per_device_train_batch_size': 8,
                'learning_rate': 3e-5,
                'warmup_steps': 200,
            }
        ]
        
        experiment_runner = HyperparameterExperiment(trainer)
        results = experiment_runner.run_experiments(train_dataset, val_dataset, experiment_configs)
        
        # 3. Select best model
        best_result = experiment_runner.get_best_model()
        best_model_path = best_result['config']['output_dir']
        print(f"🏆 Best model: {best_model_path} (Loss: {best_result['eval_loss']:.4f})")
        
        # 4. Evaluate best model
        print("📈 Step 3: Evaluating best model...")
        evaluator = ChatbotEvaluator(best_model_path)
        
        # Prepare test data
        _, _, test_df = preprocessor.split_dataset(processed_df)
        test_data = [(row['question_clean'], row['answer_clean']) 
                    for _, row in test_df.iterrows()]
        
        metrics = evaluator.evaluate_on_test_set(test_data)
        
        print("📊 Final Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 5. Deploy best model
        print("🚀 Step 4: Deploying best model...")
        os.system(f"cp -r {best_model_path} ./models/best_model/")
        print("✅ Training pipeline completed! Best model ready in ./models/best_model/")
    
if __name__ == "__main__":
    main()