# run.py - Updated with model integration
import argparse
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import TransformerChatbotTrainer, HyperparameterExperiment
from evaluation import ChatbotEvaluator
from fine_tuned_model import FineTunedEmpowermentChatbot
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Run the complete chatbot training and evaluation pipeline"""
    logger.info("Starting Women's Empowerment Chatbot Pipeline...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Data Preprocessing
    logger.info("Step 1: Data Preprocessing")
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data('dataset/womenchatbot.csv')
    processed_df = preprocessor.preprocess_dataset(df)
    conversations = preprocessor.prepare_conversation_format(processed_df)
    
    # Save processed data
    processed_df.to_csv(f'{results_dir}/processed_data.csv', index=False)
    
    # Split dataset
    train_df, val_df, test_df = preprocessor.split_dataset(processed_df)
    
    # Step 2: Model Training
    logger.info("Step 2: Model Training")
    trainer = TransformerChatbotTrainer("microsoft/DialoGPT-small")
    
    if trainer.load_model_and_tokenizer():
        # Prepare datasets
        train_conversations = preprocessor.prepare_conversation_format(train_df)
        val_conversations = preprocessor.prepare_conversation_format(val_df)
        
        train_dataset = trainer.prepare_dataset(train_conversations)
        val_dataset = trainer.prepare_dataset(val_conversations)
        
        # Define hyperparameter experiments
        experiment_configs = [
            {
                'output_dir': f'./models/experiment_1_{timestamp}',
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'learning_rate': 5e-5,
                'warmup_steps': 100,
                'weight_decay': 0.01,
                'logging_steps': 10,
                'eval_steps': 50,
            },
            {
                'output_dir': f'./models/experiment_2_{timestamp}',
                'num_train_epochs': 5,
                'per_device_train_batch_size': 8,
                'learning_rate': 3e-5,
                'warmup_steps': 200,
                'weight_decay': 0.02,
                'logging_steps': 10,
                'eval_steps': 50,
            },
            {
                'output_dir': f'./models/experiment_3_{timestamp}',
                'num_train_epochs': 4,
                'per_device_train_batch_size': 16,
                'learning_rate': 1e-4,
                'warmup_steps': 150,
                'weight_decay': 0.015,
                'logging_steps': 10,
                'eval_steps': 50,
            }
        ]
        
        # Run experiments
        experiment_runner = HyperparameterExperiment(trainer)
        results = experiment_runner.run_experiments(train_dataset, val_dataset, experiment_configs)
        
        # Save experiment results
        with open(f'{results_dir}/experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display experiment results
        print("\n" + "="*60)
        print("HYPERPARAMETER EXPERIMENT RESULTS")
        print("="*60)
        
        experiment_table = []
        for result in results:
            exp_data = {
                'Experiment': result['experiment_id'],
                'Epochs': result['config']['num_train_epochs'],
                'Batch Size': result['config']['per_device_train_batch_size'],
                'Learning Rate': result['config']['learning_rate'],
                'Loss': f"{result['eval_loss']:.4f}",
                'Perplexity': f"{result['perplexity']:.4f}"
            }
            experiment_table.append(exp_data)
            print(f"\nExperiment {result['experiment_id']}:")
            print(f"  Config: {result['config']}")
            print(f"  Evaluation Loss: {result['eval_loss']:.4f}")
            print(f"  Perplexity: {result['perplexity']:.4f}")
        
        # Save experiment table
        pd.DataFrame(experiment_table).to_csv(f'{results_dir}/experiment_table.csv', index=False)
        
        # Get best model
        best_model = experiment_runner.get_best_model()
        if best_model:
            best_model_path = best_model['config']['output_dir']
            print(f"\n🎯 Best Model: Experiment {best_model['experiment_id']}")
            print(f"📁 Model saved at: {best_model_path}")
            
            # Create symbolic link to best model
            best_model_link = "./models/best_model"
            if os.path.exists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_model_path, best_model_link)
            
            # Step 3: Load and Test Fine-Tuned Model
            logger.info("Step 3: Testing Fine-Tuned Model")
            fine_tuned_chatbot = FineTunedEmpowermentChatbot()
            
            if fine_tuned_chatbot.load_model(best_model_path):
                # Test the fine-tuned model
                test_questions = [
                    "What is women's empowerment?",
                    "How can women negotiate better salaries?",
                    "What are my legal rights as a woman?",
                    "I'm feeling overwhelmed with work-life balance"
                ]
                
                print("\n" + "="*60)
                print("FINE-TUNED MODEL TESTING")
                print("="*60)
                
                test_results = []
                for question in test_questions:
                    response = fine_tuned_chatbot.generate_response(question)
                    confidence = fine_tuned_chatbot.get_response_confidence(question, response)
                    topic = fine_tuned_chatbot._detect_topic(question)
                    
                    test_result = {
                        'question': question,
                        'response': response,
                        'confidence': confidence,
                        'topic': topic
                    }
                    test_results.append(test_result)
                    
                    print(f"\nQ: {question}")
                    print(f"A: {response}")
                    print(f"Confidence: {confidence:.2f} | Topic: {topic}")
                    print("-" * 50)
                
                # Save test results
                pd.DataFrame(test_results).to_csv(f'{results_dir}/model_test_results.csv', index=False)
            
            # Step 4: Comprehensive Evaluation
            logger.info("Step 4: Comprehensive Evaluation")
            evaluator = ChatbotEvaluator(best_model_path)
            
            # Prepare test data
            test_data = []
            for _, row in test_df.iterrows():
                test_data.append((row['question_clean'], row['answer_clean']))
            
            # Run evaluation on subset for speed
            metrics = evaluator.evaluate_on_test_set(test_data[:20])
            
            print("\n" + "="*60)
            print("COMPREHENSIVE PERFORMANCE METRICS")
            print("="*60)
            
            metrics_table = []
            for metric, value in metrics.items():
                metrics_table.append({'Metric': metric, 'Value': f"{value:.4f}"})
                print(f"{metric:20}: {value:.4f}")
            
            # Save metrics
            pd.DataFrame(metrics_table).to_csv(f'{results_dir}/performance_metrics.csv', index=False)
            
            # Qualitative analysis
            qualitative_results = evaluator.qualitative_analysis(test_data[:5])
            
            print("\n" + "="*60)
            print("QUALITATIVE ANALYSIS")
            print("="*60)
            
            qualitative_data = []
            for i, result in enumerate(qualitative_results):
                qualitative_data.append({
                    'example_id': i+1,
                    'question': result['question'],
                    'expected': result['expected'],
                    'generated': result['generated'],
                    'bleu_score': result['bleu_score'],
                    'rougeL_score': result['rougeL_score']
                })
                
                print(f"\nExample {i+1}:")
                print(f"Question: {result['question']}")
                print(f"Expected: {result['expected'][:100]}...")
                print(f"Generated: {result['generated']}")
                print(f"BLEU: {result['bleu_score']:.4f} | ROUGE-L: {result['rougeL_score']:.4f}")
            
            # Save qualitative results
            pd.DataFrame(qualitative_data).to_csv(f'{results_dir}/qualitative_analysis.csv', index=False)
            
            # Final summary
            print("\n" + "="*60)
            print("🚀 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📊 Results saved to: {results_dir}")
            print(f"🤖 Best model: {best_model_path}")
            print(f"🔗 Quick access: ./models/best_model")
            print(f"📈 Key metrics:")
            print(f"   - BLEU Score: {metrics.get('bleu_mean', 0):.4f}")
            print(f"   - ROUGE-L: {metrics.get('rougeL_mean', 0):.4f}")
            print(f"   - Success Rate: {metrics.get('success_rate', 0):.4f}")
        
        else:
            logger.error("❌ No successful experiments completed")
    
    else:
        logger.error("❌ Failed to load base model and tokenizer")

def demo_fine_tuned_model(model_path: str = "./models/best_model"):
    """Demo the fine-tuned model with interactive chat"""
    print("🤖 Women's Empowerment Chatbot Demo")
    print("Type 'quit' to exit\n")
    
    chatbot = FineTunedEmpowermentChatbot()
    
    if chatbot.load_model(model_path):
        print("✅ Model loaded successfully!")
        print("💬 You can now chat with the fine-tuned model:\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if user_input:
                response = chatbot.generate_response(user_input)
                confidence = chatbot.get_response_confidence(user_input, response)
                topic = chatbot._detect_topic(user_input)
                
                print(f"Bot: {response}")
                print(f"   [Confidence: {confidence:.2f} | Topic: {topic}]\n")
    else:
        print("❌ Failed to load the model. Please train the model first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Women\'s Empowerment Chatbot Pipeline')
    parser.add_argument('--mode', choices=['full', 'train', 'eval', 'demo'], default='full',
                       help='Pipeline mode: full, train, eval, or demo')
    parser.add_argument('--model-path', type=str, default="./models/best_model",
                       help='Path to model for evaluation/demo mode')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'train':
        # Simplified training mode
        print("Training mode - use 'full' for complete pipeline")
    elif args.mode == 'eval':
        from evaluation import ChatbotEvaluator
        run_evaluation_only(args.model_path)
    elif args.mode == 'demo':
        demo_fine_tuned_model(args.model_path)