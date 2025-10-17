# src/model_training.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerChatbotTrainer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)

            logger.info(f"Loaded model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def tokenize_function(self, examples):
        """Tokenize the conversation data"""
        tokenized = self.tokenizer(
            examples['conversation'],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None
        )
        # For causal LM, labels are same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    def prepare_dataset(self, conversations: List[str]) -> Dataset:
        """Prepare Hugging Face Dataset from conversation list"""
        dataset_dict = {'conversation': conversations}
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        return tokenized_dataset

    def train_model(self, train_dataset, val_dataset, training_args: Dict):
        """Fine-tune the model"""

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        args = TrainingArguments(
            output_dir=training_args.get('output_dir', './results'),
            overwrite_output_dir=True,
            num_train_epochs=training_args.get('num_train_epochs', 3),
            per_device_train_batch_size=training_args.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=training_args.get('per_device_eval_batch_size', 4),
            learning_rate=training_args.get('learning_rate', 5e-5),
            warmup_steps=training_args.get('warmup_steps', 500),
            weight_decay=training_args.get('weight_decay', 0.01),
            logging_dir=training_args.get('logging_dir', './logs'),
            logging_steps=training_args.get('logging_steps', 10),
            evaluation_strategy=training_args.get('evaluation_strategy', 'steps'),
            eval_steps=training_args.get('eval_steps', 50),
            save_steps=training_args.get('save_steps', 100),
            save_total_limit=training_args.get('save_total_limit', 2),
            prediction_loss_only=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )

        logger.info("Starting model training...")
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(training_args['output_dir'])
        self.tokenizer.save_pretrained(training_args['output_dir'])
        logger.info(f"Model saved to {training_args['output_dir']}")

        return trainer

    def compute_metrics(self, eval_pred):
        """Compute perplexity for causal LM"""
        eval_loss = eval_pred.metrics.get("eval_loss", 0.0)
        return {"perplexity": float(np.exp(eval_loss))}


class HyperparameterExperiment:
    def __init__(self, model_trainer: TransformerChatbotTrainer):
        self.trainer = model_trainer
        self.results = []

    def run_experiments(self, train_dataset, val_dataset, experiment_configs: List[Dict]):
        """Run multiple hyperparameter experiments"""

        for i, config in enumerate(experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment_configs)}: {config}")
            try:
                config['output_dir'] = f"./models/experiment_{i+1}"
                trainer = self.trainer.train_model(train_dataset, val_dataset, config)
                eval_results = trainer.evaluate()
                experiment_result = {
                    'experiment_id': i+1,
                    'config': config,
                    'eval_loss': eval_results['eval_loss'],
                    'perplexity': float(np.exp(eval_results['eval_loss']))
                }
                self.results.append(experiment_result)
                logger.info(f"Experiment {i+1} completed: Loss={eval_results['eval_loss']:.4f}")
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {e}")
                continue

        return self.results

    def get_best_model(self):
        """Get the best model based on evaluation loss"""
        if not self.results:
            return None
        best_result = min(self.results, key=lambda x: x['eval_loss'])
        return best_result


# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor  # Your preprocessing module

    preprocessor = DataPreprocessor()
    trainer = TransformerChatbotTrainer("microsoft/DialoGPT-small")

    df = preprocessor.load_data('../dataset/womenchatbot.csv')
    processed_df = preprocessor.preprocess_dataset(df)
    conversations = preprocessor.prepare_conversation_format(processed_df)

    if trainer.load_model_and_tokenizer():
        dataset = trainer.prepare_dataset(conversations)
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))

        experiment_configs = [
            {'num_train_epochs': 3, 'per_device_train_batch_size': 4, 'learning_rate': 5e-5, 'warmup_steps': 100},
            {'num_train_epochs': 5, 'per_device_train_batch_size': 8, 'learning_rate': 3e-5, 'warmup_steps': 200}
        ]

        experiment_runner = HyperparameterExperiment(trainer)
        results = experiment_runner.run_experiments(train_dataset, val_dataset, experiment_configs)

        print("Experiment Results:")
        for result in results:
            print(f"Experiment {result['experiment_id']}: Loss={result['eval_loss']:.4f}, Perplexity={result['perplexity']:.4f}")
