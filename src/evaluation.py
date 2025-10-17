# src/evaluation.py
import torch
from transformers import pipeline
import evaluate
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotEvaluator:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metrics
        self.bleu = evaluate.load('bleu')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore = evaluate.load('bertscore')
        
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            self.chatbot = pipeline(
                "text-generation",
                model=self.model_path,
                tokenizer=self.model_path,
                device=0 if torch.cuda.is_available() else -1
            )
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score between generated and reference text"""
        try:
            # Tokenize
            gen_tokens = generated.split()
            ref_tokens = reference.split()
            
            # Calculate BLEU score
            smoothie = SmoothingFunction().method4
            score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
            return score
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bertscore(self, generated: str, reference: str) -> float:
        """Calculate BERTScore"""
        try:
            results = self.bertscore.compute(
                predictions=[generated],
                references=[reference],
                lang="en"
            )
            return results['f1'][0]
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return 0.0
    
    def evaluate_response_quality(self, generated: str, reference: str) -> Dict:
        """Comprehensive evaluation of a single response"""
        metrics = {}
        
        # BLEU Score
        metrics['bleu'] = self.calculate_bleu_score(generated, reference)
        
        # ROUGE Scores
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        metrics.update(rouge_scores)
        
        # BERTScore
        metrics['bertscore'] = self.calculate_bertscore(generated, reference)
        
        # Length ratio (simple coherence measure)
        metrics['length_ratio'] = len(generated.split()) / max(len(reference.split()), 1)
        
        return metrics
    
    def evaluate_on_test_set(self, test_data: List[Tuple[str, str]]) -> Dict:
        """Evaluate model on test dataset"""
        if not hasattr(self, 'chatbot'):
            if not self.load_model():
                raise ValueError("Model could not be loaded")
        
        all_metrics = []
        
        for question, expected_answer in test_data:
            try:
                # Generate response
                generated_response = self.generate_response(question)
                
                # Calculate metrics
                metrics = self.evaluate_response_quality(generated_response, expected_answer)
                metrics['question'] = question
                metrics['generated'] = generated_response
                metrics['expected'] = expected_answer
                
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error evaluating question: {question}, Error: {e}")
                continue
        
        # Aggregate results
        if not all_metrics:
            return {}
        
        df_results = pd.DataFrame(all_metrics)
        aggregated_metrics = {
            'bleu_mean': df_results['bleu'].mean(),
            'bleu_std': df_results['bleu'].std(),
            'rouge1_mean': df_results['rouge1'].mean(),
            'rouge2_mean': df_results['rouge2'].mean(),
            'rougeL_mean': df_results['rougeL'].mean(),
            'bertscore_mean': df_results['bertscore'].mean(),
            'success_rate': len(all_metrics) / len(test_data),
            'total_tested': len(all_metrics)
        }
        
        return aggregated_metrics
    
    def generate_response(self, question: str, max_length: int = 150) -> str:
        """Generate response using the fine-tuned model"""
        try:
            # Format input for the model
            input_text = f"<user> {question} <sep> <bot>"
            
            # Generate response
            response = self.chatbot(
                input_text,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.chatbot.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract only the bot's response
            if '<bot>' in response:
                bot_response = response.split('<bot>')[-1].strip()
                # Remove any additional tokens
                bot_response = bot_response.split('<end>')[0].split('<sep>')[0].strip()
                return bot_response
            else:
                return response.strip()
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response at the moment."
    
    def qualitative_analysis(self, test_data: List[Tuple[str, str]], num_examples: int = 5):
        """Perform qualitative analysis on sample responses"""
        examples = []
        
        for i, (question, expected) in enumerate(test_data[:num_examples]):
            generated = self.generate_response(question)
            metrics = self.evaluate_response_quality(generated, expected)
            
            example = {
                'question': question,
                'expected': expected,
                'generated': generated,
                'bleu_score': metrics['bleu'],
                'rougeL_score': metrics['rougeL']
            }
            examples.append(example)
        
        return examples

# Example usage
if __name__ == "__main__":
    # Example evaluation
    evaluator = ChatbotEvaluator("./models/fine_tuned_model")
    
    # Sample test data (in practice, load from your test set)
    test_data = [
        ("What is women's empowerment?", "Women's empowerment means..."),
        ("How can women negotiate salaries?", "Women can negotiate salaries by..."),
    ]
    
    # Quantitative evaluation
    metrics = evaluator.evaluate_on_test_set(test_data)
    print("Quantitative Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Qualitative analysis
    qualitative_results = evaluator.qualitative_analysis(test_data)
    print("\nQualitative Analysis:")
    for i, result in enumerate(qualitative_results):
        print(f"\nExample {i+1}:")
        print(f"Question: {result['question']}")
        print(f"Expected: {result['expected']}")
        print(f"Generated: {result['generated']}")
        print(f"BLEU: {result['bleu_score']:.4f}, ROUGE-L: {result['rougeL_score']:.4f}")