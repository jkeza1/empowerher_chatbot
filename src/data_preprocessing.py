# src/data_preprocessing.py
import pandas as pd
import numpy as np
import re
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.special_tokens = {
            'user_token': '<user>',
            'bot_token': '<bot>',
            'sep_token': '<sep>',
            'end_token': '<end>'
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\?\!,]', '', text)
        
        return text
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Handle missing values
        initial_count = len(processed_df)
        processed_df = processed_df.dropna(subset=['question', 'answer'])
        logger.info(f"Removed {initial_count - len(processed_df)} rows with missing values")
        
        # Clean text columns
        processed_df['question_clean'] = processed_df['question'].apply(self.clean_text)
        processed_df['answer_clean'] = processed_df['answer'].apply(self.clean_text)
        
        # Remove empty strings after cleaning
        processed_df = processed_df[
            (processed_df['question_clean'].str.len() > 0) & 
            (processed_df['answer_clean'].str.len() > 0)
        ]
        
        # Add text length features
        processed_df['question_length'] = processed_df['question_clean'].str.len()
        processed_df['answer_length'] = processed_df['answer_clean'].str.len()
        
        # Filter out extremely short/long texts
        processed_df = processed_df[
            (processed_df['question_length'] >= 5) & 
            (processed_df['question_length'] <= 500) &
            (processed_df['answer_length'] >= 10) & 
            (processed_df['answer_length'] <= 1000)
        ]
        
        logger.info(f"Final dataset size: {len(processed_df)} rows")
        
        return processed_df
    
    def prepare_conversation_format(self, df: pd.DataFrame) -> List[str]:
        """Prepare data in conversation format for transformer training"""
        conversations = []
        
        for _, row in df.iterrows():
            conversation = f"{self.special_tokens['user_token']} {row['question_clean']} " \
                          f"{self.special_tokens['sep_token']} " \
                          f"{self.special_tokens['bot_token']} {row['answer_clean']} " \
                          f"{self.special_tokens['end_token']}"
            conversations.append(conversation)
        
        logger.info(f"Prepared {len(conversations)} conversation pairs")
        return conversations
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        # Stratified split by category to maintain distribution
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio, 
            stratify=df['category'],
            random_state=42
        )
        
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            stratify=temp_df['category'],
            random_state=42
        )
        
        logger.info(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('../dataset/womenchatbot.csv')
    
    # Preprocess data
    processed_df = preprocessor.preprocess_dataset(df)
    
    # Prepare conversation format
    conversations = preprocessor.prepare_conversation_format(processed_df)
    
    # Split dataset
    train_df, val_df, test_df = preprocessor.split_dataset(processed_df)
    
    print("Preprocessing completed successfully!")