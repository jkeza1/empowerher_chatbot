# src/chatbot.py
import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridEmpowermentChatbot:
    def __init__(self, tfidf_dataframe, fine_tuned_model_path=None):
        from creative_empowerment_chatbot import CreativeEmpowermentChatbot

        # Initialize TF-IDF-based chatbot
        self.tfidf_chatbot = CreativeEmpowermentChatbot(tfidf_dataframe)

        # Initialize fine-tuned model if available
        self.fine_tuned_chatbot = None
        try:
            from fine_tuned_model import FineTunedEmpowermentChatbot

            # Check if fine_tuned_model_path exists and has model files
            if fine_tuned_model_path and os.path.exists(fine_tuned_model_path):
                # Check for required model files
                has_config = os.path.exists(os.path.join(fine_tuned_model_path, 'config.json'))
                has_model = (os.path.exists(os.path.join(fine_tuned_model_path, 'pytorch_model.bin')) or 
                           os.path.exists(os.path.join(fine_tuned_model_path, 'model.safetensors')))
                
                if has_config and has_model:
                    model_source = fine_tuned_model_path
                    logger.info(f"Found local fine-tuned model at: {model_source}")
                else:
                    logger.info("Local model path exists but missing required files, using base model")
                    model_source = "microsoft/DialoGPT-small"
            else:
                logger.info("No local fine-tuned model found, using base DialoGPT model")
                model_source = "microsoft/DialoGPT-small"

            self.fine_tuned_chatbot = FineTunedEmpowermentChatbot(model_source)

            if self.fine_tuned_chatbot.load_model():
                print(f"✅ Model loaded successfully from: {model_source}")
            else:
                print("⚠️ Model failed to load, using TF-IDF only")
                self.fine_tuned_chatbot = None

        except Exception as e:
            print(f"⚠️ Fine-tuned model initialization error: {e}")
            self.fine_tuned_chatbot = None

        self.conversation_history = []

    def get_response(self, user_input: str, history: list = None) -> Dict[str, Any]:
        """
        Route queries intelligently between fine-tuned and TF-IDF models.
        Returns a dictionary with 'response', 'source', and 'confidence'.
        """
        # Forward external Streamlit history to TF-IDF chatbot
        tfidf_kwargs = {}
        if history is not None:
            tfidf_kwargs['external_history'] = history  # matches CreativeEmpowermentChatbot param

        # Use fine-tuned model for complex/emotional queries
        if self.fine_tuned_chatbot and self._should_use_fine_tuned(user_input):
            try:
                response = self.fine_tuned_chatbot.generate_response(user_input)
                confidence = self.fine_tuned_chatbot.get_response_confidence(user_input, response)

                if confidence > 0.6:
                    return {
                        'response': response,
                        'source': 'fine_tuned',
                        'confidence': confidence
                    }
            except Exception as e:
                print(f"⚠️ Fine-tuned model response error: {e}")

        # Fallback to TF-IDF chatbot (history-aware)
        tfidf_response = self.tfidf_chatbot.get_response(user_input, **tfidf_kwargs)
        return {
            'response': tfidf_response,  # use string directly
            'source': 'tfidf',
            'confidence': 0.8
        }

    def _should_use_fine_tuned(self, user_input: str) -> bool:
        """Determine if the query should use the fine-tuned model."""
        complex_keywords = [
            'feel', 'emotion', 'anxious', 'stress', 'overwhelmed',
            'discrimination', 'rights', 'empowerment', 'confidence',
            'mental health', 'self-esteem', 'imposter', 'depression', 'fear'
        ]
        return any(keyword in user_input.lower() for keyword in complex_keywords)


# ------------------------------
# Safe testing entry point
# ------------------------------
if __name__ == "__main__":
    import pandas as pd

    print("🔧 Testing HybridEmpowermentChatbot initialization...")

    # Dummy TF-IDF dataframe
    dummy_data = pd.DataFrame({
        'question': ['What is empowerment?'],
        'answer': ['Empowerment means self-confidence.']
    })

    # Initialize chatbot
    chatbot = HybridEmpowermentChatbot(dummy_data, fine_tuned_model_path="src/fine_tuned_model")

    # Test responses
    user_inputs = [
        "I'm feeling anxious about my job.",
        "What are women’s rights in the workplace?",
        "How can I improve my leadership skills?"
    ]

    for text in user_inputs:
        dummy_history = [{'role': 'user', 'content': text}]
        result = chatbot.get_response(text, history=dummy_history)
        print(f"\n🧠 User: {text}")
        print(f"🤖 Bot ({result['source']}): {result['response']} [conf: {result['confidence']:.2f}]")
