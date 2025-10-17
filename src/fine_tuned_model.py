# src/fine_tuned_model.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTunedEmpowermentChatbot:
    """
    A fine-tuned DialoGPT-based model designed for women's empowerment
    and emotional support conversations.
    """

    def __init__(self, model_path_or_name: str):
        """
        Initialize chatbot with local or Hugging Face model.
        """
        self.model_path_or_name = model_path_or_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Using device: {self.device}")

    def load_model(self, model_path_or_name: str = None) -> bool:
        """
        Load model and tokenizer from local path or Hugging Face.
        Returns True if successful, False otherwise.
        """
        try:
            path = model_path_or_name or self.model_path_or_name
            
            # Check if local path exists and has model files
            if os.path.exists(path):
                required_files = ['config.json', 'pytorch_model.bin'] or ['config.json', 'model.safetensors']
                if not all(os.path.exists(os.path.join(path, f)) for f in ['config.json']):
                    logger.warning(f"⚠️ Local model path exists but missing required files: {path}")
                    return False
            
            logger.info(f"🔍 Loading model from: {path}")

            # Try to load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with proper error handling
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float32,  # Use float32 for compatibility
                low_cpu_mem_usage=True
            )
            
            self.model.to(self.device)
            self.model.eval()

            logger.info("✅ Fine-tuned model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            # Clear any partially loaded components
            self.model = None
            self.tokenizer = None
            return False

    def generate_response(self, user_input: str, max_length: int = 80) -> str:
        """
        Generate a conversational response from the fine-tuned model.
        """
        if not self.model or not self.tokenizer:
            return "⚠️ Model not loaded. Please ensure the fine-tuned model is available."

        try:
            input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token, return_tensors="pt"
            ).to(self.device)

            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )

            response = self.tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            cleaned_response = response.strip().capitalize()

            # Add empathy flavor if needed
            if any(word in user_input.lower() for word in ["feel", "stress", "anxious", "sad", "confidence", "overwhelmed"]):
                cleaned_response = (
                    f"I understand how you feel. {cleaned_response} "
                    f"Remember, it's okay to take things one step at a time."
                )

            return cleaned_response

        except Exception as e:
            logger.error(f"⚠️ Response generation error: {e}")
            return "I'm here to listen. Could you please tell me more?"

    def get_response_confidence(self, user_input: str, response: str) -> float:
        """
        A lightweight similarity-based confidence scoring system.
        """
        try:
            ratio = SequenceMatcher(None, user_input.lower(), response.lower()).ratio()
            confidence = max(0.3, min(1.0, 1.0 - abs(0.5 - ratio)))  # normalized
            return round(confidence, 2)
        except Exception as e:
            logger.warning(f"⚠️ Confidence scoring error: {e}")
            return 0.5


# ✅ Standalone test mode
if __name__ == "__main__":
    print("🔧 Testing FineTunedEmpowermentChatbot...")

    model_path = "src/fine_tuned_model" if os.path.exists("src/fine_tuned_model") else "microsoft/DialoGPT-small"
    chatbot = FineTunedEmpowermentChatbot(model_path)

    if chatbot.load_model():
        test_inputs = [
            "I'm feeling anxious about my job.",
            "How can I build self-confidence?",
            "Tell me about women empowerment."
        ]

        for user_text in test_inputs:
            response = chatbot.generate_response(user_text)
            conf = chatbot.get_response_confidence(user_text, response)
            print(f"\n🧠 User: {user_text}")
            print(f"🤖 Bot: {response} [conf: {conf}]")
    else:
        print("⚠️ Model failed to load.")
