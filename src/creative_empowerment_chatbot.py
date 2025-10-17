# src/creative_empowerment_chatbot.py
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreativeEmpowermentChatbot:
    """
    A creative, TF-IDF–based chatbot designed to support women's empowerment
    and personal well-being with memory, empathy, and contextual awareness.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize chatbot with Q&A dataframe.
        Expected columns: ['question', 'answer', 'category']
        """
        self.dataframe = dataframe.fillna("")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.questions = self.dataframe['question'].tolist()
        self.answers = self.dataframe['answer'].tolist()
        self.categories = self.dataframe.get('category', ["general"] * len(self.dataframe))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

        # Memory for conversational context
        self.conversation_history = []
        self.user_info = {}

        logger.info("✅ CreativeEmpowermentChatbot initialized successfully.")

    # -------------------------------------------------------------------------
    # Memory and Context Handling
    # -------------------------------------------------------------------------
    def _rebuild_internal_history(self, external_history: Optional[List[Dict]] = None):
        """Rebuild internal conversation memory from provided history (if any)."""
        if external_history:
            self.conversation_history = [
                {"user": msg["content"], "bot": ""} if msg["role"] == "user" else
                {"user": "", "bot": msg["content"]} for msg in external_history
            ]

    def _update_conversation_memory(self, user_input: str, bot_response: str):
        """Save a new user-bot exchange to memory."""
        self.conversation_history.append({"user": user_input, "bot": bot_response})
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    # -------------------------------------------------------------------------
    # Empathy and Personalization
    # -------------------------------------------------------------------------
    def _extract_user_info(self, text: str):
        """Detect age or location clues from text to personalize responses."""
        if any(word in text.lower() for word in ['student', 'school', 'college']):
            self.user_info['identity'] = 'student'
        elif any(word in text.lower() for word in ['mother', 'child', 'kids']):
            self.user_info['identity'] = 'mother'

    def _get_creative_greeting(self):
        greetings = [
            "Hello 🌸 How are you feeling today?",
            "Hi there! 💕 What’s on your mind?",
            "Hey, lovely soul 🌿 Ready to talk about empowerment?",
            "Welcome back 🌼 I’m so glad you’re here."
        ]
        return random.choice(greetings)

    def _get_context_aware_response(self, user_input: str):
        """Provide continuity if user follows up on previous chat."""
        if not self.conversation_history:
            return None
        last_exchange = self.conversation_history[-1]
        
        # More specific context words that indicate follow-up questions
        context_indicators = [
            'same', 'that', 'it', 'what about', 'more about', 'tell me more', 
            'expand on', 'elaborate', 'continue', 'go on', 'and then'
        ]
        
        # Check for explicit follow-up indicators
        user_lower = user_input.lower().strip()
        if any(indicator in user_lower for indicator in context_indicators):
            # But don't trigger on new questions starting with "what is", "how do", etc.
            question_starters = ['what is', 'what are', 'how do', 'how can', 'why is', 'when do', 'where is']
            if not any(starter in user_lower for starter in question_starters):
                return (
                    f"I remember we talked about '{last_exchange['user'][:40]}...'. "
                    "Would you like me to expand on that?"
                )
        return None

    # -------------------------------------------------------------------------
    # Core Response Logic
    # -------------------------------------------------------------------------
    def _get_creative_fallback(self, user_input: str):
        """Creative fallback responses with empathy and memory."""
        # Check if this is a new question that we don't have information about
        question_indicators = ['what is', 'what are', 'how do', 'how can', 'why is', 'when do', 'where is', 'who is']
        is_question = any(indicator in user_input.lower() for indicator in question_indicators)
        
        if is_question:
            # For questions we can't answer, be direct but helpful
            fallbacks = [
                "I'm focused on women's empowerment, education, workplace equality, and personal growth topics. Could you ask me something about those areas? 🌸",
                "That's an interesting question! I specialize in women's empowerment, career advice, health rights, and personal development. What would you like to know about those topics? 💫",
                "I'd love to help! I'm designed to support women with topics like empowerment, education, workplace equality, and personal growth. What specific area interests you? 🌿",
                "Great question! While I may not have specific information about that topic, I'm here to help with women's empowerment, career guidance, health rights, and personal development. What would you like to explore? 💕"
            ]
        else:
            # For general statements/conversations
            if self.conversation_history:
                last_topic = self.conversation_history[-1]['user']
                last_topic_snippet = last_topic[:50] + ('...' if len(last_topic) > 50 else '')
                return (
                    f"I'd love to continue our conversation about your journey! "
                    f"You mentioned '{last_topic_snippet}' — would you like to explore that further, "
                    f"or is there something new on your heart? 💖"
                )

            fallbacks = [
                "I'm here for you 🌸 Could you tell me a bit more about what's on your mind?",
                "That's really interesting. Could you share a little more so I can better support you?",
                "I want to understand you fully 💫 — can you tell me a bit more about what you mean?",
                "I'm listening carefully 🌿 — could you describe that a little more?",
                "Your voice matters 💕 — help me understand what's happening by sharing a bit more."
            ]
        return random.choice(fallbacks)

    # -------------------------------------------------------------------------
    # Main Response Function
    # -------------------------------------------------------------------------
    def get_response(self, user_input: str, external_history: Optional[List[Dict]] = None) -> str:
        """
        Generate an empathetic, empowerment-focused response using:
        - TF-IDF similarity for factual/known queries
        - Contextual, creative fallbacks otherwise
        """
        if not user_input or not isinstance(user_input, str):
            return "Could you please repeat that? 💬"

        # Step 1: Sync history if provided
        if external_history is not None:
            self._rebuild_internal_history(external_history)

        # Step 2: Personalization and context awareness
        self._extract_user_info(user_input)
        context_response = self._get_context_aware_response(user_input)
        if context_response:
            self._update_conversation_memory(user_input, context_response)
            return context_response

        # Step 3: Greetings and check-ins
        if any(word in user_input.lower() for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
            response = self._get_creative_greeting()
            self._update_conversation_memory(user_input, response)
            return response

        # Step 4: TF-IDF similarity match
        user_vec = self.vectorizer.transform([user_input])
        cosine_similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        best_match_index = np.argmax(cosine_similarities)
        best_match_score = cosine_similarities[best_match_index]

        if best_match_score > 0.35:
            response = self.answers[best_match_index]
        else:
            response = self._get_creative_fallback(user_input)

        # Step 5: Store in memory and return
        self._update_conversation_memory(user_input, response)
        return response


# -------------------------------------------------------------------------
# Local Testing
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Sample mini dataset for test run
    data = {
        'question': [
            'What can I do when I feel anxious?',
            'How can I relieve period cramps?',
            'How do I build confidence at work?',
            'I’m having trouble sleeping lately.'
        ],
        'answer': [
            'Try deep breathing or grounding techniques when anxiety hits. 🌿',
            'A heating pad or warm bath can help ease period pain. 💜',
            'Start small — speak up once per meeting and celebrate it. 💪',
            'Avoid screens before bed and practice slow breathing. 😴'
        ],
        'category': [
            'mental_health', 'period', 'confidence', 'sleep'
        ]
    }

    df = pd.DataFrame(data)
    bot = CreativeEmpowermentChatbot(df)

    print("Chatbot ready! Type 'exit' to stop.")
    while True:
        user_in = input("You: ")
        if user_in.lower() == 'exit':
            print("Chatbot: Take care, beautiful soul 💫")
            break
        response = bot.get_response(user_in)
        print("Chatbot:", response)
