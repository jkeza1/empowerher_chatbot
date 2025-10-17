# test_complete.py (in project root)
import sys
import os

# Add src to path - correct for project root location
sys.path.append('src')

def test_complete_system():
    print("🧪 Testing Complete Modular System...")
    
    try:
        # Test data preprocessing
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data('dataset/womenchatbot.csv')  # Path from root
        processed_df = preprocessor.preprocess_dataset(df)
        print("✅ Data preprocessing working")
        
        # Test TF-IDF chatbot
        from creative_empowerment_chatbot import CreativeEmpowermentChatbot
        tfidf_bot = CreativeEmpowermentChatbot(processed_df)
        response = tfidf_bot.get_response("Hello")
        print("✅ TF-IDF chatbot working")
        
        # Test hybrid system
        from chatbot import HybridEmpowermentChatbot
        hybrid_bot = HybridEmpowermentChatbot(processed_df)
        
        test_questions = [
            "What is women's empowerment?",
            "I'm feeling stressed",
            "How to build confidence?"
        ]
        
        for question in test_questions:
            result = hybrid_bot.get_response(question)
            print(f"❓ {question}")
            print(f"   🤖 Source: {result['source']}")
            print(f"   💬 Response: {result['response'][:80]}...")
            print()
        
        print("🎉 Complete system integration successful!")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")

if __name__ == "__main__":
    test_complete_system()