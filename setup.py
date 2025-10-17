from setuptools import setup

setup(
    name="empowerher-chatbot",
    version="1.0.0",
    description="Women's empowerment chatbot",
    python_requires=">=3.8",
    install_requires=[
        "streamlit==1.28.0",
        "pandas==1.5.3",
        "scikit-learn==1.3.2", 
        "numpy==1.24.3",
        "scipy==1.10.1",
        "threadpoolctl==3.1.0",
        "joblib==1.2.0",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "plotly==5.15.0",
        "wordcloud==1.9.2"
    ],
)