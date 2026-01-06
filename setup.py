from setuptools import setup, find_packages

setup(
    name="cyberthreat_insight",
    version="0.1.0",
    author="Atsu Vovor",
    author_email="atsu.vovor@bell.net",
    description="Cyber Threat Insight: AI-driven cybersecurity analytics using synthetic data and anomaly detection",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/atsuvovor/CyberThreat_Insight",
    packages=find_packages(exclude=("notebooks", "tests")),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "shap",
        "imbalanced-learn",
        "torch",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
