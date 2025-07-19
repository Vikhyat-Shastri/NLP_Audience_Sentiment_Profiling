from setuptools import setup, find_packages

setup(
    name='NLP_Audience_Sentiment_Profiling',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A sentiment analysis project using NLP techniques on IMDB reviews.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/NLP_Audience_Sentiment_Profiling',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'nltk',
        'contractions',
        'wordcloud',
        'matplotlib',
        'seaborn',
        'joblib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)