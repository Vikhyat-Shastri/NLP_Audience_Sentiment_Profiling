def extract_tfidf_features(df, text_column='cleaned_review', max_features=5000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    return tfidf_matrix, vectorizer

def extract_additional_features(df):
    df['word_count'] = df['cleaned_review'].apply(lambda x: len(x.split()))
    df['char_count'] = df['cleaned_review'].apply(lambda x: len(x))
    df['avg_word_length'] = df['char_count'] / df['word_count']
    
    return df[['word_count', 'char_count', 'avg_word_length']]