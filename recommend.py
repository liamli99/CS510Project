import re
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel


####################################################################################################
# Recommendation Metric
####################################################################################################

def recommendation_metric(similarity, rating, alpha=0.5, beta=0.5):
    return alpha * similarity + beta * rating / 100

####################################################################################################
# Recommendation Functions
####################################################################################################

def recommend1(title, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Title'])
    
    index = df[df['Title'] == title].index[0]
    
    input_vector = tfidf_matrix[index]
    similarity = cosine_similarity(tfidf_matrix, input_vector)

    df['Similarity'] = similarity
    df['Recommendation_Metric'] = recommendation_metric(df['Similarity'], df['Rating'])
    recommended_df = df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
    recommended_recipes_titles = []
    recommended_recipes_images = []
    # recommended_recipes_ratings = []
    recommended_recipes_stats = []
    recommended_recipes_instructions =[]
    
    for recipes_title in recommended_df['Title']:
        recommended_recipes_titles.append(recipes_title)
    for recipes_image in recommended_df['Image_Name']:
        recommended_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in recommended_df['Rating']:
    #     recommended_recipes_ratings.append(recipes_rating)
    for _, row in recommended_df.iterrows():
        recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
    for recipes_instruction in recommended_df['Instructions']:
        recommended_recipes_instructions.append(recipes_instruction)
    
    return recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions

def recommend2(inputValue1, inputValue2, df):
    filtered_df = df.copy()

    def not_include(lst, str):
        return all(ele not in str for ele in lst)
    
    if inputValue1 != "":
        inputValue1_lst = re.split('[, ]+', inputValue1.lower())
        mask = filtered_df.apply(lambda row: not_include(inputValue1_lst, row['Cleaned_Ingredients']), axis=1)
        filtered_df = filtered_df[mask]
    
    corpus = filtered_df['Cleaned_Ingredients'].tolist() + [inputValue2]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    input_vector = tfidf_matrix[-1]
    similarity = cosine_similarity(tfidf_matrix[:-1], input_vector)

    filtered_df['Similarity'] = similarity
    filtered_df['Recommendation_Metric'] = recommendation_metric(filtered_df['Similarity'], filtered_df['Rating'])
    recommended_df = filtered_df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
    recommended_recipes_titles = []
    recommended_recipes_images = []
    # recommended_recipes_ratings = []
    recommended_recipes_stats = []
    recommended_recipes_instructions =[]

    for recipes_title in recommended_df['Title']:
        recommended_recipes_titles.append(recipes_title)
    for recipes_image in recommended_df['Image_Name']:
        recommended_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in recommended_df['Rating']:
    #     recommended_recipes_ratings.append(recipes_rating)
    for _, row in recommended_df.iterrows():
        recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
    for recipes_instruction in recommended_df['Instructions']:
        recommended_recipes_instructions.append(recipes_instruction)
        
    return recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions

# BM25
def recommand1_BM25_lengthNormalization(title, df):
  # print("Column names:", df.columns.tolist())


  tokenized_titles = [title.split() for title in df['Title']]
  bm25 = BM25Okapi(tokenized_titles)

  index = df[df['Title'] == title].index[0]
  query = tokenized_titles[index]

  scores = bm25.get_scores(query)

  df['Similarity'] = scores

  recommended_df = df.sort_values(by='Similarity', ascending=False).head(5)
  recommended_recipes = []

  for recipes in recommended_df['Title']:
      recommended_recipes.append(recipes)

  return recommended_recipes

def recommend_jaccard(title, df):

    tokenized_titles = [set(title.lower().split()) for title in df['Title']]
    mlb = MultiLabelBinarizer()
    title_vectors = mlb.fit_transform(tokenized_titles)


    index = df[df['Title'] == title].index[0]
    query_vector = title_vectors[index]

    similarities = [jaccard_score(query_vector, title_vector, average='binary') for title_vector in title_vectors]

    df['Similarity'] = similarities
    recommended_df = df.sort_values(by='Similarity', ascending=False).head(5)
    recommended_recipes = [recipe for recipe in recommended_df['Title']]

    return recommended_recipes


def recommend_word2Vec(title, df):


    tokenized_titles = [title.lower().split() for title in df['Title']]

    model = Word2Vec(sentences=tokenized_titles, vector_size=100, window=5, min_count=1, workers=4)

    def document_vector(doc):
        # Remove out-of-vocabulary words and compute the mean
        return np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)

    doc_vectors = np.array([document_vector(doc) for doc in tokenized_titles])


    index = df[df['Title'] == title].index[0]
    query_vector = document_vector(tokenized_titles[index])


    similarities = cosine_similarity([query_vector], doc_vectors)[0]

    df['Similarity'] = similarities

    recommended_df = df.sort_values(by='Similarity', ascending=False).head(5)
    recommended_recipes = []

    for recipes in recommended_df['Title']:
        recommended_recipes.append(recipes)

    return recommended_recipes



def get_bert_embeddings(texts, batch_size=10):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_embeddings = outputs.pooler_output.squeeze().to('cpu').numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)
def recommend_bert(title, df):
    print("Column names:", df.columns.tolist())

    # Compute BERT embeddings for all titles including the query title
    embeddings = get_bert_embeddings(df['Title'].tolist())

    index = df[df['Title'] == title].index[0]
    query_embedding = embeddings[index]

    similarities = cosine_similarity([query_embedding], embeddings)[0]

    df['Similarity'] = similarities

    recommended_df = df.sort_values(by='Similarity', ascending=False).head(5)
    recommended_recipes = [recipe for recipe in recommended_df['Title']]

    return recommended_recipes
