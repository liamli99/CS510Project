import numpy as np
import torch

from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

from sklearn.metrics.pairwise import cosine_similarity

# BM25
def recommand1_BM25(title, df):
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

def recommend1_word2Vec(title, df):
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
def recommend1_bert(title, df):
    # Compute BERT embeddings for all titles including the query title
    embeddings = get_bert_embeddings(df['Title'].tolist())

    index = df[df['Title'] == title].index[0]
    query_embedding = embeddings[index]

    similarities = cosine_similarity([query_embedding], embeddings)[0]

    df['Similarity'] = similarities

    recommended_df = df.sort_values(by='Similarity', ascending=False).head(5)
    recommended_recipes = [recipe for recipe in recommended_df['Title']]

    return recommended_recipes