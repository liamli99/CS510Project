import re
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def recommendation_metric(similarity, rating, alpha=0.5, beta=0.5):
    return alpha * similarity + beta * rating / 100

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

def train_word2vec(tokenized_texts):
    # Tune the parameters?
    model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=5, min_count=1, workers=4)
    model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=10)
    return model

def get_document_vectors(tokenized_texts, model):
    # Obtain document-level embeddings by averaging word embeddings in the document
    document_vectors = []
    for tokens in tokenized_texts:
        vector = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
        if not np.isnan(np.sum(vector)):  # Check if the mean resulted in a NaN vector
            document_vectors.append(vector)
        else:
            document_vectors.append(np.zeros(model.vector_size))  # Use a zero vector if no words matched
    return np.array(document_vectors)

def recommend1_word2vec(title, df):
    tokenized_titles = [title.split() for title in df['Title']]
    model = train_word2vec(tokenized_titles)
    document_vectors = get_document_vectors(tokenized_titles, model)

    index = df[df['Title'] == title].index[0]
    
    query_vector = document_vectors[index]
    similarity = cosine_similarity(document_vectors, [query_vector])

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
        recommended_recipes_images.append('../archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in recommended_df['Rating']:
    #     recommended_recipes_ratings.append(recipes_rating)
    for _, row in recommended_df.iterrows():
        recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
    for recipes_instruction in recommended_df['Instructions']:
        recommended_recipes_instructions.append(recipes_instruction)
    
    return recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions

def recommend2_word2vec(inputValue1, inputValue2, df):
    filtered_df = df.copy()

    def not_include(lst, str):
        return all(ele not in str for ele in lst)
    
    if inputValue1 != "":
        inputValue1_lst = re.split('[, ]+', inputValue1.lower())
        mask = filtered_df.apply(lambda row: not_include(inputValue1_lst, row['Cleaned_Ingredients']), axis=1)
        filtered_df = filtered_df[mask]

    tokenized_ingredients = [ingredient.split() for ingredient in filtered_df['Cleaned_Ingredients']]
    model = train_word2vec(tokenized_ingredients)
    document_vectors = get_document_vectors(tokenized_ingredients, model)
    
    query_vector = np.mean([model.wv[word] for word in inputValue2.split() if word in model.wv], axis=0)
    similarity = cosine_similarity(document_vectors, [query_vector])

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
        recommended_recipes_images.append('../archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in recommended_df['Rating']:
    #     recommended_recipes_ratings.append(recipes_rating)
    for _, row in recommended_df.iterrows():
        recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
    for recipes_instruction in recommended_df['Instructions']:
        recommended_recipes_instructions.append(recipes_instruction)
        
    return recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions