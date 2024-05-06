import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Ingredients'])
    
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