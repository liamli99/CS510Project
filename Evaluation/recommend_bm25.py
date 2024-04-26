import re
from rank_bm25 import BM25Okapi

def recommendation_metric(similarity, rating, alpha=0.5, beta=0.5):
    return alpha * similarity + beta * rating / 100

# https://pypi.org/project/rank-bm25/

def recommend1_bm25(title, df):
    tokenized_titles = [title.split() for title in df['Title']]
    bm25 = BM25Okapi(tokenized_titles)
    
    index = df[df['Title'] == title].index[0]
    
    tokenized_query = tokenized_titles[index]
    scores = bm25.get_scores(tokenized_query)

    df['Similarity'] = scores
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

def recommend2_bm25(inputValue1, inputValue2, df):
    filtered_df = df.copy()

    def not_include(lst, str):
        return all(ele not in str for ele in lst)
    
    if inputValue1 != "":
        inputValue1_lst = re.split('[, ]+', inputValue1.lower())
        mask = filtered_df.apply(lambda row: not_include(inputValue1_lst, row['Cleaned_Ingredients']), axis=1)
        filtered_df = filtered_df[mask]

    tokenized_ingredients = [ingredient.split() for ingredient in filtered_df['Cleaned_Ingredients']]
    bm25 = BM25Okapi(tokenized_ingredients)
    
    query = inputValue2
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    filtered_df['Similarity'] = scores
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