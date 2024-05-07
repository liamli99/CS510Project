import re
from transformers import BertTokenizer, BertModel
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import torch

data = pickle.load(open("data.pkl", 'rb'))

def recommendation_metric(similarity, rating, alpha=0.5, beta=0.5):
    return alpha * similarity + beta * rating / 100

# https://huggingface.co/docs/transformers/model_doc/bert

def recommend_bert(title, df, embeddings_file='bert_embeddings.pkl'):
    # print("Column names:", df.columns.tolist())

    # Load precomputed embeddings from pickle
    if 'embeddings' not in df.columns:
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        df['embeddings'] = list(embeddings)

    title_mask = df['Title'] == title
    if not title_mask.any():
        return f"No recipes found with title: {title}"

    query_embedding = df.loc[title_mask, 'embeddings'].values[0]

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], list(df['embeddings']))[0]
    df['Similarity'] = similarities

    recommended_df = df.sort_values(by='Similarity', ascending=False).head(5)
    recommended_recipes = recommended_df['Title'].tolist()

    return recommended_recipes



def get_bert_embeddings(texts, batch_size=50):
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
            # Use mean of the last hidden states
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).to('cpu').numpy()
            embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def compute_and_save_embeddings(df, filename='bert_embeddings.pkl'):
    ''' Compute BERT embeddings for all titles in the provided DataFrame and save them to a file.'''

    embeddings = get_bert_embeddings(df['Title'].tolist())
    # Save embeddings with pickle
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings saved to", filename)


def recommend1_bert(title, df, embeddings_file='bert_embeddings.pkl'):
    
    # TODO
    # Load precomputed embeddings from pickle
    if 'embeddings' not in df.columns:
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        df['embeddings'] = list(embeddings)

    # print(f"shape of embeddings: {df['embeddings'][0][0].shape}")

    title_mask = df['Title'] == title
    if not title_mask.any():
        return f"No recipes found with title: {title}"

    query_embedding = df.loc[title_mask, 'embeddings'].values[0]

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], list(df['embeddings']))[0]
    
    

    df['Similarity'] = similarities
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

# def recommend2_bert(inputValue1, inputValue2, df, embeddings_file='bert_embeddings.pkl'):
#     filtered_df = df.copy()
#     # print(f"length of filtered_df: {len(filtered_df)}")
   

#     def not_include(lst, str):
#         return all(ele not in str for ele in lst)
    
  
#     # TODO
#     if 'embeddings' not in filtered_df.columns:
#         with open(embeddings_file, 'rb') as f:
#             embeddings = pickle.load(f)
#         filtered_df['embeddings'] = list(embeddings)
#     # print(f"filtered_df: {filtered_df['Title']}")
#     # print(f"length of df['embeddings']: {len(df['embeddings'])}")
    
#     title_mask = filtered_df['Title'] == inputValue2
#     print(f"true or false: {title_mask.any()}")
#     if not title_mask.any():
#         return f"No recipes found with title: {inputValue2}"
    
#     if inputValue1 != "":
#         inputValue1_lst = re.split('[, ]+', inputValue1.lower())
#         title_mask = filtered_df.apply(lambda row: not_include(inputValue1_lst, row['Cleaned_Ingredients']), axis=1)
#         filtered_df = filtered_df[title_mask]

#     query_embedding = filtered_df .loc[title_mask, 'embeddings'].values[0]

#     # Calculate similarities
#     similarities = cosine_similarity([query_embedding], list(filtered_df ['embeddings']))[0]

#     filtered_df['Similarity'] = similarities
#     filtered_df['Recommendation_Metric'] = recommendation_metric(filtered_df['Similarity'], filtered_df['Rating'])
#     recommended_df = filtered_df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
#     recommended_recipes_titles = []
#     recommended_recipes_images = []
#     # recommended_recipes_ratings = []
#     recommended_recipes_stats = []
#     recommended_recipes_instructions =[]

#     for recipes_title in recommended_df['Title']:
#         recommended_recipes_titles.append(recipes_title)
#     for recipes_image in recommended_df['Image_Name']:
#         recommended_recipes_images.append('../archive/images/' + recipes_image + '.jpg')
#     # for recipes_rating in recommended_df['Rating']:
#     #     recommended_recipes_ratings.append(recipes_rating)
#     for _, row in recommended_df.iterrows():
#         recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
#     for recipes_instruction in recommended_df['Instructions']:
#         recommended_recipes_instructions.append(recipes_instruction)
        
#     return recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions


def recommend2_bert(inputValue1, inputValue2, df, embeddings_file='bert_embeddings.pkl'):
    # Make a copy of the dataframe to work with
    filtered_df = df.copy()
    
    # Load precomputed embeddings from pickle
    if 'embeddings' not in filtered_df.columns:
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        filtered_df['embeddings'] = list(embeddings)
    
    # Check if the provided title exists in the DataFrame
    title_mask = filtered_df['Title'] == inputValue2
    if not title_mask.any():
        return f"No recipes found with title: {inputValue2}"

    query_embedding = filtered_df.loc[title_mask, 'embeddings'].values[0]

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], list(filtered_df['embeddings']))[0]
    
    # Assigning the similarity scores to the DataFrame and calculating a recommendation metric
    filtered_df['Similarity'] = similarities
    # Assume recommendation_metric function is defined elsewhere
    filtered_df['Recommendation_Metric'] = recommendation_metric(filtered_df['Similarity'], filtered_df['Rating'])
    recommended_df = filtered_df.sort_values(by='Recommendation_Metric', ascending=False).head(5)

    # Prepare to filter out unwanted keywords
    if inputValue1:
        inputValue1_lst = re.split('[, ]+', inputValue1.lower())
        recommended_df = recommended_df[~recommended_df['Title'].apply(lambda x: any(unw in x.lower() for unw in inputValue1_lst))]

    # Initializing lists to hold the filtered recommendations
    recommended_recipes_titles = []
    recommended_recipes_images = []
    recommended_recipes_stats = []
    recommended_recipes_instructions = []
    
    # Iterating over the filtered recommendations
    for _, row in recommended_df.iterrows():
        recommended_recipes_titles.append(row['Title'])
        recommended_recipes_images.append('../archive/images/' + row['Image_Name'] + '.jpg')
        recommended_recipes_stats.append(f"{round(row['Similarity'], 4)} | {row['Rating']} | {round(row['Recommendation_Metric'], 4)}")
        recommended_recipes_instructions.append(row['Instructions'])
    
    return recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions