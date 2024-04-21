import streamlit as st
import pickle
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pickle.load(open("data.pkl", 'rb'))
recipe_list = data['Title'].values

####################################################################################################
# Recommendation Metric
####################################################################################################

def recommendation_metric(similarity, rating, alpha=0.5, beta=0.5):
    return alpha * similarity + beta * rating / 100

####################################################################################################
# Recommendation Function
####################################################################################################

def recommend1(title, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Title'])
    similarity = cosine_similarity(tfidf_matrix)
    
    index = df[df['Title'] == title].index[0]

    df['Similarity'] = similarity[index]
    df['Recommendation_Metric'] = recommendation_metric(similarity[index], df['Rating'])
    recommended_df = df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
    recommended_recipes = []
    recommended_recipes_images = []
    # recommended_recipes_ratings = []
    recommended_recipes_stats = []
    recommended_recipes_intro =[]
    
    for recipes_title in recommended_df['Title']:
        recommended_recipes.append(recipes_title)
    for recipes_intro in recommended_df['Instructions']:
        recommended_recipes_intro.append(recipes_intro)
    for recipes_image in recommended_df['Image_Name']:
        recommended_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in recommended_df['Rating']:
    #     recommended_recipes_ratings.append(recipes_rating)
    for _, row in recommended_df.iterrows():
        recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
    
    return recommended_recipes, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_intro

def recommend2(inputValue1, inputValue2, df):
    filtered_df = df.copy()

    def not_include(lst, str):
        return all(ele not in str for ele in lst)
    
    if inputValue1 != "":
        inputValue1_lst = re.split('[, ]+', inputValue1.lower())
        mask = filtered_df.apply(lambda row: not_include(inputValue1_lst, row['Cleaned_Ingredients']), axis=1)
        filtered_df = filtered_df[mask]
        
    tfidf_vectorizer = TfidfVectorizer()

    all_texts = filtered_df['Cleaned_Ingredients'].tolist() + [inputValue2]
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    input_vector = tfidf_matrix[-1]
    similarity = cosine_similarity(tfidf_matrix[:-1], input_vector)

    filtered_df['Similarity'] = similarity
    filtered_df['Recommendation_Metric'] = recommendation_metric(similarity.squeeze(), filtered_df['Rating'])
    recommended_df = filtered_df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
    recommended_recipes = []
    recommended_recipes_images = []
    # recommended_recipes_ratings = []
    recommended_recipes_stats = []
    recommended_recipes_intro =[]

    for recipes_intro in recommended_df['Instructions']:
        recommended_recipes_intro.append(recipes_intro)
    for recipes_title in recommended_df['Title']:
        recommended_recipes.append(recipes_title)
    for recipes_image in recommended_df['Image_Name']:
        recommended_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in recommended_df['Rating']:
    #     recommended_recipes_ratings.append(recipes_rating)
    for _, row in recommended_df.iterrows():
        recommended_recipes_stats.append(str(round(row['Similarity'], 4)) + ' | ' + str(row['Rating']) + ' | ' + str(round(row['Recommendation_Metric'], 4)))
        
    return recommended_recipes, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_intro


####################################################################################################
# Function 1: Show Recommended Recipes with Similar Ingredients
####################################################################################################

st.header("Recipe Recommendation System")
with st.container():
    selectValue1 = st.selectbox("Select a recipe from dropdown:", recipe_list)
    if st.button("Show Recommended Recipes with Similar Ingredients"):
        data = pickle.load(open("data.pkl", 'rb'))
        recommended_recipes, recommended_recipes_images, recommended_recipes_stats,recommended_recipes_intro  = recommend1(selectValue1, data)
        for i in range(len(recommended_recipes)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(recommended_recipes_images[i])
            
            with col2:
                with st.expander(recommended_recipes[i]):
                    st.image(recommended_recipes_images[i])
                    st.text(recommended_recipes_stats[i])
                    st.markdown("### Receipt Intro")
                    st.write(recommended_recipes_intro[i])
st.markdown("---")
####################################################################################################
# Function 2: Show Recommended Recipes with Given Ingredients
####################################################################################################
with st.container():
    inputValue1 = st.text_input("Enter ingredients you don't want to cook with:")
    inputValue2 = st.text_input("Enter ingredients you would like to cook with:")

    if st.button("Show Recommended Recipes with Given Ingredients"):
        data = pickle.load(open("data.pkl", 'rb'))
        recommended_recipes, recommended_recipes_images, recommended_recipes_stats,recommended_recipes_intro  = recommend2(selectValue1,inputValue2, data)
        for i in range(len(recommended_recipes)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(recommended_recipes_images[i])
            
            with col2:
                with st.expander(recommended_recipes[i]):
                    st.image(recommended_recipes_images[i])
                    st.text(recommended_recipes_stats[i])
                    st.markdown("### Receipt Intro")
                    st.write(recommended_recipes_intro[i])
st.markdown("---")                
####################################################################################################
# Function 3: Natural Languages Handle
####################################################################################################

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  
        return response.text  
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None  
    
with st.container():
    inputValue3 = st.text_input("Enter natural query to describe what Ingredients you have and what receipt you like")
    if st.button("Submit your natural query"):
        data = pickle.load(open("data.pkl", 'rb'))
        API_URL = "https://api-inference.huggingface.co/models/ilsilfverskiold/tech-keywords-extractor"
        headers = {"Authorization": "Bearer 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"}
        output = query(inputValue3)
        recommended_recipes, recommended_recipes_images, recommended_recipes_stats,recommended_recipes_intro  = recommend2( "", inputValue3, data)
        for i in range(len(recommended_recipes)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(recommended_recipes_images[i])
            
            with col2:
                with st.expander(recommended_recipes[i]):
                    st.image(recommended_recipes_images[i])
                    st.text(recommended_recipes_stats[i])
                    st.markdown("### Receipt Intro")
                    st.write(recommended_recipes_intro[i])
st.markdown("---")  
####################################################################################################
# Function 4: Rate Recipes
####################################################################################################
with st.container():
    selectValue2 = st.selectbox("Select a recipe to rate:", recipe_list)
    user_rating = st.slider("Rate this recipe:", 1, 100, 50)

    def update_rating(title, rating, df):
        df.at[df[df['Title'] == title].index[0], 'Rating'] = rating
        df.to_pickle("data.pkl")

    if st.button("Submit Rating"):
        data = pickle.load(open("data.pkl", 'rb'))
        update_rating(selectValue2, user_rating, data)
        st.success("Thank you for rating!")
    
