import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pickle.load(open("data.pkl", 'rb'))
recipe_list = data['Title'].values

####################################################################################################
# Recommendation Metric
####################################################################################################

def recommendation_metric(similarity, rating, alpha=0.5, beta=0.5):
    return alpha * similarity + beta * rating

####################################################################################################
# Recommendation Function
####################################################################################################

def recommend1(title, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Title'])
    similarity = cosine_similarity(tfidf_matrix)
    
    index = df[df['Title'] == title].index[0]

    df['Recommendation_Metric'] = recommendation_metric(similarity[index], df['Rating'])
    recommended_list = df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
    recommended_recipes = []
    recommended_recipes_images = []
    recommended_recipes_ratings = []
    
    for recipes_title in recommended_list['Title']:
        recommended_recipes.append(recipes_title)
    for recipes_image in recommended_list['Image_Name']:
        recommended_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    for recipes_rating in recommended_list['Rating']:
        recommended_recipes_ratings.append(recipes_rating)
    
    return recommended_recipes, recommended_recipes_images, recommended_recipes_ratings

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

    filtered_df['Recommendation_Metric'] = recommendation_metric(similarity.squeeze(), filtered_df['Rating'])
    recommended_list = filtered_df.sort_values(by='Recommendation_Metric', ascending=False).head(5)
    
    recommended_recipes = []
    recommended_recipes_images = []
    recommended_recipes_ratings = []
    
    for recipes_title in recommended_list['Title']:
        recommended_recipes.append(recipes_title)
    for recipes_image in recommended_list['Image_Name']:
        recommended_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    for recipes_rating in recommended_list['Rating']:
        recommended_recipes_ratings.append(recipes_rating)
        
    return recommended_recipes, recommended_recipes_images, recommended_recipes_ratings

####################################################################################################
# Function 1: Show Recommended Recipes with Similar Ingredients
####################################################################################################

st.header("Recipe Recommendation System")

selectValue1 = st.selectbox("Select a recipe from dropdown:", recipe_list)

if st.button("Show Recommended Recipes with Similar Ingredients"):
    data = pickle.load(open("data.pkl", 'rb'))
    recommended_recipes, recommended_recipes_images, recommended_recipes_ratings = recommend1(selectValue1, data)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_recipes[0])
        st.image(recommended_recipes_images[0])
        st.text(recommended_recipes_ratings[0])
    with col2:
        st.text(recommended_recipes[1])
        st.image(recommended_recipes_images[1])
        st.text(recommended_recipes_ratings[1])
    with col3:
        st.text(recommended_recipes[2])
        st.image(recommended_recipes_images[2])
        st.text(recommended_recipes_ratings[2])
    with col4:
        st.text(recommended_recipes[3])
        st.image(recommended_recipes_images[3])
        st.text(recommended_recipes_ratings[3])
    with col5:
        st.text(recommended_recipes[4])
        st.image(recommended_recipes_images[4])
        st.text(recommended_recipes_ratings[4])

####################################################################################################
# Function 2: Show Recommended Recipes with Given Ingredients
####################################################################################################

inputValue1 = st.text_input("Enter ingredients you don't want to cook with:")
inputValue2 = st.text_input("Enter ingredients you would like to cook with:")

if st.button("Show Recommended Recipes with Given Ingredients"):
    data = pickle.load(open("data.pkl", 'rb'))
    recommended_recipes, recommended_recipes_images, recommended_recipes_ratings = recommend2(inputValue1, inputValue2, data)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_recipes[0])
        st.image(recommended_recipes_images[0])
        st.text(recommended_recipes_ratings[0])
    with col2:
        st.text(recommended_recipes[1])
        st.image(recommended_recipes_images[1])
        st.text(recommended_recipes_ratings[1])
    with col3:
        st.text(recommended_recipes[2])
        st.image(recommended_recipes_images[2])
        st.text(recommended_recipes_ratings[2])
    with col4:
        st.text(recommended_recipes[3])
        st.image(recommended_recipes_images[3])
        st.text(recommended_recipes_ratings[3])
    with col5:
        st.text(recommended_recipes[4])
        st.image(recommended_recipes_images[4])
        st.text(recommended_recipes_ratings[4])

####################################################################################################
# Function 3: Rate Recipes
####################################################################################################

selectValue2 = st.selectbox("Select a recipe to rate:", recipe_list)
user_rating = st.slider("Rate this recipe:", 1, 100, 50)

def update_rating(title, rating, df):
    df.at[df[df['Title'] == title].index[0], 'Rating'] = rating
    df.to_pickle("data.pkl")

if st.button("Submit Rating"):
    data = pickle.load(open("data.pkl", 'rb'))
    update_rating(selectValue2, user_rating, data)
    st.success("Thank you for rating!")