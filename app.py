import streamlit as st
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pickle.load(open("data.pkl", 'rb'))

recipe_list = data['Title'].values

def recommend1(title, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Title'])
    similarity = cosine_similarity(tfidf_matrix)
    
    index = df[df['Title'] == title].index[0]
    recommended_list = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_recipes = []
    
    for i in recommended_list[0:5]:
        recommended_recipes.append(df.iloc[i[0]].Title)
    
    return recommended_recipes

def recommend2(inputValue, df):
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the B attribute and transform both the input text and the text in B
    all_texts = df['Cleaned_Ingredients'].tolist() + [inputValue]
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    # Get the TF-IDF vector for the input text (the last one in the matrix)
    input_vector = tfidf_matrix[-1]

    # Compute cosine similarity between the input vector and all vectors in the matrix (except the last one)
    similarity = cosine_similarity(tfidf_matrix[:-1], input_vector)

    # Flatten the array of similarities, make it part of the dataframe
    df['Similarity'] = similarity.flatten()

    # Sort the DataFrame by similarity in descending order and select the top 5
    recommended_list = df.sort_values(by='Similarity', ascending=False).head(5)
    recommended_recipes = []
    
    for recipes in recommended_list['Title']:
        recommended_recipes.append(recipes)
        
    return recommended_recipes


st.header("Recipe Recommendation System")

selectValue = st.selectbox("Select recipe from dropdown:", recipe_list)

if st.button("Show Recommended Recipes with Similar Ingredients"):
    recommended_recipes = recommend1(selectValue, data)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_recipes[0])
    with col2:
        st.text(recommended_recipes[1])
    with col3:
        st.text(recommended_recipes[2])
    with col4:
        st.text(recommended_recipes[3])
    with col5:
        st.text(recommended_recipes[4])

inputValue = st.text_input("Enter ingredients you would like to cook with:")

if st.button("Show Recommended Recipes with these Ingredients"):
    recommended_recipes = recommend2(inputValue, data)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_recipes[0])
    with col2:
        st.text(recommended_recipes[1])
    with col3:
        st.text(recommended_recipes[2])
    with col4:
        st.text(recommended_recipes[3])
    with col5:
        st.text(recommended_recipes[4])