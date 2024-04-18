import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pickle.load(open("data.pkl", 'rb'))

recipe_list = data['Title'].values

def recommend1(title, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Ingredients'])
    similarity = cosine_similarity(tfidf_matrix)
    
    index = df[df['Title'] == title].index[0]
    recommended_list = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_recipes = []
    
    for i in recommended_list[0:5]:
        recommended_recipes.append(df.iloc[i[0]].Title)
    
    return recommended_recipes

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

    filtered_df['Similarity'] = similarity.flatten()

    recommended_list = filtered_df.sort_values(by='Similarity', ascending=False).head(5)
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

inputValue1 = st.text_input("Enter ingredients you don't want to cook with:")
inputValue2 = st.text_input("Enter ingredients you would like to cook with:")

if st.button("Show Recommended Recipes with these Ingredients"):
    recommended_recipes = recommend2(inputValue1, inputValue2, data)
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