import streamlit as st
import pickle
import requests

from recommend import recommend1, recommend2

data = pickle.load(open("data.pkl", 'rb'))
recipe_list = data['Title'].values

####################################################################################################
# Function 1: Show Recommended Recipes with Similar Ingredients
####################################################################################################

st.header("Recipe Recommendation System")

with st.container():
    selectValue1 = st.selectbox("Select a recipe from dropdown:", recipe_list)
    
    if st.button("Show Recommended Recipes with Similar Ingredients"):
        data = pickle.load(open("data.pkl", 'rb'))
        recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions  = recommend1(selectValue1, data)
        
        for i in range(len(recommended_recipes_titles)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(recommended_recipes_images[i])
            
            with col2:
                with st.expander(recommended_recipes_titles[i]):
                    st.image(recommended_recipes_images[i])
                    st.text(recommended_recipes_stats[i])
                    st.markdown("### Recipe Instruction")
                    st.write(recommended_recipes_instructions[i])
st.markdown("---")

####################################################################################################
# Function 2: Show Recommended Recipes with Given Ingredients
####################################################################################################

with st.container():
    inputValue1 = st.text_input("Enter ingredients you don't want to cook with:")
    inputValue2 = st.text_input("Enter ingredients you would like to cook with:")

    if st.button("Show Recommended Recipes with Given Ingredients"):
        data = pickle.load(open("data.pkl", 'rb'))
        recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats,recommended_recipes_instructions  = recommend2(inputValue1, inputValue2, data)
        
        for i in range(len(recommended_recipes_titles)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(recommended_recipes_images[i])
            
            with col2:
                with st.expander(recommended_recipes_titles[i]):
                    st.image(recommended_recipes_images[i])
                    st.text(recommended_recipes_stats[i])
                    st.markdown("### Recipe Instruction")
                    st.write(recommended_recipes_instructions[i])
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
        
        recommended_recipes_titles, recommended_recipes_images, recommended_recipes_stats, recommended_recipes_instructions  = recommend2("", inputValue3, data)
        
        for i in range(len(recommended_recipes_titles)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(recommended_recipes_images[i])
            
            with col2:
                with st.expander(recommended_recipes_titles[i]):
                    st.image(recommended_recipes_images[i])
                    st.text(recommended_recipes_stats[i])
                    st.markdown("### Recipe Instruction")
                    st.write(recommended_recipes_instructions[i])
st.markdown("---")  

####################################################################################################
# Function 4: Search and Rate Recipes
####################################################################################################

def search(title, df):
    searched_df = df[df['Title'] == title]

    searched_recipes_titles = []
    searched_recipes_images = []
    # searched_recipes_ratings = []
    searched_recipes_stats = []
    searched_recipes_instructions =[]

    for recipes_title in searched_df['Title']:
        searched_recipes_titles.append(recipes_title)
    for recipes_image in searched_df['Image_Name']:
        searched_recipes_images.append('archive/images/' + recipes_image + '.jpg')
    # for recipes_rating in searched_df['Rating']:
    #     searched_recipes_ratings.append(recipes_rating)
    for _, row in searched_df.iterrows():
        searched_recipes_stats.append(str(row['Rating']))
    for recipes_instruction in searched_df['Instructions']:
        searched_recipes_instructions.append(recipes_instruction)
        
    return searched_recipes_titles, searched_recipes_images, searched_recipes_stats, searched_recipes_instructions

def update_rating(title, rating, df):
    df.at[df[df['Title'] == title].index[0], 'Rating'] = rating
    df.to_pickle("data.pkl")

with st.container():
    inputValue4 = st.text_input("Search a recipe to rate:")
    
    if st.button("Search"):
        data = pickle.load(open("data.pkl", 'rb'))
        searched_recipes_titles, searched_recipes_images, searched_recipes_stats, searched_recipes_instructions = search(inputValue4, data) 

        if not searched_recipes_titles:
            st.error("Recipe doesn't exist!")
        
        for i in range(len(searched_recipes_titles)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(searched_recipes_images[i])
            
            with col2:
                with st.expander(searched_recipes_titles[i]):
                    st.image(searched_recipes_images[i])
                    st.text(searched_recipes_stats[i])
                    st.markdown("### Recipe Instruction")
                    st.write(searched_recipes_instructions[i])

    user_rating = st.slider("Rate this recipe:", 1, 100, 50)

    if st.button("Submit Rating"):
        data = pickle.load(open("data.pkl", 'rb'))
        update_rating(inputValue4, user_rating, data)
        st.success("Thank you for rating!")