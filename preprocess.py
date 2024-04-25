import pandas as pd
import re
import spacy
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter

# Don't use word_tokenize
def extract_words(ingredients):
    word_pattern = r'\b\w+\b'
    # list of strings
    words_list = re.findall(word_pattern, ingredients.lower())
    # convert it to a string
    words_str = ' '.join(words_list)
    return words_str

# Super slow, but super accurate!
def lemmatization(ingredients):
    # Download necessary spaCy resource
    # spacy.cli.download("en_core_web_sm")
    
    # Load English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(ingredients)

    # Lemmatize each token and remove duplicates!
    lemmatized_words_list = list(set([token.lemma_ for token in doc]))
    lemmatized_words_str = ' '.join(lemmatized_words_list)
    
    return lemmatized_words_str

# def lemmatization(ingredients):
#     # Download necessary NLTK resource
#     # nltk.download('wordnet')

#     lemmatizer = WordNetLemmatizer()
#     words_list = word_tokenize(ingredients)
    
#     # Lemmatize each token and remove duplicates!
#     lemmatized_words_list = list(set([lemmatizer.lemmatize(word) for word in words_list]))
#     lemmatized_words_str = ' '.join(lemmatized_words_list)

#     return lemmatized_words_str

def remove_stop_words(ingredients):
    # Download necessary NLTK resources
    # nltk.download('punkt')
    # nltk.download('stopwords')
    
    # Load English stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenize the string into words, this is similar to what we did in extract_words
    words_list = word_tokenize(ingredients)
    
    filtered_words_list = [word for word in words_list if word not in stop_words]
    filtered_words_str = ' '.join(filtered_words_list)
    return filtered_words_str

def remove_numbers(ingredients):  
    integer_float_pattern = r'\b\d*\.?\d+\b'
    unicode_fraction_pattern = r'\b\d*[\u00BC-\u00BE\u2150-\u215E]\b'
    
    combined_pattern = f'{integer_float_pattern}|{unicode_fraction_pattern}'
    new_ingredients = re.sub(combined_pattern, '', ingredients)
    
    # remove extra space
    new_ingredients = ' '.join(new_ingredients.split())
    return new_ingredients

def remove_cooking_metrics(ingredients):
    cooking_metrics = ['ml', 'milliliter', 'millilitre', 'cc', 'l', 'liter', 'litre', 'dl', 'deciliter', 'decilitre',
                       'teaspoon', 't', 'tsp', 'tablespoon', 'tbl', 'tbs', 'tbsp', 'fluid ounce', 'fl oz', 'gill', 'cup', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 
                       'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram', 'kilogramme', 'pound', 'lb', 'ounce', 'oz',
                       'mm', 'millimeter', 'millimetre', 'cm', 'centimeter', 'centimetre', 'm', 'meter', 'metre', 'inch', 'in', 'yard',
                       'milli', 'centi', 'deci', 'hecto', 'kilo']
    cooking_metrics_pattern = r'\b(' + '|'.join(re.escape(metric) for metric in cooking_metrics) + r')\b'
    
    new_ingredients = re.sub(cooking_metrics_pattern, '', ingredients)
    new_ingredients = ' '.join(new_ingredients.split())
    return new_ingredients

def data_cleaning_1(ingredients):
    return remove_cooking_metrics(remove_numbers(remove_stop_words(lemmatization(extract_words(ingredients)))))
    
def find_common_items(df):
    string_list = []
    for ingredients in df['Cleaned_Ingredients']:
        string_list.append(ingredients)
    
    # combine list of strings into a single string
    total_string = ' '.join(string_list)
    words_list = re.findall(r'\b\w+\b', total_string)
    return Counter(words_list).most_common(50)  

# NLTK lemmatization
# [('salt', 9002), ('oil', 7387), ('fresh', 6356), ('chopped', 6340), ('large', 5674), ('pepper', 5374), ('olive', 5007), ('ground', 4925), ('sugar', 4816), ('kosher', 4775), ('butter', 4371), ('garlic', 4195), ('cut', 3880), ('black', 3842), ('clove', 3831), ('finely', 3822), ('juice', 3820), ('sliced', 3745), ('plus', 3650), ('unsalted', 3546), ('freshly', 3452), ('egg', 3442), ('leaf', 3336), ('onion', 3217), ('lemon', 3201), ('grated', 3165), ('white', 3121), ('red', 3092), ('peeled', 2917), ('flour', 2855), ('divided', 2851), ('whole', 2570), ('thinly', 2497), ('piece', 2491), ('extra', 2477), ('vegetable', 2419), ('cream', 2410), ('stick', 2390), ('water', 2359), ('purpose', 2356), ('virgin', 2327), ('vinegar', 2315), ('medium', 2263), ('small', 2233), ('dried', 1923), ('milk', 1906), ('wine', 1877), ('powder', 1876), ('green', 1855), ('halved', 1821)]

# spaCy lemmatization
# [('salt', 9074), ('oil', 7389), ('chop', 6410), ('fresh', 6357), ('large', 5698), ('pepper', 5375), ('olive', 5007), ('sugar', 4816), ('kosher', 4775), ('slice', 4599), ('butter', 4375), ('garlic', 4195), ('ground', 4056), ('cut', 3882), ('black', 3842), ('juice', 3837), ('clove', 3831), ('finely', 3822), ('plus', 3650), ('freshly', 3452), ('egg', 3442), ('unsalted', 3378), ('peel', 3359), ('onion', 3217), ('lemon', 3201), ('dry', 3140), ('white', 3121), ('red', 3092), ('flour', 2856), ('divide', 2832), ('leave', 2826), ('whole', 2570), ('thinly', 2497), ('piece', 2491), ('extra', 2476), ('vegetable', 2419), ('cream', 2413), ('stick', 2391), ('water', 2359), ('purpose', 2356), ('virgin', 2327), ('vinegar', 2315), ('medium', 2263), ('small', 2249), ('seed', 2204), ('powder', 2039), ('grate', 2037), ('milk', 1906), ('wine', 1877), ('halve', 1861)]

def remove_common_items(ingredients):
    common_items = ['salt', 'oil', 'chop', 'fresh', 'large', 'sugar', 'kosher', 'slice', 'ground', 'cut', 'black', 'juice', 'clove', 'finely', 'plus', 'freshly', 'unsalted', 'peel', 'dry', 'white', 'red', 'divide', 'leave', 'whole', 'thinly', 'piece', 'extra', 'vegetable', 'stick', 'water', 'purpose', 'virgin', 'medium', 'small', 'seed', 'powder', 'grate', 'halve']

    common_items_pattern = r'\b(' + '|'.join(re.escape(metric) for metric in common_items) + r')\b'
    
    new_ingredients = re.sub(common_items_pattern, '', ingredients)
    new_ingredients = ' '.join(new_ingredients.split())
    return new_ingredients

def data_cleaning_2(ingredients):
    return remove_common_items(ingredients)

if __name__ == "__main__":
    data = pd.read_csv('archive/data.csv')
    data.fillna("", inplace=True)
    data.drop(columns=[data.columns[0], 'Cleaned_Ingredients'], inplace=True)
    data['Rating'] = 50

    data['Cleaned_Ingredients'] = data['Ingredients'].apply(lambda ingredients: data_cleaning_1(ingredients))
    # print(find_common_items(data))
    data['Cleaned_Ingredients'] = data['Cleaned_Ingredients'].apply(lambda ingredients: data_cleaning_2(ingredients))

    pickle.dump(data, open('data.pkl', 'wb'))
    pickle.dump(data, open('archive/data.pkl', 'wb'))
    pickle.dump(data, open('Evaluation/data.pkl', 'wb'))