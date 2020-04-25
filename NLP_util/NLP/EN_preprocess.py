import pandas as pd # our main data management package
import matplotlib.pyplot as plt # our main display package
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # our model
from sklearn.model_selection import train_test_split



def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return new_text

# make all text lowercase
def text_lowercase(text):
    return text.lower()
# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result
# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text


# remove stopwords defined in NLTK
def remove_stopwords(text , stop_words):
    text = [i for i in text if not i in stop_words]
    return text


def lemmatize(text ,lemmatizer):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

class NLP_Eng_Processor:
    
    def __init__(self ,  df ):
        """
        df(DataFrame) : Raw data in text data frame format
        
        """
        self.df = df.copy()
        self.original_text = df['text'].copy()
         # remove stopwords
        self.stop_words = set(stopwords.words('english'))
         # lemmatize
        self.lemmatizer = WordNetLemmatizer()
    
   

    def preprocessing(self , text ,I_want_lemmatize = False):
        
        try:
            text = remove_numbers(text)
            text = remove_punctuation(text)
            text = text_lowercase(text)
            text = remove_urls(text)
            text = tokenize(text)
            text = remove_stopwords(text , self.stop_words)
            if I_want_lemmatize:
                text = lemmatize(text , self.lemmatizer)
            text = ' '.join(text)
        except:
            text = ' '
        return text 
    
    def run_preprocess_script(self  ,I_want_lemmatize = False):
        """
        Exe the script to preprocess dataframe text 
        """
        all_text = self.df['text']
        clean_text  = []
        for text in all_text:
            clean_text.append(self.preprocessing(text , I_want_lemmatize ))
            
        # add the preprocessed text as a column
        self.df['text'] = clean_text
        temp_df = pd.DataFrame([ pd.Series(clean_text) , self.original_text]).T
        temp_df.columns = ['clean_text' , 'original_text']
        return temp_df

        
        
    