# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd


import malaya

import nltk
from nltk.corpus import stopwords

import re


import streamlit as st

english_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

from stop_words import get_stop_words

from module.helper_functions import open_html

from lda import main_function

import json
import text_eda.plots as plots 
import streamlit as st
import pandas as pd
import text_eda.preprocessor as pp
import streamlit.components.v1 as components
from PIL import Image
from helper_functions import *

import streamlit as st
# components allow for pyLDAvis interactive graph display
import streamlit.components.v1 as components
import spacy
spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from gensim import corpora
import random
import pyLDAvis.gensim
import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import math
from text_eda.helper_functions import open_html
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
def main():

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)

            
    @st.cache(suppress_st_warning=True)
    def load_data(uploaded_file):
        

        df = pd.read_csv(uploaded_file)
                
 
        return df
    



        
    uploaded_file = st.file_uploader('Upload CSV file to begin', type='csv')

    #if upload then show left bar
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        AgGrid(df)

        """
        this function selects the text feature from the uploaded csv file
        ----------
        df: A pandas Dataframe 
        """
        text_column = st.selectbox('Select the text column',(list(df.columns)))
        
        df = df[text_column]
        data =  pd.DataFrame(df)
            

        
        st.write("Welcome to the DQW for Text analysis. ",
                    "As unstructured data, text input analysis for ",
                    "NLP models is of crucial importance. This dashboard ",
                    "offers visualisation of descriptive statistics of a ",
                    "text input file uploaded in form of csv or txt. ",
                    "Please select input method on the left, pick if you wish to ",
                    "preprocess it and select the plot you want to use to analyse it.")
        
        # Side panel setup
        # Step 1 includes Uploading and Preprocessing data (optional)
        # display_app_header(main_txt = "Step 1",
        #                 sub_txt= "Upload data",
        #                 is_sidebar=True)
        
        # data_input_mthd = st.sidebar.radio("Select Data Input Method",
        #                                 ('Copy-Paste text', 
        #                                     'Upload a CSV file',
        #                                     'Import a json file'))
        
        # st.subheader('Choose data to analyse :alembic:')
        # data,txt  = check_input_method(data_input_mthd)
        
        # data,text_column = select_text_feature(data)
        
        # display_app_header_1(sub_txt= "Preprocess data",
        #                 is_sidebar=True)
        
        # clean_data_opt = st.sidebar.radio("Choose wisely",
        #                                 ('Skip preprocessing', 
        #                                 'Run preprocessing'))
        
        # # clean data #######
        # if clean_data_opt=='Skip preprocessing':
        #         st.subheader('Using Raw data :cut_of_meat:')  #Raw data header
                
        #         display_app_header(main_txt = "Step 2",
        #             sub_txt= "Analyse data",
        #             is_sidebar=True)
                
        #         selected_plot = st.sidebar.radio(
        #         "Choose 1 plot", ('Length of text', 
        #                         'Word count',
        #                         'Average word length',
        #                         'Stopwords',
        #                         'Unique word count',
        #                         'N-grams',
        #                         'Topic modelling',
        #                         'Wordcloud',
        #                         'Sentiment',
        #                         'NER',
        #                         'POS',
        #                         'Complexity Scores')
        #         )
                
        # else:
        st.subheader('Using Clean Data :droplet:')  #Clean data header

        data = pp.clean_data(data,feature=text_column)
        st.success('Data cleaning successfuly done!')
        
        image = Image.open("text_eda/pp.png")
        st.image(image, caption='Preprocessing steps done by DQW')

        # display_app_header(main_txt = "Step 2",
        #                 sub_txt= "Analyse data",
        #                 is_sidebar=True)
            
        selected_plot = st.sidebar.radio(
        "Choose 1 plot", ('Length of text', 
                        'Word count',
                        'Average word length',
                        'Unique word count',
                        'N-grams',
                        'Topic modelling',
                        'Wordcloud',
                        'Sentiment',
                        'NER',
                        'POS',
                        'Complexity Scores')
        )
        # final step
        st.download_button(
            label="Download clean data",
            data=data.to_csv().encode('utf-8'),
            file_name='clean_data.csv',
            mime='text/csv',
        )
                        
        st.subheader('A preview of input data is below, please select plot to start analysis :bar_chart:')
        st.write(data.head(5))
        
        plots.plot(selected_plot,
                data,
                text_column)
        
        def tokenize(text):
            lda_tokens = []
            tokens = parser(text)
            for token in tokens:
                if token.orth_.isspace():
                    continue
                elif token.like_url:
                    lda_tokens.append('URL')
                elif token.orth_.startswith('@'):
                    lda_tokens.append('SCREEN_NAME')
                else:
                    lda_tokens.append(token.lower_)
            return lda_tokens   
        
        def get_lemma(word):
            lemma = wn.morphy(word)
            if lemma is None:
                return word
            else:
                return lemma
            
        def get_lemma2(word):
            return WordNetLemmatizer().lemmatize(word)

        def prepare_text_for_lda(text):
            tokens = tokenize(text)
            tokens = [token for token in tokens if len(token) > 4]
            tokens = [token for token in tokens if token not in en_stop]
            tokens = [get_lemma(token) for token in tokens]
            return tokens
        
        data = data[data[text_column].notnull()]
        input = prepare_text_for_lda(str(data[text_column]))
 


if __name__ == '__main__':
    main()