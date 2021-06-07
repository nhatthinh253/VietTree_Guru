# import library
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.metrics import Recall
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import random
import seaborn as sns

# for Quiz generation
import wikipedia
# import pprint
# import itertools
import re
# import pke
# import string
# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize
# from flashtext import KeywordProcessor

#MCQ
# import requests
# import json
# import re
# import random
# from pywsd.similarity import max_similarity
# from pywsd.lesk import adapted_lesk
# from pywsd.lesk import simple_lesk
# from pywsd.lesk import cosine_lesk
# from nltk.corpus import wordnet as wn
import pickle



# disable warning
st.set_option('deprecation.showfileUploaderEncoding', False)


from PIL import Image

@st.cache(allow_output_mutation = True)
def load_banner(path = './App photos/banner0.png'):

    banner = Image.open(path)
    return banner
    

banner = load_banner()
st.image(banner,
use_column_width=True
)


## Tiltle
#st.title('Plant Quiz - Tree Guru')
st.markdown(f'<h1 style="text-align: center; color: black;">VietTree Guru</h1>', 
                unsafe_allow_html=True) 

# Bodys
@st.cache()
def load_image(path = './App photos/logo1.png'):
    image = Image.open(path)
    return image

logo = load_image()
st.sidebar.image(logo, use_column_width=True)


st.sidebar.header("Navigation")

content = ["The idea", "Implementation", "Scan your leaf", "Fun with quiz"]
nag = st.sidebar.radio("Go to:",content)




# function to load model:
@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('./tree1.h5')
    
    
    return model, index_to_label

index_to_label = {0: 'Alstonia Scholaris',1: 'Arjun', 2: 'Chinar', 3: 'Guava', 4: 'Jamun', 5: 'Jatropha', 6: 'Lemon', 7: 'Mango', 8: 'Pomegranate', 9: 'Pongamia Pinnata'}
if nag == 'The idea':
    #st.header("About this project:")

    st.markdown(f'<h2 style="text-align: left; font-famlily: Arial; color: #282828;">The idea </h2', 
                unsafe_allow_html=True) 
    st.markdown('''Trees are vital: they give us  oxygen, store carbon, stabilize the soil, and give life to the world’s wildlife. That’s where I got the idea of creating a plant-identification tool: **Vietree Guru**.
    
Reasons for recognizing and knowing more about trees and plants are countless, going from professional reasons to plain curiosity; arborists, landscape architects, foresters, biology students, environment experts, farmers… or simply nature enthusiasts: these are the people VietreeGuru is made for.

Obviously VietreeGuru is not the first plant-identification tool in the market, but not many plant identifiers are built to recognize trees typical of the Vietnamese ecosystem; scope of VietreeGuru is to fill in this gap, and, furthermore, provide insightful information, and create fun learning experience. After identifying the plant, VietreeGuru returns general information about the classified tree, with a link to a webpage for more detailed specifications, and generates a 10-question quiz to test user’s expertise on the subject.   


Identification, information, testing: enjoy **VietreeGuru!**
''')


    about_photo = Image.open('./App photos/photo-1557427083-363bf1d4f82d.jpg')
    st.image(about_photo, use_column_width=True)

elif nag == 'Implementation':
    m1 = Image.open('./App photos/Screen Shot 2020-10-22 at 16.10.23.png')
    m2 = Image.open('./App photos/Screen Shot 2020-10-22 at 16.10.43.png')
    m3 = Image.open('./App photos/Screen Shot 2020-10-23 at 03.05.48.png')
    st.image(m1, use_column_width=True)
    st.image(m2, use_column_width=True)
    st.image(m3, use_column_width=True)


elif nag == 'Scan your leaf':
    with st.spinner  ('Loading Model into Memory'):
        tree_model, index_to_label = load_model()

    st.markdown(f'<h2 style="text-align: left; color: #282828; font-family: Arial; ">Scan your leaf</h2>', 
    unsafe_allow_html=True) 
    st.markdown("Upload a photo of the leaf taken by your smartphone to know whose tree it is.")
    uploaded_file = st.file_uploader("Choose an image ...", type=["jpg","png","jpeg"])

    if uploaded_file is None:
        st.text("Please upload an image file")

    else: 
        image = Image.open(uploaded_file)
        st.image(image, caption='your photo', width = 400)# to show photo
    
    @st.cache()
    def import_and_predict(image, tree_model):
        
        image = ImageOps.fit(image, (224,224))
        
        img_array = np.array(image)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        prediction = np.argmax(tree_model.predict(preprocessed_img), axis = -1)
        result = index_to_label[prediction[0]]
        return result

   

    if st.button('Predict'):
        result = import_and_predict(image,tree_model)
        st.markdown(f'<h2 style="text-align: center; color: green;">This is {result}</h2>', 
    unsafe_allow_html=True) 

        look_up = {"Mango": "The mango",
                    "Alstonia Scholaris":"Alstonia scholaris", 
                    "Arjun": "Terminalia arjuna",
                    "Chinar": "Platanus orientalis",
                    "Guava": "The Guava",
                    "Jamun": "Syzygium cumini",
                    "Jatropha":"Jatropha",
                    "Lemon":"Lemon",
                    "Pomegranate": "Pomegranate",
                    "Pongamia Pinnata":"Pongamia Pinnata"
                    }

        page = wikipedia.page(look_up[result])
        summary = wikipedia.summary(look_up[result], sentences = 4)
        #summary = p.summary
        
        st.write(summary)

        st.write(f"Read full article on Wikipead about {result} on Wikipeadia: {page.url}")

        #if result:
        #if st.button("Test your knowledge")





#quiz = st.sidebar.radio("Go to", ["", "Fun with quiz"])
if nag == "Fun with quiz": 

    option = st.selectbox("Test your knowledge about:", list(index_to_label.values()), 7)

    #st.subheader(f"Test your knowledge about {option}")
    st.write("")

    
    file =  "./Text/"+ option +".txt"

    plant = pickle.load(open( file, "rb" ))

    keys, keyword_sentence_mapping_unique, key_distractor_list = plant[0], plant[1], plant[2]


    @st.cache
    def question(index):
        #random.shuffle(keys)

        key = keys[index-1]
        sentence = keyword_sentence_mapping_unique[key]
        pattern = re.compile(key, re.IGNORECASE)
        output = pattern.sub("............", sentence)
        
        
        top4choices = [key] + random.sample(key_distractor_list[key][:10], 3)
        random.shuffle(top4choices)
        
        return key, output, top4choices
        

    score = 0
    wrong = ""
    solution = {}
    random.shuffle(keys)
    for i in range(1,4):
        k,o,c = question(i)
        st.markdown(f'**Question {i}: **{o}')
        a = st.radio("Your answer",c)
        solution["Question "+ str(i)]=k
        #if a == k:
        if a.lower() == k.lower():
            score += 10
            
        else:
            wrong += "Question " + str(i) + "; "
            
    
    

    if st.button("Submit"): 
        #st.success(f'Your score: {score}')
        st.markdown(f'<h3 style="text-align: left; font-famlily: Arial; color: #006600 ;"> Score: {score}/30 </h3', 
                unsafe_allow_html=True) 

        if score == 30:
            #st.markdown(":tada: :tada: :tada: :tada: :tada:")
            st.markdown(f'<h3 style="text-align: left; font-famlily: Arial; color: #006600;"> Congratulations. You are an expert!</h3', 
                unsafe_allow_html=True) 
            
            congrats = Image.open("./App photos/Screen Shot 2020-10-22 at 23.46.22.png")
            st.image(congrats, width=300)
            st.balloons()

        else:
            st.warning(f'Wrong Answer(s): {wrong}')
            
    if st.checkbox("Show solutions"):
        
        st.write(solution)
            
        
    

st.sidebar.header("Contribute")
st.sidebar.info(
        '''This project is contributed and maintained by:   
        **Thinh Cao**: [GitHub](https://github.com/nhatthinh253) | [LinkedIn](https://linkedin.com/in/nhatthinh253)
        The source code can be found in this [Github Repo](https://github.com/nhatthinh253/Vietree_Guru).'''
        
    )
st.sidebar.header("About")
st.sidebar.info("This app is maintained by [**Thinh Cao (Jake)**](https://linkedin.com/in/nhatthinh253). You can reach me at nhatthinh253@gmail.com")
