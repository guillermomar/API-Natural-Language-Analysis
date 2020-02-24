from flask import Flask, request
from pymongo import MongoClient, InsertOne
import requests
from bson.json_util import dumps

from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity as distance
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob

import pandas as pd
import numpy as np

import os
from config import dbURL

### Conection to MongoDB
client = MongoClient(dbURL)
db = client["GoT"]
collection_cha = db["characters"]
collection_con = db["conversations"]


### Flask end-points
import random
app = Flask(__name__)



@app.route('/character_insert/<name>') ### this function pretend to insert a new document in our MongoDB
def character_insert(name): ### It requires a name to create in our collection the character document

        id = 0
        lastId = list(collection_cha.find({},{'_id':1}).sort('$natural',-1).limit(1))    ### as you can see here, we dont want the extra large ids that Mongo insert by default,
        if len(lastId) > 0:                                                              ### so we will create our auto-increment id. In addition, if the query dont detect any id
            for e in lastId:                                                             ### it will assign the id = 1
                for _,v in e.items():
                    id = v + 1
            
                    collection_cha.insert_one( {
                    "_id": id,
                    "c_name": name} )

        else:
            id = 1
            collection_cha.insert_one( {
            "_id": id,
            "c_name": name} )



@app.route('/conversation_insert/<house>/<name>/<line>') ### this function pretend to insert a new document in our MongoDB
def conversation_insert(house,name,line):                ### It requires a house,name and line text to create in our second collection the conversation document
    house = str(house)
    print(house)
    name = str(name)
    line = str(line)

    
    id = 0    ### In the same way as before, it will assign auto incremental ids to every document.
    lastId = list(collection_con.find({},{'_id':1}).sort('$natural',-1).limit(1))   
    if len(lastId) > 0: 
        for e in lastId:                                                          
            for _,v in e.items():
                id = v + 1

                c_id_list = list(collection_cha.find({'c_name':name},{'_id':1}))
                id_c = [value for dictionary in c_id_list for key,value in dictionary.items()][0] ### Here we search for character id which it is in chracters collection 
                                                                                                  ### to assign it to the conversation
                collection_con.insert_one( {          ### finally we create the document with all this info.
                                "_id": id,            ### as you can see we will have a document for line in the show
                                "house": house,
                                "id_c": id_c,
                                "c_name": name,
                                "line": line
                                } )
    else:
        id = 1
        c_id_list = list(collection_cha.find({'c_name':name},{'_id':1}))
        id_c = [value for dictionary in c_id_list for key,value in dictionary.items()][0]

        collection_con.insert_one( {
                        "_id": id,
                        "house": house,
                        "id_c": id_c,
                        "c_name": name,
                        "line": line
                        } )





@app.route('/get_user_id/<name>') ### this function pretend to retrieve a the character id in our MongoDB
def get_user_id(name):
  match =  list(collection_cha.find({"c_name":name},{"_id":1,"c_name":1}))
  return match[0]


@app.route('/get_character_lines/<id_c>')  ### this function pretend to retrieve all the lines for a character id in our MongoDB
def get_character_lines(id_c):
    id_c = int(id_c)
    match = collection_con.find({"id_c":id_c},{"_id":1,"line":1})
    return dumps(match)

@app.route('/get_house_conversation/<house>') ### this function pretend to retrieve all the lines for a house in our MongoDB
def get_house_conversation(house):
    match = collection_con.find({"house":house},{"_id":1,"c_name":1,"line":1})
    return dumps(match)



@app.route('/get_house_characters/<house>') ### this function pretend to retrieve all the characters that belongs to a house in our MongoDB
def get_house_characters(house):
    match = collection_con.find({"house":house}).distinct("c_name")
    return dumps(match)



@app.route('/character_friend_recommender/<name>') ### this function will analyze all messages from all characters and tell
def character_friend_recommender(name):            ### us who is the recommended friend for our character

    characters = list(collection_con.find({}).distinct("c_name"))

    sentiment_text = {} ### we start the anlysis of sentiment

    for character in characters:
        lines = []
        text= ""
        text_clean = ""

        match = list(collection_con.find({"c_name":character})) ### we get all the info of the character

        for dictionary in match:   ### then we make a dictionary with his lines
            lines.append(dictionary["line"])

        for line in lines: ### after that, we get a string with all the text
            text += line
            
            # removing symbols from the text to improve the analysis.
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens_clean = [e for e in words if e not in stop_words]
        
        for word in tokens_clean:
            text_clean += word
                        
        sentiment_text[character] = text_clean  ### now we have a dictionary with character:text

    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(sentiment_text.values())


    doc_term_matrix = sparse_matrix.todense()
    df_sentiment = pd.DataFrame(doc_term_matrix, ### we create our data frame
                    columns=count_vectorizer.get_feature_names(), 
                    index=sentiment_text.keys())

    similarity_matrix = distance(df_sentiment,df_sentiment)

    sim_df = pd.DataFrame(similarity_matrix, columns=sentiment_text.keys(), index=sentiment_text.keys())

    np.fill_diagonal(sim_df.values, 0) # Remove diagonal max values and set those to 0

    sim_df_idmax = pd.DataFrame(sim_df.idxmax())  ### now we have the similarity matrix and we can proceed to get the recommended friend
    return (f"The recommended friend for {name} is:" + " " + np.asarray(sim_df_idmax.loc[name])[0])


@app.route('/character_sentiment/<name>') ### this function will analyze the sentiment from a character
def character_sentiment(name):

    lines = []
    sentiments = []
    polarity = []
    subjetivity = []
    match = list(collection_con.find({"c_name":name}))

    for dictionary in match:
        lines.append(dictionary["line"])


    for line in lines:
    
            sentiments.append(TextBlob(line).sentiment) ### here we use textblob to get the sentiment of all the character lines

            
    for sentiment in sentiments:  ### now we append  the values to a list so we can get the mean
        if sentiment[0] != 0.0 or sentiment[1] != 0.0:
            polarity.append(sentiment[0])
            subjetivity.append(sentiment[1])
    
    single_line_analysis_list = []
    zipped = list(zip(lines,sentiments))
    for single_line_analysis in zipped:
        single_line_analysis_list.append(single_line_analysis)

    ### finally we have the mean of polarity and subjetivity and all the anlysis for every line

    return \
    (f"The polarity of {name} is {np.mean(polarity)}, and his subjetivity is {np.mean(subjetivity)}") + \
    (f"                                                                      ") + \
    (f"Also, if you want to check the stats line by line take a look over here:") + \
    (f"                                                                      ") + \
    (f"{single_line_analysis_list}")


@app.route('/house_sentiment/<house>')  ### house version of character sentiment
def house_sentiment(house):

    lines = []
    sentiments = []
    polarity = []
    subjetivity = []
    match = list(collection_con.find({"house":house}))

    for dictionary in match:
        lines.append(dictionary["line"])


    for line in lines:
    
            sentiments.append(TextBlob(line).sentiment)

            
    for sentiment in sentiments:
        if sentiment[0] != 0.0 or sentiment[1] != 0.0:
            polarity.append(sentiment[0])
            subjetivity.append(sentiment[1])
    
    single_line_analysis_list = []
    zipped = list(zip(lines,sentiments))
    for single_line_analysis in zipped:
        single_line_analysis_list.append(single_line_analysis)


    return \
    (f"The polarity of {house} is {np.mean(polarity)}, and his subjetivity is {np.mean(subjetivity)}") + \
    (f"                                                                      ") + \
    (f"Also, if you want to check the stats line by line take a look over here:") + \
    (f"                                                                      ") + \
    (f"{single_line_analysis_list}")



@app.route('/house_friend_recommender/<conversation>')  ### house version of character recommender
def house_friend_recommender(conversation):

    houses = list(collection_con.find({}).distinct("house"))

    sentiment_text = {}

    for house in houses:
        lines = []
        text= ""
        text_clean = ""

        match = list(collection_con.find({"house":house}))

        for dictionary in match:
            lines.append(dictionary["line"])

        for line in lines:
            text += line
            
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens_clean = [e for e in words if e not in stop_words]
        
        for word in tokens_clean:
            text_clean += word
                        
        sentiment_text[house] = text_clean

    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(sentiment_text.values())


    doc_term_matrix = sparse_matrix.todense()
    df_sentiment = pd.DataFrame(doc_term_matrix, 
                    columns=count_vectorizer.get_feature_names(), 
                    index=sentiment_text.keys())

    similarity_matrix = distance(df_sentiment,df_sentiment)

    sim_df = pd.DataFrame(similarity_matrix, columns=sentiment_text.keys(), index=sentiment_text.keys())

    np.fill_diagonal(sim_df.values, 0) 

    sim_df_idmax = pd.DataFrame(sim_df.idxmax())
    return (f"The recommended house for {conversation} is:" + " " + np.asarray(sim_df_idmax.loc[conversation])[0])


@app.route('/character_house_recommender/<name>')  ### here we will recommend a house to a character! 
def character_house_recommender(name):             ### it is similar to what we have worked until now
    houses = list(collection_con.find({}).distinct("house"))


    sentiment_text = {}

    for house in houses:
        lines = []
        text= ""
        text_clean = ""

        match = list(collection_con.find({"house":house}))

        for dictionary in match:               ### the main difference is that we have to remove from the house the lines of our character to avoid always 
            if dictionary["c_name"] != name:   ### recommending his own house
                lines.append(dictionary["line"])
                
        for line in lines:
            text += line
            
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens_clean = [e for e in words if e not in stop_words]
        
        for word in tokens_clean:
            text_clean += word
                        
        sentiment_text[house] = text_clean
        
    
    characert_lines = []  ### here we add our character and his lines to our dictionary
    character_text = ""
    character_text_clean = ""
    match = list(collection_con.find({"c_name":name}))
    
    for dictionary in match:
        if dictionary["c_name"] == name:
            characert_lines.append(dictionary["line"])

    for line in characert_lines:
        character_text += line

    character_words = nltk.word_tokenize(character_text)
    stop_words = set(stopwords.words('english'))
    tokens_clean_character = [e for e in character_words if e not in stop_words]
    for word in tokens_clean_character:
        character_text_clean += word
    
    sentiment_text[name] = character_text_clean
    

    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(sentiment_text.values())


    doc_term_matrix = sparse_matrix.todense()
    df_sentiment = pd.DataFrame(doc_term_matrix, 
                    columns=count_vectorizer.get_feature_names(), 
                    index=sentiment_text.keys())

    similarity_matrix = distance(df_sentiment,df_sentiment)

    sim_df = pd.DataFrame(similarity_matrix, columns=sentiment_text.keys(), index=sentiment_text.keys())

    np.fill_diagonal(sim_df.values, 0) 

    sim_df_idmax = pd.DataFrame(sim_df.idxmax())
    return (f"The recomended house for {name} is:" + " " + np.asarray(sim_df_idmax.loc[name])[0])




app.run("0.0.0.0", os.getenv("PORT"), debug=True)

