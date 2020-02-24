# Overview

![alt text](https://i0.wp.com/clay-atlas.com/wp-content/uploads/2019/08/python_nltk.png "NLTK")

In this repo you will find an API whose main functionality is:

     Store characters in a MongoDB.

     Store lines of dialogue in a MongoDB.

     Analyze the polarity and subjetivity(sentiment) of characters and group of characters with Natural Language Toolkit.

     Stablish the affinity between characters and group of characters based on the sentiment similarity anysis. 

# The API is cloud deployed in this url via heroku:

https://api-natural-language-analysis.herokuapp.com/

##### You also will find a jupyter notebook with an example in the present repository

# API end-points

### API end-points to insert information

1. Insert a name in database:
'/character_insert/<name>'

2. Insert a lines of dialogue in database:
'/conversation_insert/<house>/<name>/<line>'


### API end-points to obtain information

3. Get character id in database:
'/get_user_id/<name>'

4. Get character lines in database:
'/get_character_lines/<id_c>'

5. Get character group in database:
'/get_house_characters/<house>'

6. Get group lines of dialogue in database:
'/get_house_conversation/<house>'


### API end-points for sentiment analysis

7. Analyse de sentiment of a character:
'/character_sentiment/<name>'

8. Recommend the closest character to another character:
'/character_friend_recommender/<name>

9. Analyse de sentiment of a group:
'/house_sentiment/<house>'

10. Recommend the closest group to another group:
'/house_friend_recommender/<conversation>'

11. Recommend the closest group to a character:
'/character_house_recommender/<name>'
