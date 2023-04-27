import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple
from enum import Enum

import util

class Chatbot:
    """Class that implements the chatbot for HW 6."""

    class ExpectedReplies(Enum):
      TITLE = 1
      SENTIMENT = 2
      CLARIFICATION = 3
      CONTINUE = 4

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'moritokary bot' # TODO: Give your chatbot a new name.

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # TODO: put any other class variables you need here
        # This enum is used in process() to guide the chatbot
        self.expectedreply = self.ExpectedReplies.TITLE

        # This array is used to temporarily store movie candidates before disambiguating them
        self.candidates = []

        # This dict stores up to 5 key-value pairs of movie_idx-sentiment for the user
        self.user_ratings = dict()

        self.response = ""

        self.current_title = ""
        self.possible_movie_idx = []
        self.current_movie_idx = -1
        self.current_sentiment = 0

        self.recommended_movies = []
        self.NUM_RECOMMENDED_MOVIES = 5

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return "Hi, I am a chatbot which makes movie recommendations based on your previous preferences. I'm a great and fun way to learn about movies and try something new!"
        """
        Your task is to implement the chatbot as detailed in the HW6
        instructions (README.md).

        To exit: write ":quit" (or press Ctrl-C to force the exit)

        TODO: Write the description for your own chatbot here in the `intro()` function.
        """

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = f"Hello! My name is {self.name}, and I'm here to help recommend movies. What do you want to do today?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "I hope I was able to help! Come back if you want more movie recommendations!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        
        self.response = ""

        line = self.fix_simple_spelling(line)
        
        def extract_movie():
            if len(self.possible_movie_idx) == 0:
                self.response += "I can't seem to find '{}'. Did you mean something else?".format(self.current_title)
                self.expectedreply = self.ExpectedReplies.TITLE
            elif len(self.possible_movie_idx) > 1:
                self.candidates = [self.titles[indices][0] for indices in self.possible_movie_idx]
                self.response += "Please be more precise. Which movie did you mean in this list: {}? ".format(self.candidates)
                self.expectedreply = self.ExpectedReplies.CLARIFICATION
            else:
                self.response += "Thank you! "
                self.current_movie_idx = self.possible_movie_idx[0]
                self.user_ratings[self.current_movie_idx] = self.current_sentiment
                if len(self.user_ratings) == 5:
                    self.recommended_movies = self.recommend_movies(self.user_ratings, self.NUM_RECOMMENDED_MOVIES)
                    self.response += "That's enough for me to make a recommendation. I suggest you watch {}. Would you like another recommendation? (Or enter :quit if you're done.)".format(self.recommended_movies[0])
                    self.recommended_movies.pop(0)
                    self.expectedreply = self.ExpectedReplies.CONTINUE
                else:
                    self.response += "I want to hear more about movies! Tell me about another movie you have seen."
                    self.expectedreply = self.ExpectedReplies.TITLE
        
        def extract_sentiment():
            # TODO: we can swap this out with predict_sentiment_rule_based too
            self.current_sentiment = self.predict_sentiment_statistical(line)
            if self.current_sentiment != 0:
                self.response += "Ok, you {sentiment} '{title}'. ".format(sentiment = "liked" if self.current_sentiment > 0 else "disliked", title=self.current_title)
            else:
                self.response += "I'm sorry, I'm not quite sure if you liked '{title}'. Tell me more about '{title}'. ".format(title=self.current_title)

        # the chatbot is expecting a title
        if self.expectedreply == self.ExpectedReplies.TITLE:

            # extract the title
            titles = self.extract_titles(line)
            if len(titles) == 0:
                # is the user expressing an emotion? if so, respond appropriately
                if (self.respond_to_emotion(line)):
                    self.response += self.respond_to_emotion(line)
                    return self.response

                self.response += "Sorry, I don't understand. Tell me about a movie that you've seen with the title in quotation marks."
                return self.response
            elif len(titles) > 1:
                self.response += "Let's focus on just one movie at a time. "
            self.current_title = titles[0]

            # extract the sentiment
            extract_sentiment()
            if self.current_sentiment == 0:
                return self.response

            # extract the movie
            # don't include the quotation marks
            self.possible_movie_idx = self.find_movies_idx_by_title(self.current_title)
            extract_movie()

        # the chatbot is expecting a sentiment
        elif self.expectedreply == self.ExpectedReplies.SENTIMENT:
            extract_sentiment()
    
        # the chatbot is expecting a clarification
        elif self.expectedreply == self.ExpectedReplies.CLARIFICATION:
            self.possible_movie_idx = self.disambiguate_candidates(line, self.possible_movie_idx)
            extract_movie()

        # the chatbot will continue spitting out recommendations if it's not "no" until it runs out
        else:
            if "no" in line:
                self.response += "Ok!"
            else:
                if self.recommended_movies:
                    self.response += "Another recommendation is '{}'. Would you like to hear another? (Or enter :quit if you're done.)".format(self.recommended_movies[0])
                    self.recommended_movies.pop(0)
                else:
                    self.response += "Sorry, but I am out of recommendations."

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return self.response

    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################  

        pattern = r'"([^"]*)"'
        return re.findall(pattern, user_input)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def find_movies_idx_by_title(self, title:str) -> list:
        """ Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think 
              of a more concise approach 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                 
        result = []
        title = title.replace('(','\(').replace(')','\)')
        pattern = f'.*{title}.*'
        for i, movie in enumerate(self.titles):
            match = re.match(pattern, movie[0])
            if match:
                result.append(i)
        return result
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def disambiguate_candidates(self, clarification:str, candidates:list) -> list: 
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                 
        
        result = []
        pattern = fr'.*{clarification}.*'
        for c in candidates:
            title = self.titles[c][0]
            match = re.search(pattern, title)
            if match:
                result.append(c)
        return result
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment. 
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count) 
        and negative sentiment category (neg_tok_count)

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                  
        pattern = r"\w+"
        tokens = re.findall(pattern, user_input)
        pos_tok_count = 0
        neg_tok_count = 0

        for tok in tokens:
            t = tok.lower()
            if t in self.sentiment:
                if self.sentiment[t] == 'pos':
                    pos_tok_count += 1
                else:
                    neg_tok_count += 1

        if pos_tok_count > neg_tok_count:
            return 1
        elif neg_tok_count > pos_tok_count:
            return -1
        else:
            return 0
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        #load training data  
        texts, y = util.load_rotten_tomatoes_dataset()

        self.model = None #variable name that will eventually be the sklearn Logistic Regression classifier you train 
        self.count_vectorizer = None #variable name will eventually be the CountVectorizer from sklearn 

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################                                                
        self.count_vectorizer = CountVectorizer(lowercase = True, stop_words='english')
        
        X_train = self.count_vectorizer.fit_transform([i.lower() for i in texts]).toarray()
        Y_train = np.array([[1 if label=="Fresh" else -1] for label in y])

        logistic_regression_classifier = sklearn.linear_model.LogisticRegression(penalty='l2')
        self.model = logistic_regression_classifier.fit(X_train, np.ravel(Y_train))
        
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        bag_of_words_representation = self.count_vectorizer.transform([user_input.lower()])
        if not bag_of_words_representation.toarray()[0].any():
            return 0
        return self.model.predict(bag_of_words_representation)[0]
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movie({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################    


        # use self.ratings, and user_ratings
        user_rating_all_movies = np.array([0 if i not in user_ratings.keys() else user_ratings[i] for i in range(len(self.titles))])

        ratings_matrix = np.array(self.ratings)
        recommendations = util.recommend(user_rating_all_movies, ratings_matrix, num_return)                                            
        return [self.titles[i][0] for i in recommendations]  # TODO: delete and replace this line
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def fix_simple_spelling(self, line: str) -> str:
        """
        This function takes a line and returns it with simple spelling mistakes fixed, if any. 
        
        Arguments:     
            - line (str) : a user-supplied line of text

        Returns: 
            - a string with the correct spelling

        Example: 
            fixed = chatbot.fix_simple_spelling('I liek "Avatar"')
            print(fixed) // prints 'I like "Avatar"'
        """
        spelling_mistakes = {
            "liek": "like",
            "lieked": "liked",
            "loev": "love",
            "loeved": "loved",
            "disliek": "dislike",
            "dislieked": "disliked",
            "graet": "great",
            "wosrt": "worst",
            "horible": "horrible",
            "terible": "terrible",
            "amazeing": "amazing",
            "fritening": "frightening",
            "romnatic": "romantic",
            "funy": "funny",
            "sacry": "scary",
            "exiting": "exciting"
        }

        return ' '.join([spelling_mistakes[word] if word in spelling_mistakes else word for word in line.split()])

    def respond_to_emotion(self, line: str) -> str:
        """
        This function takes in a line and and identifies whether the user is expressing an emotion and if so, responds to it.
        
        Arguments:
            - line (str) : a user-supplied line of text
            
        Returns:
            - a string which responds to the user's emotion 
            - None if the user does not express an emotion
            
        Example:
            response = chatbot.respond_to_emotion("I am angry")
            print(response) // prints "Oh no! I'm sorry to hear that you are angry. Maybe a movie will make you feel better?"
            
        """
        emotions = [
            "happy", 
            "sad",
            "angry",
            "upset", 
            "overjoyed",
            "calm",
            "excited"
        ]
        
        for e in emotions:
            pattern = f'(am|feel) ({e})'
            match = re.findall(pattern, line)
            if match:
                emotion = match[0][1]
                if self.sentiment[emotion] == 'pos':
                    return f"That's great! I'm happy you are {emotion}. Keep up the good vibes with a movie!"
                elif self.sentiment[emotion] == 'neg':
                    return f"Oh no! I'm sorry to hear that you are {emotion}. Maybe a movie will make you feel better?"
                else:
                    return f"I understand that you are {emotion}."
        return None

    def function3(): 
        """
        Any additional functions beyond two count towards extra credit  
        """
        pass 


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



