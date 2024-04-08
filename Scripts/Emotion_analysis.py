import pandas as pd

# Load the spaCy model
!python -m spacy download pt_core_news_lg # initalize the spaCy model (pt_core_news_lg for portuguese, es_core_news_lg for spanish and en_core_web_lg for english

# Import dataset
df = pd.read_excel('tweets.xlsx') #add the excel files with the tweets for each language
print(df)

# Install Pysentimiento
!pip install pysentimiento
!pip install transformers[torch]

# Implement the sentiment analysis
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="emotion", lang="pt") # initialize the analyzer ("pt" for portuguese, "es" for spanish and "en" for english)
df['Emotion'] = df.apply(lambda row: analyzer.predict(row['Text']).output, axis=1) # get the emotion for each tweet (joy, anger, disgust, sadness, surprise, fear or other)
df['Score_emotion'] = df.apply(lambda row: analyzer.predict(row['Text']).probas, axis=1) # get the emotion probability for each tweet
print(df)

# Save the results
df.to_excel("tweets_emotions.xlsx")
