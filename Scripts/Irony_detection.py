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
analyzer = create_analyzer(task="irony", lang="pt") # initialize the analyzer ("pt" for portuguese, "es" for spanish and "en" for english)
df['Irony'] = df.apply(lambda row: analyzer.predict(row['Text']).output, axis=1) # get the irony detection for each tweet (ironic or not ironic)
df['Score_irony'] = df.apply(lambda row: analyzer.predict(row['Text']).probas, axis=1) # get the irony detection probability for each tweet
print(df)

# Save the results
df.to_excel("tweets_irony.xlsx")
