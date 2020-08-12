import re
import pandas as pd
import numpy as np

# When running this script first time
# Uncomment these two lines
import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
pst = PorterStemmer()


df = pd.read_csv('datasets/original dataset.csv')

# Removing data of videos which are deleted
deleted_videos = df[df['video_error_or_removed'] == True].index
df = df.drop(df.index[deleted_videos])

# Removing duplicated videos
df = df.drop_duplicates(subset='video_id', keep='first')


# Text Columns
def preprocess_str(col_name):
    df[col_name] = df.apply(
        lambda row: re.sub(r'[^0-9A-Za-z\s]', '', row[col_name]).lower(),
        axis=1
    )

def text_cols():
    # Video title
    df['original_title'] = df['title']
    preprocess_str('title')
    # Channel title
    df['original_channel_title'] = df['channel_title']
    preprocess_str('channel_title')

    # Description
    df['original_description'] = df['description']
    df['description'] = df['description'].fillna('')
    df['description'] = df.apply(
        lambda row: re.sub(r'[^0-9A-Za-z\s]', '', re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', row['description'])).lower(),
        axis=1
    )


# Tags
def tags_preprocess(row):
    tags = re.sub(r'"', '', re.sub(r'\|[^a-zA-Z]+\|', '|', row['tags']))
    tags = re.sub(r'[!@#$%^&*()-_+={}\[\]|\\"\';:.,<>/?`~]', ' ', tags)

    tags = word_tokenize(tags)
    final_tags = ''

    for tag in tags:
        stemed_tag = pst.stem(tag.lower())
        if stemed_tag not in final_tags and not any([word for word in stop_words if stemed_tag in word]):
            final_tags += stemed_tag + ' '

    return final_tags[:-1]

def tags():
    df.loc[df['tags'] == '[none]', ['tags']] = ''   # Some videos have no tags
    df['tags'].fillna('')
    df['tags'] = df.apply(tags_preprocess, axis=1)


# Dates
def dates():
    df['trending_date'] = df.apply(lambda row: row['trending_date'].replace('.', '-'), axis=1)  # Trending Date
    df['publish_date'] = df.apply(lambda row: row['publish_time'].split('T')[0], axis=1)        # Publish Date

# Calling functions to preprocess data
text_cols()
tags()
dates()

# Comments count
df['total_comments'] = df['comment_count']

# Rearranging Columns
# And ignoring unnecessary Columns
df = df[['title', 'video_id', 'channel_title', 'views', 'likes', 'dislikes', 'total_comments', 'description', 'tags', 'publish_date', 'trending_date', 'thumbnail_link', 'original_title', 'original_channel_title', 'original_description']]


# Saving
df.to_csv('datasets/preprocessed.csv', index=False)
# print(df)

print('Preprocessing is done!\nPreprocessed data is saved to datasets/preprocessed.csv\nNow run the server.')
