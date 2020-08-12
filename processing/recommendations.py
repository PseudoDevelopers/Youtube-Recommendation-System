import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class recommender:
    def __init__(self):
        self.df = pd.read_csv('datasets/preprocessed.csv')

        self.cos_simi = self.train_model(self.df['tags'].values.astype('U'))

        
    def get_recommended_videos(self, requested_video_id):
        requested_video = self.df.loc[self.df['video_id'] == requested_video_id].to_dict(orient='records')


        if len(requested_video) == 0:
            return {
                'requested_video': None,
                'channel_recommended_videos': None,
                'other_recommended_videos': None
            }
        

        # Splitting the tags
        requested_video = requested_video[0]
        requested_video['tags'] = requested_video['tags'].split(' ')

        # Now we find similar videos
        all_videos = self.search_recommended_videos(self.df, requested_video_id)

        all_sorted_videos = self.sort_remove_videos(all_videos)

        if len(all_sorted_videos) == 0:
            return {
                'requested_video': requested_video,
                'channel_recommended_videos': None,
                'other_recommended_videos': None
            }


        video_indexes = [video[0] for video in all_sorted_videos]
        recommended_videos = self.df.iloc[video_indexes]

        channel_recommended_videos = recommended_videos.loc[recommended_videos['channel_title'] == requested_video['channel_title']]
        other_recommended_videos = recommended_videos.loc[recommended_videos['channel_title'] != requested_video['channel_title']]

        # Getting top videos
        channel_recommended_videos = channel_recommended_videos.head(15)
        other_recommended_videos = other_recommended_videos.head(30)

        channel_recommended_videos = channel_recommended_videos.to_dict(orient='records') if len(channel_recommended_videos) > 0 else None
        other_recommended_videos = other_recommended_videos.to_dict(orient='records') if len(other_recommended_videos) > 0 else None
        
        return {
            'requested_video': requested_video,
            'channel_recommended_videos': channel_recommended_videos,
            'other_recommended_videos': other_recommended_videos
        }

    def search_videos(self, query):
        new_df = self.df.append({ 'tags': query }, ignore_index=True)

        cos_simi = self.train_model(new_df['tags'].values.astype('U'))
        
        query_index = len(new_df) - 1
        all_videos = list(enumerate(cos_simi[query_index]))

        all_sorted_videos = self.sort_remove_videos(all_videos)[1:]

        if len(all_sorted_videos) == 0:
            return None

        video_indexes = [video[0] for video in all_sorted_videos]
        recommended_videos = new_df.iloc[video_indexes]
        
        return recommended_videos.to_dict(orient='records')

    def search_recommended_videos(self, df, requested_video_id):
        requested_video_index = df.index[df['video_id'] == requested_video_id][0]
        all_videos = list(enumerate(self.cos_simi[requested_video_index]))

        return all_videos

    def train_model(self, column):
        cv = CountVectorizer()


        count_matrix = cv.fit_transform(column)
        return cosine_similarity(count_matrix)

    def sort_remove_videos(self, df):
        all_sorted_videos = sorted(df, key=lambda tup: tup[1], reverse=True)    # Sorting videos
        all_sorted_videos = [video for video in all_sorted_videos if video[1] > 0.0]    # Removing non matched videos

        return all_sorted_videos

    def get_random_videos(self):
        return self.df.sample(n=50, replace=False).to_dict(orient='records')
