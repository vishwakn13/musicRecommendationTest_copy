import pandas as pd
import spotipy
import spotipy.util as util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

spotify_data = pd.read_csv('SpotifyFeatures copy.csv')
spotify_data.head()

spotify_features_df = spotify_data
genre_OHE = pd.get_dummies(spotify_features_df.genre)
key_OHE = pd.get_dummies(spotify_features_df.key)

scaled_features = MinMaxScaler().fit_transform([
    spotify_features_df['acousticness'].values,
    spotify_features_df['danceability'].values,
    spotify_features_df['duration_ms'].values,
    spotify_features_df['energy'].values,
    spotify_features_df['instrumentalness'].values,
    spotify_features_df['liveness'].values,
    spotify_features_df['loudness'].values,
    spotify_features_df['speechiness'].values,
    spotify_features_df['tempo'].values,
    spotify_features_df['valence'].values,
])


scope = 'user-library-read'
# noinspection PyDeprecation
token = util.prompt_for_user_token(scope, client_id=client_id, client_secret=client_secret,
                                   redirect_uri='http://localhost/')

sp = spotipy.Spotify(auth=token)
playlist_dic = {}
playlist_cover_art = {}

for i in sp.current_user_playlists()['items']:
    playlist_dic[i['name']] = i['uri'].split(':')[2]
    playlist_cover_art[i['uri'].split(':')[2]] = i['images'][0]['url']


# print(playlist_dic)


def generate_playlist_df(playlist_name, playlist_dic, spotify_data):
    playlist = pd.DataFrame()

    for i, j in enumerate(sp.playlist(playlist_dic[playlist_name])['tracks']['items']):
        playlist.loc[i, 'artist'] = j['track']['artists'][0]['name']
        playlist.loc[i, 'track_name'] = j['track']['name']
        playlist.loc[i, 'track_id'] = j['track']['id']
        playlist.loc[i, 'url'] = j['track']['album']['images'][1]['url']
        playlist.loc[i, 'date_added'] = j['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])

    playlist = playlist[playlist['track_id'].isin(spotify_data['track_id'].values)].sort_values('date_added',
                                                                                                ascending=False)

    return playlist


# choosing playlist to get input from
playlist_df = generate_playlist_df('test', playlist_dic, spotify_data)

# print(playlist_df.head())


'''def visualize_cover_art(playlist_df):
    temp = playlist_df['url'].values
    plt.figure(figsize=(15, int(0.625 * len(temp))), facecolor='#8cfc03')
    columns = 5

    for i, url in enumerate(temp):
        plt.subplot(int(len(temp) / columns + 1), columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        s = ''
        plt.xlabel(s.join(playlist_df['track_name'].values[i].split('  ')[:4]), fontsize=10, fontweight='bold')
        plt.tight_layout(h_pad=0.8, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()'''

# visualize_cover_art(playlist_df)

playlist_df.to_csv('playlist.csv')

spotify_features_df = spotify_data

genre_OHE = pd.get_dummies(spotify_features_df.genre)
# print(genre_OHE.head())

key_OHE = pd.get_dummies(spotify_features_df.key)
key_OHE.head()

spotify_features_df[
    ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness',
     'tempo', 'valence']] = scaled_features.T

spotify_features_df = spotify_features_df.drop('genre', axis=1)
spotify_features_df = spotify_features_df.drop('artist_name', axis=1)
spotify_features_df = spotify_features_df.drop('track_name', axis=1)
spotify_features_df = spotify_features_df.drop('popularity', axis=1)
spotify_features_df = spotify_features_df.drop('key', axis=1)
spotify_features_df = spotify_features_df.drop('mode', axis=1)
spotify_features_df = spotify_features_df.drop('time_signature', axis=1)

spotify_features_df = spotify_features_df.join(genre_OHE)
spotify_features_df = spotify_features_df.join(key_OHE)


# print(spotify_features_df.shape)

# print(spotify_features_df.head())

def generate_playlist_vector(spotify_features, playlist_df, weight_factor):
    spotify_features_playlist = spotify_features[spotify_features['track_id'].isin(playlist_df['track_id'].values)]
    spotify_features_playlist = spotify_features_playlist.merge(playlist_df[['track_id', 'date_added']], on='track_id',
                                                                how='inner')

    spotify_features_nonplaylist = spotify_features[~spotify_features['track_id'].isin(playlist_df['track_id'].values)]

    playlist_feature_set = spotify_features_playlist.sort_values('date_added', ascending=False)

    most_recent_date = playlist_feature_set.iloc[0, -1]

    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix, 'days_from_recent'] = int(
            (most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days)

    playlist_feature_set['weight'] = playlist_feature_set['days_from_recent'].apply(lambda x: weight_factor ** (-x))

    playlist_feature_set_weighted = playlist_feature_set.copy()

    playlist_feature_set_weighted.update(
        playlist_feature_set_weighted.iloc[:, :-3].mul(playlist_feature_set_weighted.weight.astype(int), 0))

    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-3]

    return playlist_feature_set_weighted_final.sum(axis=0), spotify_features_nonplaylist


playlist_vector, nonplaylist_df = generate_playlist_vector(spotify_features_df, playlist_df, 1.2)


# print(playlist_vector.shape)

def generate_recommendation(spotify_data, playlist_vector, nonplaylist_df):
    non_playlist = spotify_data[spotify_data['track_id'].isin(nonplaylist_df['track_id'].values)].copy()
    non_playlist.loc[:, 'sim'] = cosine_similarity(nonplaylist_df.drop(['track_id'], axis=1).values,
                                                   playlist_vector.drop(labels='track_id').values.reshape(1, -1))[:, 0]
    non_playlist_top15 = non_playlist.sort_values('sim', ascending=False).head()
    non_playlist_top15['url'] = non_playlist_top15['track_id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])

    return non_playlist_top15


top15 = generate_recommendation(spotify_data, playlist_vector, nonplaylist_df)

# print(top15.head())

# visualize_cover_art(top15)

# print(top15[['track_name']])

for index, row in top15.iterrows():
    print()
    print(f"Track: {row['track_name']}")
    print(f"Artist: {row['artist_name']}")
    # print(f"Genre: {row['genre']}")
    # print(f"Popularity: {row['popularity']}")
    print(f"Similarity: {row['sim']}")
