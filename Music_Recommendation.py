from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import base64

TOKEN = ""
CLIENT_ID = '1494687ba5df49f4bf8d0815f018f4e1'
CLIENT_SECRET = '39468bdecca148ba89f2d5659c863b5a'

MODEL = ['tempo', 'danceability', 'loudness', 'acousticness', 'energy','time_signature']

REQUEST_HEADERS = {}

def make_authorization():
    message = f"{CLIENT_ID}:{CLIENT_SECRET}"
    message_bytes = message.encode('ascii')
    base64Bytes = base64.b64encode(message_bytes)
    base64Message = base64Bytes.decode('ascii')

    headers = {'Authorization' : f"Basic {base64Message}"}
    data = {'grant_type' : 'client_credentials'}
    url = "https://accounts.spotify.com/api/token"
    r = requests.post(url, headers=headers, data=data)
    global TOKEN
    TOKEN = r.json()['access_token']
    global REQUEST_HEADERS
    REQUEST_HEADERS = {'Content-Type' : 'applications/json', 'Authorization' : 'Bearer ' + TOKEN}

def algo(song, dataset):
    #Remove by year
    song_year = int(song['release_date'].split("-")[0])
    dataset['year'] = dataset['release_date'].apply(lambda x: int(x.split("-")[0]))
    dataset = dataset[abs(dataset['year'] - song_year) < 30]
    
    #Remove by Popularity
    song_pop = int(song['popularity'])
    dataset = dataset[dataset['popularity'] > (song_pop - 15)]
      
    #Remove by Key
    song_key = song['key']
    dataset = dataset[dataset['key'] == song_key]

    #Remove by mode
    song_mode = int(song['mode'])
    dataset = dataset[dataset['mode'] == song_mode]
    
    #Choose features
    dataset_features = dataset[MODEL]
    song_features = song[MODEL]
    print(len(dataset_features))
    #Normalize
    scaler = MinMaxScaler()
    dataset_features = pd.DataFrame(scaler.fit_transform(dataset_features))
    song_features = scaler.transform([song_features])

    #Find Closest delta E
    distances = dataset_features.sub(song_features, axis=1).pow(2).sum(axis=1).pow(0.5)
    distances = distances[distances != 0.0]
    min_distance_index = distances.argmin()
    top_3 = np.argpartition(distances, 3)
    
    #Print Song
    predicted_song = np.array(dataset)[min_distance_index]
    print(np.array(dataset)[top_3[0]] + "\n")
    print(np.array(dataset)[top_3[1]] + "\n")
    print(np.array(dataset)[top_3[2]])
    return predicted_song
          
    

def find_song(search_input):
    search_input_list = search_input.split(":")
    song_name = search_input[0].strip()
    requset = "https://api.spotify.com/v1/search?q=track:" + song_name + "&type=track,artist"

    if len(search_input_list) > 1:
        song_artist = search_input_list[1].strip()
        requset = "https://api.spotify.com/v1/search?q=track:" + song_name + "%20artist:" + song_artist + "&type=track,artist"
    
    response = requests.get(requset, headers=REQUEST_HEADERS)

    search_results = list(response.json()['tracks']['items'])

    print("Choose one of the following...")
    for i in range(len(search_results)):
        result_name = search_results[i]['name']
        result_id = search_results[i]['id']
        result_artists = ', '.join([artist['name'] for artist in search_results[i]['artists']])
        print(f"{i}) Name: {result_name} - Artists: {result_artists} - Id: {result_id}")
        
    chosen_song_index = input("")
    chosen_song_id = search_results[int(chosen_song_index)]['id']
    return chosen_song_id

def get_song_by_uri(song_uri, dataset):
    if song_uri in dataset:
        return pd.Series(dataset[dataset['uri'] == song_uri])
    else:
        request_features = "https://api.spotify.com/v1/audio-features/" + song_uri
        response_features = requests.get(request_features, headers=REQUEST_HEADERS)

        request_details = "https://api.spotify.com/v1/tracks/" + song_uri
        response_details = requests.get(request_details, headers=REQUEST_HEADERS)

        song_details = response_details.json()
        song_features = response_features.json()


        song_features['release_date'] = song_details['album']['release_date']
        song_features['popularity'] = song_details['popularity']
        song = pd.Series(song_features)
        return song




def main():
    make_authorization()
    dataset = pd.read_csv(".\Data\\tracks.csv")
    dataset = dataset.dropna()

    song_name = input("Enter song name: ")
    song_id = find_song(song_name)  
    song = get_song_by_uri(song_id,dataset)

    while True:
        predicted_song = algo(song, dataset)
        if input("Do you want to continue? (y/n)") != 'y':
            break
        song = dataset[dataset['id'] == predicted_song[0]].iloc[0]
        print(song)
if __name__ == "__main__":
    main()
