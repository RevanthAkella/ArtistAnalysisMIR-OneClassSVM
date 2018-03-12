import pandas as pd
import spotipy
import spotipy.util as util
token = util.prompt_for_user_token("test",client_id='5ecc5fdee7ed4093b334e1d81242fdb1',client_secret='206e5d1dba0444ac9ea0275d8ff50229',redirect_uri='http://localhost/')

sp = spotipy.Spotify(auth=token)
sp.trace=False
playlist = sp.user_playlist("revanth95", "spotify:user:revanth95:playlist:3Bw08ajfiwORxiFUIfbiLS")
songs = playlist["tracks"]["items"]
ids = []
for i in range(len(songs)):
    ids.append(songs[i]["track"]["id"])

features = sp.audio_features(ids)
for i in range(len(songs)):
    features[i]["artists"] = (songs[i]["track"]["artists"][0]["name"])
    features[i]["name"] = (songs[i]["track"]["name"])
print (features[1])
cols_to_keep =['artists','name','acousticness','danceability','energy','instrumentalness','liveness','loudness','mode','speechiness','tempo','valence','track_href']
df = pd.DataFrame(features)
df[cols_to_keep].to_csv('PinkGuy.csv',sep=';')
