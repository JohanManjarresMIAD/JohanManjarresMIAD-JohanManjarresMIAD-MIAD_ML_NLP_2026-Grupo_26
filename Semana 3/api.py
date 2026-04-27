from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Song Popularity API',
    description='Song Popularity Prediction API'
)

ns = api.namespace('predict', description='Song Popularity Regressor')

modelo = joblib.load('pop_song_regressor.pkl')
artist_means = joblib.load('artista_means.pkl')
artist_maxs  = joblib.load('artista_maxs.pkl')
album_means  = joblib.load('album_means.pkl')
album_stds   = joblib.load('album_stds.pkl')
genre_means  = joblib.load('genero_means.pkl')
genre_stds   = joblib.load('genero_stds.pkl')
global_mean  = joblib.load('global_mean.pkl')
variables_modelo    = joblib.load('variables_modelo.pkl')
generos_disponibles = joblib.load('generos_disponibles.pkl')

parser = ns.parser()
parser.add_argument('artists', type=str, required=True, location='args')
parser.add_argument('album_name', type=str, required=True, location='args')
parser.add_argument('track_genre', type=str, required=True, choices=generos_disponibles, location='args')
parser.add_argument('duration_min', type=float, required=True, location='args')
parser.add_argument('tempo', type=float, required=True, location='args')
parser.add_argument('loudness', type=float, required=True, location='args')
parser.add_argument('acousticness', type=float, required=True, location='args')
parser.add_argument('instrumentalness', type=float, required=True, location='args')
parser.add_argument('energy', type=float, required=True, location='args')
parser.add_argument('valence', type=float, required=True, location='args')
parser.add_argument('speechiness', type=float, required=True, location='args')
parser.add_argument('liveness', type=float, required=True, location='args')
parser.add_argument('danceability', type=float, required=True, location='args')
parser.add_argument('key', type=int, required=True, location='args')
parser.add_argument('mode', type=int, required=True, location='args')
parser.add_argument('time_signature', type=int, required=True, location='args')
parser.add_argument('explicit', type=str, required=True, choices=['True', 'False'], location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

def crear_df_como_entrenamiento(args):
    artists     = args['artists'].strip().lower()
    album_name  = args['album_name'].strip().lower()
    track_genre = args['track_genre'].strip().lower()

    artist_pop_mean = artist_means.get(artists, global_mean)
    artist_pop_max  = artist_maxs.get(artists, global_mean)
    album_pop_mean  = album_means.get(album_name, artist_pop_mean)
    album_pop_std   = album_stds.get(album_name, 0)
    genre_pop_mean  = genre_means.get(track_genre, global_mean)
    genre_pop_std   = genre_stds.get(track_genre, 0)

    row = {
        'explicit':         args['explicit'],
        'danceability':     args['danceability'],
        'energy':           args['energy'],
        'key':              str(args['key']),
        'loudness':         args['loudness'],
        'mode':             str(args['mode']),
        'speechiness':      args['speechiness'],
        'acousticness':     args['acousticness'],
        'instrumentalness': args['instrumentalness'],
        'liveness':         args['liveness'],
        'valence':          args['valence'],
        'tempo':            args['tempo'],
        'time_signature':   str(args['time_signature']),
        'duration_min':     args['duration_min'],
        'artist_pop_mean':  artist_pop_mean,
        'artist_pop_max':   artist_pop_max,
        'album_pop_mean':   album_pop_mean,
        'album_pop_std':    album_pop_std,
        'genre_pop_mean':   genre_pop_mean,
        'genre_pop_std':    genre_pop_std,
    }

    return pd.DataFrame([row], columns=variables_modelo)

@ns.route('/')
class SongPopularityApi(Resource):

    @ns.expect(parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        df_ingresado = crear_df_como_entrenamiento(args)

        prediccion = modelo.predict(df_ingresado)[0]
        return {'result': float(prediccion)}, 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)