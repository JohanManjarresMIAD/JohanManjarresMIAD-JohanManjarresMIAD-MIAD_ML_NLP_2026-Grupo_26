"""
Movie Genre Classification API — Proyecto 2
Patrón idéntico a Semana 3/api.py (Flask + flask-restx, artefactos joblib).

Artefactos requeridos (misma carpeta que api.py):
  tfidf_vectorizer_v1.pkl  — TfidfVectorizer ajustado sobre el training set completo
  genre_clf_v1.pkl         — OneVsRestClassifier(LogisticRegression) multi-etiqueta
  mlb_genres_v1.pkl        — MultiLabelBinarizer para mapear índices → nombres
  output_columns_v1.pkl    — lista de 24 columnas ['p_Action', ..., 'p_Western']

Arrancar:
  cd "Semana 7" && python api.py            # desarrollo
  cd "Semana 7" && gunicorn api:app --bind 0.0.0.0:$PORT  # producción

Ejemplo de request:
  GET /predict/?plot=A+serial+killer+teaches+his+secrets+to+a+video+store+clerk
"""

import re
import joblib
import pandas as pd
from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Movie Genre Classification API',
    description='Predicts genre probabilities from a movie plot synopsis.',
)

ns = api.namespace('predict', description='Genre Classifier')

# ---------------------------------------------------------------------------
# Carga de artefactos (igual que Proyecto 1: fallan rápido en import time)
# ---------------------------------------------------------------------------
vectorizer   = joblib.load('tfidf_vectorizer_v1.pkl')
classifier   = joblib.load('genre_clf_v1.pkl')
mlb          = joblib.load('mlb_genres_v1.pkl')
output_cols  = joblib.load('output_columns_v1.pkl')   # ['p_Action', ..., 'p_Western']

# ---------------------------------------------------------------------------
# Preprocesamiento — debe ser idéntico al clean_text del notebook
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Limpieza básica del plot (paridad exacta con el notebook)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------------------------
# Parser de argumentos
# ---------------------------------------------------------------------------
parser = ns.parser()
parser.add_argument('plot',  type=str, required=True,  location='args',
                    help='Sinopsis de la película (en inglés).')
parser.add_argument('title', type=str, required=False, location='args', default='',
                    help='Título de la película (opcional, no usado en el modelo v1).')
parser.add_argument('year',  type=int, required=False, location='args', default=0,
                    help='Año de lanzamiento (opcional, no usado en el modelo v1).')

# ---------------------------------------------------------------------------
# Schema de respuesta
# ---------------------------------------------------------------------------
genre_fields = {col: fields.Float(description=f'Probabilidad para {col[2:]}')
                for col in output_cols}
resource_fields = api.model('GenreProbabilities', genre_fields)

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@ns.route('/')
class MovieGenreApi(Resource):

    @ns.expect(parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        plot_clean = clean_text(args['plot'])

        X = vectorizer.transform([plot_clean])
        proba = classifier.predict_proba(X)[0]   # shape (24,)

        result = {col: float(p) for col, p in zip(output_cols, proba)}
        return result, 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
