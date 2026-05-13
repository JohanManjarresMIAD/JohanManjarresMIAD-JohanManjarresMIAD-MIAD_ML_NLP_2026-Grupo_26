"""
Movie Genre Classification API — Proyecto 2
Patrón idéntico a Semana 3/api.py (Flask + flask-restx, artefactos joblib).

Modelo final: TF-IDF (1,2) + OneVsRest(LogisticRegression(class_weight='balanced')).
La ablación en S7P1_proyecto2.ipynb (sección 10) mostró que sentimiento y keywords
binarias no aportan AUC; el modelo final usa solo TF-IDF con los pesos de clase
balanceados.

Artefactos requeridos (misma carpeta que api.py):
  tfidf_vectorizer.pkl  — TfidfVectorizer refit sobre 100% del training
  genre_clf.pkl         — OvR LogisticRegression(class_weight='balanced')
  mlb_genres.pkl        — MultiLabelBinarizer (orden Kaggle)
  output_columns.pkl    — ['p_Action', ..., 'p_Western']

Arrancar:
  cd "Semana 7" && python api.py            # desarrollo (puerto 5000)
  cd "Semana 7" && gunicorn api:app --bind 0.0.0.0:$PORT  # producción

Ejemplo de request:
  GET /predict/?plot=A+serial+killer+teaches+his+secrets+to+a+video+store+clerk
"""

import re
import os
import joblib
from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Movie Genre Classification API',
    description='Predicts genre probabilities from a movie plot synopsis (TF-IDF + LogReg balanced).',
)

ns = api.namespace('predict', description='Genre Classifier')

# ---------------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name):
    return joblib.load(os.path.join(HERE, name))

vectorizer  = _load('tfidf_vectorizer.pkl')
classifier  = _load('genre_clf.pkl')
mlb         = _load('mlb_genres.pkl')
output_cols = _load('output_columns.pkl')

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
                    help='Título de la película (opcional, no usado en el modelo).')
parser.add_argument('year',  type=int, required=False, location='args', default=0,
                    help='Año de lanzamiento (opcional, no usado en el modelo).')

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
        return {col: float(p) for col, p in zip(output_cols, proba)}, 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
