# FastAPI application deployed with Heroku

# =========================================
# API pour l'octroi de crédits bancaires
# Author: Fatima Meguellati
# Last Modified: 08 Aout 2022
# =========================================

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Model import Features, Client
import joblib
import pickle
from lime import lime_tabular
from fastapi.encoders import jsonable_encoder
import json


# 2. Create app and model objects
app = FastAPI()

# 3. On récupère tous les éléments liés à notre modèle de prévision : pipeline, données clients...
# On charge le pipeline
model_pipeline = joblib.load('pipeline_bank_lgbm.joblib')

# On récupère notre fichier clients de prévisions
file_clients = open("fichierClient.pkl", "rb")
#file_clients = open("application_test.pkl", "rb") #fichier client initial
donnees_clients = pickle.load(file_clients)
file_clients.close()

# On récupère nos features calculées par le modèle au format pkl
file_features = open("features.pkl", "rb")
features = pickle.load(file_features)
file_features.close()

# On a besoin du fichier de données 'train' pour obtenir les explanations de lime, récupéré au format pickle
file_X_train = open("X_train_Nono2.pkl", "rb")
donnees_train = pickle.load(file_X_train)
file_X_train.close()


# 4. On décrit ici les différents endpoints de notre API

@app.post('/client')  # endpoint pour vérifier si le numéro client existe dans la base de données client
def client_recherche(client : Client):
    client_existe = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == client.num_client]
    return {
        not client_existe.empty
    }

@app.post('/predict')  # endpoint pour obtenir la prévision
def predict_clientscoring_features(client: Client):
    # On récupère les features du client
    data = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == client.num_client, features]
    prediction = model_pipeline.predict(data)[0]
    proba = model_pipeline.predict_proba(data)

    return {
        prediction, proba[0][1]
        #'Sa probabilite de faillite est de ': f"{proba[0][1] * 100:.2f} %"
    }


@app.post('/lime')
def explain_lime(client: Client):
    # On récupère les features du client
    data = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == client.num_client, features]
    explainer = lime_tabular.LimeTabularExplainer(donnees_train, mode="classification", feature_names=features)
    exp = explainer.explain_instance(data.values[0],
                                     model_pipeline.predict_proba, num_features=20)
    mongraph_html = exp.as_html()
    #Reste à faire : comment afficher les predict_proba et predicted_value dans le html, pour l'instant bug

    return {
        mongraph_html
    }

@app.post('/clientdata')  # endpoint pour obtenir les données descriptives du client
def client_recherche(client : Client):
    client_donnees = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == client.num_client, :]
    result = client_donnees.to_json(orient="records")
    parsed = json.loads(result)

    return {
        json.dumps(parsed, indent=4)
    }


@app.get("/")
async def main():
    content = """
    <body>
    <h2> Bienvenue sur l'API permettant d'obtenir des prévisions d'octroi de prêt</h2>
    <p> You can view the FastAPI UI by heading to localhost:8000 </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """
    return ("Bienvenue dans mon API avec lime")


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
