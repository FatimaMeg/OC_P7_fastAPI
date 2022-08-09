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
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh import layouts
from bokeh.layouts import gridplot
import pandas as pd
import numpy as np


# 2. Create app and model objects
app = FastAPI()

# 3. On récupère tous les éléments liés à notre modèle de prévision : pipeline, données clients...
# On charge le pipeline
model_pipeline = joblib.load('pipeline_bank_lgbm.joblib')

# On récupère notre fichier clients de prévisions
file_clients = open("clients_test_pred.pkl", "rb")
#file_clients = open("application_test.pkl", "rb") #fichier client initial
donnees_clients = pickle.load(file_clients)
file_clients.close()

# On récupère nos features calculées par le modèle au format pkl
file_features = open("features.pkl", "rb")
features = pickle.load(file_features)
file_features.close()

# On a besoin du fichier de données 'train' pour obtenir les explanations de lime, récupéré au format pickle
file_X_train = open("X_train.pkl", "rb")
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

@app.post('/graphs')  # endpoint pour obtenir les graphiques du client
def client_graphs(client : Client):
    client_donnees = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == client.num_client, :]
    
    n_bins = 20  
    graph=[]

    j=0
    for i in features:
        j=j+1
        arr_hist = "arr_hist"+str(j)
        edges = "edges"+str(j)
        h = "h"+str(j)
        arr_hist, edges = np.histogram(donnees_clients[i], bins = n_bins, 
                                            range = [donnees_clients[i].min(),donnees_clients[i].max()])
    
        hist = pd.DataFrame({i: arr_hist, 
                            'left': edges[:-1], 
                        'right': edges[1:]})

        h = figure(plot_height = 600, plot_width = 600, 
                            title = i, 
                            x_axis_label = i, 
                            y_axis_label = 'Nombre')

        h.quad(bottom=0, top=hist[i],
                            left=hist['left'], right=hist['right'], 
                            fill_color='blue', line_color='black')
            
        graph.append(h)
    #st.bokeh_chart(gridplot([graph],width=250, height=250) )
    
    import jinja2
    from bokeh.embed import components, json_item

    template = jinja2.Template("""
    <!DOCTYPE html>
    <html lang="en-US">

    <link
        href="http://cdn.pydata.org/bokeh/dev/bokeh-0.13.0.min.css"
        rel="stylesheet" type="text/css"
    >
    <script 
        src="http://cdn.pydata.org/bokeh/dev/bokeh-0.13.0.min.js"
    ></script>

    <body>

        <h1>Hello Bokeh!</h1>
    
        <p> Below is a simple plot of stock closing prices </p>
    
        {{ script }}
    
        {{ div }}

    </body>

    </html>
    """)

    arr_hist0, edges0 = np.histogram(donnees_clients['EXT_SOURCE_2'], bins = n_bins, 
                                             range = [donnees_clients['EXT_SOURCE_2'].min(),donnees_clients['EXT_SOURCE_2'].max()])
        
    hist0 = pd.DataFrame({'EXT_SOURCE_2': arr_hist0, 
                             'left': edges[:-1], 
                            'right': edges[1:]})

    h0 = figure(plot_height = 600, plot_width = 600, 
                             title = 'EXT_SOURCE_2', 
                             x_axis_label = 'EXT_SOURCE_2', 
                             y_axis_label = 'Nombre')

    h0.quad(bottom=0, top=hist0['EXT_SOURCE_2'],
                             left=hist0['left'], right=hist0['right'], 
                             fill_color='blue', line_color='black')
    
    script, div = components(h0)

    p = figure()
    x=[1,2,3]
    y=[4,5,6]
    p.circle(x, y)

    item_text = json.dumps(json_item(p, "myplot"))

    return {
        #template.render(script=script, div=div)
        item_text
        
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
