#Procfile pour Heroku deployment
web : sh setup.sh && streamlit run streamlit_app.py && uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT