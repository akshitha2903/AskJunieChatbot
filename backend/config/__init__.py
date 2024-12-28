import os
from typing import Dict, Any

def load_secrets() -> Dict[str, str]:
    # First try environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    gmaps_key = os.getenv("GOOGLEMAPS_API_KEY")
    
    # If not found, try files
    if not openai_key:
        try:
            with open("config/apikey.txt", "r") as file:
                openai_key = file.read().strip()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"OpenAI API key not found: {e}")
            
    if not gmaps_key:
        try:
            with open("config/gmapi.txt", "r") as file:
                gmaps_key = file.read().strip()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Google Maps API key not found: {e}")

    # Initialize database
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "travel_agent_db"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "akshitha2903")
    }

    return {
        "OPENAI_API_KEY": openai_key,
        "GOOGLEMAPS_API_KEY": gmaps_key,
        "DB_CONFIG": db_config
    }

import mysql.connector
def connect_to_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="akshitha2903",
        database="mydatabase"
    )
    return connection
