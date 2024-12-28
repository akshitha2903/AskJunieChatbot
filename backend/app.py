from app.services.travel_agent import TravelAgent
from app import create_app
import uvicorn
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=5000
    )
