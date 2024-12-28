from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Union, List
from app.features.complete_template import TripDetails
import re
from app.services.aitrip import Aitrip
templates = Jinja2Templates(directory="../frontend/public")
router = APIRouter()
class ItineraryRequest(BaseModel):
    query: str
    session_id: str
    
class LocationModel(BaseModel):
    name: str
    address: str
    is_start: bool = False
    is_end: bool = False
    lat: Optional[float] = None
    lng: Optional[float] = None

class ItineraryResponse(BaseModel):
    success: bool
    itinerary_text: Optional[str] = None
    locations: Optional[List[LocationModel]] = None
    route_info: Optional[Dict] = None
    route_services: Optional[Dict] = None
    error: Optional[str] = None

class ConversationResponse(BaseModel):
    type: str
    content: Union[Dict, Dict[str, str]]
    trip_details: Optional[TripDetails] = None
    
class RVDetails(BaseModel):
    length: Optional[float] = None
    height: Optional[float] = None
    type: Optional[str] = None

class Location(BaseModel):
    startloc: str
    endloc: str

class TripPreferences(BaseModel):
    looking_for: Optional[List[str]] = None  # Optional
    travel_dates: Optional[Dict[str, str]] = None  # Optional: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    rv_details: Optional[RVDetails] = None  # Optional RV details
    route_type: Optional[List[str]] = None  # Optional
    specific_needs: Optional[List[str]] = None  # Optional
    explore: Optional[List[Union[str, Dict[str, str]]]] = None  # Optional: predefined or custom {"custom": "custom location"}
    location: Location
    amenities: Optional[List[Dict[str, str]]] = None  # Optional
    budget: Optional[str] = None
    
class DayDetails(BaseModel):
    day: str
    content: str

class TripPlan(BaseModel):
    travel_dates: str
    travel_budget: str
    campgrounds: int
    places: int
    daily_plans: List[DayDetails]
    
def parse_itinerary_text(itinerary_text: str) -> List[DayDetails]:
    daily_plans = []
    # Regex pattern to match day sections
    day_pattern = r"Day (\d+): ([^\n]+\n.*?)(?=Day \d+:|### Budget|$)"
    
    # Find all day sections
    day_sections = re.finditer(day_pattern, itinerary_text, re.DOTALL)
    
    for match in day_sections:
        day_num = match.group(1)
        content = match.group(2).strip()
        
        daily_plan = DayDetails(
            day=f"Day {day_num}",
            content=content
        )
        daily_plans.append(daily_plan)
    
    return daily_plans

@router.get("/")
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "gmaps_api_key": request.app.state.gmaps_api_key
    })

@router.post("/init-session")
async def initialize_session(request: Request):  # Added Request parameter
    try:
        session_id = request.app.state.travel_agent.conversation_manager.initialize_session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ConversationResponse)
async def chat_endpoint(itinerary_request: ItineraryRequest, request: Request):
    if not itinerary_request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    result = await request.app.state.travel_agent.handle_conversation(
        itinerary_request.query, 
        itinerary_request.session_id
    )
    return result


# @router.on_event("startup")
# async def startup_event():
#     global travel_agent
#     travel_agent = await app.state.aitrip.initialize()
    
@router.post("/trip-plan", response_model=TripPlan)
async def create_trip_plan(preferences: TripPreferences, request: Request):
    travel_agent = request.app.state.aitrip
    # Generate itinerary
    try:
        itinerary_result = await travel_agent.generate_itinerary(preferences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating itinerary: {str(e)}")

    # If generation fails, raise an HTTPException
    if not itinerary_result.get('success'):
        raise HTTPException(status_code=500, detail=itinerary_result.get('error', 'Failed to generate itinerary'))

    # Construct TripPlan response
    daily_plans = parse_itinerary_text(itinerary_result.get('itinerary_text', ''))
        
        # Create the TripPlan response
    trip_plan = TripPlan(
        travel_dates=f"{preferences.travel_dates.get('start', 'N/A')} to {preferences.travel_dates.get('end', 'N/A')}",
        travel_budget=itinerary_result.get('estimated_budget', 'N/A'),
        campgrounds=len(itinerary_result.get('campgrounds', [])),
        places=len(itinerary_result.get('places', [])),
        daily_plans=daily_plans
    )
    
    return trip_plan