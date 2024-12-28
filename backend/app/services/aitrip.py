import os
import random
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import asyncio
from config import load_secrets
# Models (keep existing models)
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
    budget: Optional[str] = None  # Optional: Cheap, balanced, luxury, flexible
    
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

app = FastAPI()
secrets = load_secrets()

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

# def initialize_session(self) -> str:
#         """
#         Initializes a new session and returns the unique session_id.
#         """
#         session_id = str(uuid.uuid4())
#         self.conversations[session_id] = []
#         self.trip_details[session_id] = TripDetails()
#         return session_id

# @app.post("/init-session")
# async def initialize_session(request: Request):  # Added Request parameter
#     try:
#         session_id = request.app.state.travel_agent.conversation_manager.initialize_session()
#         return {"session_id": session_id}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
import re
import logging
from app.features.complete_template import ItineraryTemplate
from app.features.location_template import Location, EnhancedLocationParser, RouteServices
from app.services.data_integration import CampgroundSearchService

class Aitrip():
    def __init__(self):
        # Initialize a logger for the TravelAgent class
        self.logger = logging.getLogger("TravelAgent")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=secrets['OPENAI_API_KEY'])
        self.location_parser = EnhancedLocationParser(self.chat_model)
        self.itinerary_template = ItineraryTemplate
        self.itinerary_chain = None
        self._campground_service = None
        self._google_maps_api_key = secrets['GOOGLEMAPS_API_KEY']
    
    @property
    def campground_service(self):
        """
        Lazy loading property for CampgroundSearchService.
        Only initializes when first accessed.
        """
        if self._campground_service is None:
            self.logger.info("Initializing CampgroundSearchService...")
            self._campground_service = CampgroundSearchService(openai_api_key=secrets['OPENAI_API_KEY'])
        return self._campground_service
    @property
    def route_services(self):
        return RouteServices(self._google_maps_api_key)
    async def _setup_itinerary_chain(self):
        """Async setup of the itinerary chain"""
        chain = LLMChain(
            llm=self.chat_model,
            prompt=ChatPromptTemplate.from_template(self.itinerary_template)
        )
        return chain
    async def initialize(self):
        """Async initialization method"""
        self.itinerary_chain = await self._setup_itinerary_chain()
        return self
    

    async def create_itinerary(self, campground_names: List[str], preferences_dict: Dict, locations: List[str]) -> str:
            """
            Create a detailed day-to-day itinerary based on preferences, locations, and campgrounds.
            Args:
                campgrounds (List[str]): List of campground names.
                preferences_dict (Dict): Preferences for the trip.
                locations (List[str]): Locations extracted from the itinerary.
            
            Returns:
                str: A detailed day-to-day itinerary.
            """
            input_variables = {
                "preferences": f"""### Trip Preferences:
            - Travel Dates: {preferences_dict.get('travel_dates', 'Not specified')}
            - Location: {preferences_dict.get('location', 'Not specified')}
            - Budget: {preferences_dict.get('budget', 'Flexible')}
            - RV Details: {preferences_dict.get('rv_details', 'None')}
            - Route Type: {preferences_dict.get('route_type', 'Flexible')}
            - Specific Needs: {', '.join(preferences_dict.get('specific_needs', [])) if preferences_dict.get('specific_needs') else 'None'}
            - Explore: {preferences_dict.get('explore', 'Not specified')}
            - Amenities: {', '.join([f"{key}: {value}" for amenity in preferences_dict.get('amenities', []) for key, value in amenity.items()]) if preferences_dict.get('amenities') else 'Not specified'}""",
                "locations": ', '.join(locations),
                "campgrounds": ', '.join(campground_names)
            }
            
            template = """
            You are an expert travel planner tasked with creating a personalized day-to-day itinerary for a trip.
            The trip details are as follows:
            
            {preferences}
            Take the start and end location into account when creating the itinerary.
            
            ### These are Mandatory Locations to Cover in the trip:
            {locations}
            
            ### Mandatory Campgrounds to Stay at:
            {campgrounds}
            
            ### Nomad Rules:
            - Drive no longer than 3 hours per day.
            - Stay at each location for at least 3 nights.
            - Arrive at each destination by 3 PM local time.

            ### Instructions:
            1. Create an engaging, day-by-day itinerary starting from the first location and ending at the end location. DO NOT CREATE A ROUND TRIP; IT SHOULD BE ONE-WAY.
            2. For each day, include:
            - Morning, afternoon, and evening activities, focusing on attractions, landmarks, hikes, cultural spots, or scenic views at or near the specified locations.
            - Use campgrounds strictly for stays and rest, not as attractions or activities.
            - Highlight any unique or memorable experiences at the attractions for that day.
            3. Ensure the itinerary aligns with the budget, travel dates, and preferences provided.
            4. Adhere to the following rules for stays:
            - Use the provided campgrounds for overnight stays based on proximity to the day's ending location.
            5. Ensure all activities ALIGN WITH THE NOMAD RULES
            6. Finally at the end of the itinerary include the complete budget for the trip. STRICTLY FOLLOW THIS FORMAT like this example: "Budget: $1000" (IT SHOULD BE PROPER ESTIMATED NUMBERS )

            ### Budget Breakdown Requirements:
            - Itemize costs for:
                1. Campground fees (per night)
                2. Activity costs
                3. Fuel estimates
                4. Food and dining
                5. Miscellaneous expenses
            - Provide subtotal for each category
            - Include final total budget
            
            ### Additional Guidelines:
            1. MUST use every listed campground at least once
            2. MUST visit every listed location
            3. Include travel time estimates between locations
            4. Consider seasonal activities and weather
            5. Account for RV-specific needs and limitations
            6. Incorporate user preferences into activity selection
            
            Create a detailed itinerary following all above requirements, ensuring a logical flow between locations while maintaining the nomad rules.
            
            Now, create a custom itinerary.
            """
            chain = LLMChain(
                llm=self.chat_model,
                prompt=ChatPromptTemplate.from_template(template)
            )
            itinerary_output = await chain.arun(input_variables)
            budget_extraction_prompt = f"""
            Extract the 'Total Budget' from the following itinerary. If the budget is not specified, respond with 'Budget not specified' else just the budget. For example, if the budget is $1000, respond with '$1000'.
            Itinerary:
            {itinerary_output}
            """
            self.logger.info(f"Extracting budget from itinerary: {itinerary_output}")
            # Query the LLM for budget extraction
            budget_chain = LLMChain(
                llm=self.chat_model,
                prompt=ChatPromptTemplate.from_template(budget_extraction_prompt)
            )
            budget_output = await budget_chain.arun({})
            estimated_budget = budget_output.strip()
            self.logger.info(f"Estimated Budget: {estimated_budget}")
            return itinerary_output, estimated_budget
    async def generate_itinerary(self, preferences: TripPreferences):
        """Generate a complete itinerary based on collected trip preferences"""
        try:
            # Convert preferences into a format suitable for the itinerary generation logic
            
            self.logger.info("Starting itinerary generation...")
            
            # Run the itinerary generation chain using preferences
            itinerary_text = self.itinerary_chain.run(trip_details=preferences)
            # Parse locations from the generated itinerary
            self.logger.info("Initial itinerary generated...")
            locations = self.location_parser.extract_locations(itinerary_text)
            self.logger.info(f"Locations extracted from itinerary")
            geocoded_locations = self.route_services.geocode_locations(locations)
            self.logger.info(f"Geocoded Locations got")
            campgrounds = []
            self.logger.info(f"Searching for campgrounds...")
            for loc in geocoded_locations:
                result = await self.campground_service.search_campgrounds(preferences, loc)
                campgrounds.append(result)
            campground_names = set([
                campground['name'] 
                for sublist in campgrounds
                for campground in sublist 
                if 'name' in campground
            ])
            self.logger.info(f"Campgrounds: {campground_names}")
            preferences_dict = preferences.dict()
            self.logger.info(f"Preferences: {preferences_dict}")
            addresses = [loc.address for loc in geocoded_locations]
            print(addresses)
            # Enhance the itinerary with additional attractions based on locations
            enhanced_itinerary, estimated_budget = await self.create_itinerary(campground_names, preferences_dict, addresses)
            self.logger.info(f"Enhanced Itinerary and budget generated!")
            # Add geocoding data to the locations
            locations = self.route_services.geocode_locations(locations)

            # Get detailed route information based on geocoded locations
            route_info = self.route_services.get_route_info(locations)

            # Find nearby services (e.g., gas stations, rest areas) along the route
            route_services = {
                "services": self.route_services.find_nearby_services(locations)
            }

            # Construct the final response
            self.logger.info("Itinerary generation completed successfully.")
            return {
                "success": True,
                "itinerary_text": enhanced_itinerary,
                "estimated_budget": estimated_budget,
                "campgrounds": campground_names,
                "places": addresses,
                "locations": [vars(loc) for loc in locations],  # Convert location objects to dictionaries
                "route_info": route_info,
                "route_services": route_services
            }

        except Exception as e:
            # Log the error and return a failure response
            self.logger.error(f"Error generating itinerary: {e}")
            return {
                "success": False,
                "error": "Failed to generate itinerary. Please try again."
            }

# travel_agent = None

# @app.on_event("startup")
# async def startup_event():
#     global travel_agent
#     travel_agent = await Aitrip().initialize()

# class TripPlanResponse(BaseModel):
#     plan: TripPlan
#     campgrounds: List[str]
#     places: List[str]
# @app.post("/trip-plan", response_model=TripPlan)
# async def create_trip_plan(preferences: TripPreferences):
#     # Generate itinerary
#     try:
#         itinerary_result = await travel_agent.generate_itinerary(preferences)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating itinerary: {str(e)}")

#     # If generation fails, raise an HTTPException
#     if not itinerary_result.get('success'):
#         raise HTTPException(status_code=500, detail=itinerary_result.get('error', 'Failed to generate itinerary'))

#     # Construct TripPlan response
#     daily_plans = parse_itinerary_text(itinerary_result.get('itinerary_text', ''))
        
#         # Create the TripPlan response
#     trip_plan = TripPlan(
#         travel_dates=f"{preferences.travel_dates.get('start', 'N/A')} to {preferences.travel_dates.get('end', 'N/A')}",
#         travel_budget=itinerary_result.get('estimated_budget', 'N/A'),
#         campgrounds=len(itinerary_result.get('campgrounds', [])),
#         places=len(itinerary_result.get('places', [])),
#         daily_plans=daily_plans
#     )
    
#     return trip_plan
#     # return TripPlanResponse(
#     #     plan=trip_plan,
#     #     campgrounds=itinerary_result.get('campgrounds', []),
#     #     places=itinerary_result.get('places', [])
#     # )
# @app.get("/")
# async def root():
#     return {
#         "message": "Welcome to the Travel Planning API",
#         "endpoints": {
#             "root": "/",
#             "trip_plan": "/trip-plan",
#             "docs": "/docs",
#             "redoc": "/redoc"
#         },
#         "status": "active"
#     }
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(
#         "aitrip:app",  # Change this to match your file name
#         host="127.0.0.1",
#         port=8000,
#         reload=True  # Enable auto-reload during development
#     )
#Format of the response should be like this:
# {
#         "rvType": {
#             "type": "travelTrailer",
#             "length": 30,
#             "width": 8
#         },
#         "dateRange": {
#             "step": 2,
#             "startDate": "2024-12-20T00:00:00.000Z",
#             "endDate": "2024-12-30T00:00:00.000Z"
#         },
#         "desiredAmenities": {
#             "comfort": {
#                 "privateShower": true,
#                 "airConditioning": true
#             },
#             "electricalOptions": [
#                 "30 Amp",
#                 "50 Amp"
#             ],
#             "siteFeatures": [
#                 "Pull-Through",
#                 "Picnic Table"
#             ]
#         },
#         "tripType": [
#             "relaxingGetaway",
#             "familyCamping"
#         ],
#         "destinations": [
#             "nationalParks",
#             "beachDestinations"
#         ],
#         "routePreferences": [
#             "scenicRoute"
#         ],
#         "budget": [
#             "balanced"
#         ],
#         "specificNeeds": [
#             "Family-Friendly Stops",
#             "Pet-Friendly Areas"
#         ]
#     }