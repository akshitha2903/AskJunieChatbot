import os
import time
import logging
import traceback
from typing import List, Dict, Optional, Union
from datetime import datetime
from functools import cached_property
import asyncio

import openai
import googlemaps
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

from app.api.routes import router
from config import load_secrets
from app.features.location_template import Location, EnhancedLocationParser, RouteServices
from app.features.complete_template import TripDetails,EnhancedConversationManager,EnhancedConversationTemplate,ItineraryTemplate
from app.features.detail_extraction_template import detail_extraction
from app.services.session_manager import DatabaseManager
from app.services.data_integration import TravelSearchService

class TravelAgent:
    def __init__(self, openai_api_key, google_maps_api_key, db_manager, model="gpt-4o-mini", temperature=0.7, debug=True):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._openai_api_key = openai_api_key
        self._google_maps_api_key = google_maps_api_key
        self._model = model
        self._temperature = temperature
        self._debug = debug
        self.db_manager = db_manager
        self.current_itinerary = {}
        self.modification_history = {}
        self.detail_extraction_template = detail_extraction
        self._campground_service = None

    @cached_property
    def chat_model(self):
        return ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            openai_api_key=self._openai_api_key
        )

    @cached_property
    def gmaps(self):
        return googlemaps.Client(key=self._google_maps_api_key)

    @cached_property
    def location_parser(self):
        return EnhancedLocationParser(self.chat_model)

    @cached_property
    def route_services(self):
        return RouteServices(self._google_maps_api_key)

    @cached_property
    def conversation_manager(self):
        return EnhancedConversationManager()

    @cached_property
    def conversation_template(self):
        return EnhancedConversationTemplate()

    @cached_property
    def itinerary_template(self):
        return ItineraryTemplate

    @cached_property
    def campground_service(self):
        """
        Lazy loading property for CampgroundSearchService.
        Only initializes when first accessed.
        """
        if self._campground_service is None:
            self.logger.info("Initializing CampgroundSearchService...")
            self._campground_service = TravelSearchService(self._openai_api_key)
        return self._campground_service

    @cached_property
    def conversation_chain(self):
        return LLMChain(
            llm=self.chat_model,
            prompt=self.conversation_template.chat_prompt,
            verbose=False,
            output_key="conversation_response"
        )

    @cached_property
    def extraction_chain(self):
        return LLMChain(
            llm=self.chat_model,
            prompt=ChatPromptTemplate.from_template(self.detail_extraction_template),
            verbose=False
        )

    @cached_property
    def itinerary_chain(self):
        return LLMChain(
            llm=self.chat_model,
            prompt=ChatPromptTemplate.from_template(self.itinerary_template),
            verbose=False
        )
    def _extract_trip_details(self, message,session_id):
        try:
            conversation_history = self.db_manager.get_conversation_history(session_id)
            # Get conversation history for context
            conversation_text = " ".join([msg['content'] for msg in conversation_history])
            
            extraction_result = self.extraction_chain.run(
                message=message, 
                conversation_history=conversation_text
            )
            
            extracted_details = {}
            
            for line in extraction_result.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.lower() == 'none':
                        continue
                    
                    if key == 'start_location':
                        # Remove any quotes
                        value = value.strip('"')
                    elif key == 'destination':
                        # Remove any quotes
                        value = value.strip('"')
                    elif key == 'interests':
                        value = [i.strip() for i in value.split(',')]
                    elif key == 'num_travelers':
                        try:
                            value = int(value)
                        except:
                            continue
                    
                    extracted_details[key] = value
                    
            self.db_manager.update_trip_details(session_id, extracted_details)
            return extracted_details
        
        except Exception as e:
            self.logger.error(f"Error extracting trip details: {e}")
            return {}
            
    def _analyze_message_type(self, query: str, conversation_history: list) -> str:
        """
        Use the LLM to analyze the type of message/query from the user.
        Returns: 'casual', 'modification', or 'informational'
        """
        analysis_prompt = f"""
        Analyze the following user message in the context of a travel planning conversation.
        Previous messages: {conversation_history[-2:] if conversation_history else 'None'}
        
        Current message: {query}
        
        Classify the message into ONE of these categories:
        - casual: General conversation, general questions,travel tips, acknowledgments, thanks, or simple replies,compliments
        - modification: Requests to change or modify existing plans
        - informational: Regarding Itinerary plan or trip plans 
        
        Return only one word: casual or modification or informational
        """
        
        try:
            result = self.chat_model.invoke(analysis_prompt).content.strip().lower()
            return result if result in ['casual', 'modification', 'informational'] else 'informational'
        except Exception as e:
            self.logger.error(f"Error analyzing message type: {e}")
            return 'informational'
    
    def _is_modification_request(self, query: str) -> bool:
        """
        Detect if the query is requesting modifications to the existing itinerary
        """
        modification_keywords = [
            'change', 'modify', 'update', 'switch', 'instead', 
            'rather', 'prefer', 'different', 'alternative',
            'replace', 'swap', 'adjust'
        ]
        return any(keyword in query.lower() for keyword in modification_keywords)
    
    def _extract_modifications(self, query: str, current_trip_details: TripDetails) -> dict:
        """
        Extract modification requests from the query while preserving existing trip details
        
        Args:
            query (str): The modification request from the user
            current_trip_details (TripDetails): Current trip details to preserve
            
        Returns:
            dict: Modified trip details with preserved original values
        """
        try:
            # Start with current trip details as a dictionary
            modified_details = current_trip_details.dict()
            
            modification_prompt = f"""
            Given the user's modification request, identify ONLY the specific changes requested.
            
            Current trip details:
            - Start Location: {current_trip_details.start_location}
            - Destination: {current_trip_details.destination}
            - Duration: {current_trip_details.duration}
            - Budget: {current_trip_details.budget}
            - Travel Style: {current_trip_details.travel_style}
            - Number of Travelers: {current_trip_details.num_travelers}
            - Interests: {', '.join(current_trip_details.interests) if current_trip_details.interests else 'None'}
            - Special Requirements: {current_trip_details.special_requirements}
            - Dates: {current_trip_details.dates}
            
            User's modification request: {query}
            
            Return ONLY the explicitly requested changes in this format:
            field = new_value
            
            Only include fields that were specifically mentioned for modification.
            """
            
            result = self.chat_model.predict(modification_prompt)
            
            # Parse only explicitly requested changes
            for line in result.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value and value.lower() != 'none':
                        # Special handling for interests field
                        if key == 'interests':
                            if isinstance(value, str):
                                # Split by comma and clean up each interest
                                interests = [i.strip() for i in value.split(',')]
                                # Combine with existing interests
                                existing_interests = current_trip_details.interests or []
                                modified_details[key] = list(set(existing_interests + interests))
                        else:
                            modified_details[key] = value
            
            # Ensure the modification preserves any existing fields not being modified
            for key, value in current_trip_details.dict().items():
                if key not in modified_details and value is not None:
                    modified_details[key] = value
                    
            return modified_details
                    
        except Exception as e:
            self.logger.error(f"Error extracting modifications: {e}")
            # Return current trip details if modification extraction fails
            return current_trip_details.dict()

    async def create_itinerary(self, campground_names: List[str], offer_names: List[str], preferences: TripDetails, locations: List[str]) -> str:
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
                - Start Location: {preferences.start_location if preferences.start_location else 'Not specified'}
                - Destination: {preferences.destination if preferences.destination else 'Not specified'}
                - Travel Dates: {preferences.dates if preferences.dates else 'Not specified'}
                - Duration: {preferences.duration if preferences.duration else 'Not specified'}
                - Budget: {preferences.budget if preferences.budget else 'Flexible'}
                - Travel Style: {preferences.travel_style if preferences.travel_style else 'Not specified'}
                - Number of Travelers: {preferences.num_travelers if preferences.num_travelers else 'Not specified'}
                - Interests: {', '.join(preferences.interests) if preferences.interests else 'Not specified'}
                - Special Requirements: {preferences.special_requirements if preferences.special_requirements else 'None'}""",
                "locations": ', '.join(locations),
                "campgrounds": ', '.join(campground_names),
                "offers": ', '.join(offer_names)
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
            
            ### Mandatory Offers to mention in the itinerary:
            {offers}
            
            ### Nomad Rules:
            - Drive no longer than 3 hours per day.
            - Stay at each location for at least 3 nights.
            - Arrive at each destination by 3 PM local time.

            ### Instructions:
            1. Create an engaging, day-by-day itinerary starting from the first location and ending at the end location. 
            IMPORTANT: DO NOT CREATE A ROUND TRIP; IT SHOULD BE ONE-WAY.
            2. For each day, include:
            - Morning, afternoon, and evening activities, focusing on attractions, landmarks, hikes, cultural spots, or scenic views at or near the specified locations.
            - Use campgrounds strictly for stays and rest, not as attractions or activities.
            - Highlight any unique or memorable experiences at the attractions for that day.
            - Mention the offers in the itinerary whenever possible.
            3. Ensure the itinerary aligns with the budget, travel dates, and preferences provided.
            4. Adhere to the following rules for stays:
            - Use the provided campgrounds for overnight stays based on proximity to the day's ending location.
            5. Ensure all activities ALIGN WITH THE NOMAD RULES
            6. Finally at the end of the itinerary include the complete budget for the trip THAT SHOULD ALIGN WITH USER BUDGET PREFERENCES. STRICTLY FOLLOW THIS FORMAT like this example: "Budget: $1000" (IT SHOULD BE PROPER ESTIMATED NUMBERS )

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
            return itinerary_output
    async def generate_itinerary(self, trip_details: TripDetails):
        """Generate a complete itinerary based on collected trip details"""
        try:
            
            itinerary_text = self.itinerary_chain.run(trip_details=trip_details.dict())
            
            # Parse locations
            locations = self.location_parser.extract_locations(itinerary_text)
            geocoded_locations = self.route_services.geocode_locations(locations)
            self.logger.info(f"Geocoded Locations got")
            campgrounds = []
            offers = []
            self.logger.info(f"Searching for campgrounds and offers...")
            for loc in geocoded_locations:
                result = await self.campground_service.search_attractions(trip_details, loc)
                # Fix: Only extend if result contains lists
                if isinstance(result.get('campgrounds', []), list):
                    campgrounds.extend(result['campgrounds'])
                if isinstance(result.get('offers', []), list):
                    offers.extend(result['offers'])
            
            # Fix: Flatten and process campgrounds/offers more safely
            campground_names = {
                camp.get('name', '')
                for camp in campgrounds
                if isinstance(camp, dict) and camp.get('name')
            }
            
            offer_names = {
                offer.get('title', '')
                for offer in offers
                if isinstance(offer, dict) and offer.get('title')
            }
            addresses = [loc.address for loc in geocoded_locations]
            enhanced_itinerary=await self.create_itinerary(campground_names, offer_names, trip_details, addresses)
            locations = self.route_services.geocode_locations(locations)
            # Get route information
            route_info = self.route_services.get_route_info(locations)
            
            # Get nearby services
            route_services = {
                "services": self.route_services.find_nearby_services(locations)
            }
            
            return {
                "success": True,
                "itinerary_text": enhanced_itinerary,
                "locations": [vars(loc) for loc in locations],
                "route_info": route_info,
                "route_services": route_services
            }
                    
        except Exception as e:
            self.logger.error(f"Error generating itinerary: {e}")
            return {
                "success": False,
                "error": "Failed to generate itinerary. Please try again."
            }
    
    def get_route_services(self, locations):
        """
        Get nearby services along the route using RouteServices
        """
        return {
            "services": self.route_services.find_nearby_services(locations)
        }
        
    async def handle_conversation(self, query: str, session_id: str) -> dict:
        try:
            # Get current context
            self.db_manager.create_or_update_session(session_id)
            conversation_history = self.db_manager.get_conversation_history(session_id)
            trip_details_dict = self.db_manager.get_trip_details(session_id)
            trip_details = TripDetails(**trip_details_dict)
            
            # Check for existing itinerary
            current_itinerary = self.db_manager.get_current_itinerary(session_id)
            # Analyze message type
            message_type = self._analyze_message_type(query, conversation_history)
                # Check if query is specifically about campgrounds or offers
            is_campground_query = any(word in query.lower() for word in ['campground', 'campsite', 'camping', 'camp'])
            is_offer_query = any(word in query.lower() for word in ['offer', 'deal', 'discount', 'promotion'])

            # If asking specifically about campgrounds or offers
            if is_campground_query or is_offer_query:
                # Extract location from query or use trip details
                locations = self.location_parser.extract_locations(query)
                print(locations)
                locations = self.route_services.geocode_locations(locations)
                print(locations)
                if not locations:
                    # Use destination from trip details if available
                    if trip_details.destination:
                        locations = self.route_services.geocode_locations(trip_details.destination)
                    # Use start location as fallback
                    elif trip_details.start_location:
                        locations = self.route_services.geocode_locations(trip_details.start_location)
                    else:
                        return {
                            "type": "conversation",
                            "content": {
                                "response": "I'd be happy to help find campgrounds or offers! Could you please specify a location you're interested in?"
                            },
                            "trip_details": trip_details
                        }
                
                for location in locations:
                    # Search for attractions
                    search_results = await self.campground_service.search_attractions(query, location)
                    print(search_results)
                # Filter results based on query type
                if is_campground_query:
                    relevant_results = search_results.get('campgrounds', [])
                    result_type = "campgrounds"
                else:
                    relevant_results = search_results.get('offers', [])
                    result_type = "offers"
                # Format response
                if relevant_results:
                    response = f"I found several {result_type} near {location.name}:\n\n"
                    for idx, item in enumerate(relevant_results[:10], 1):
                        if result_type == "campgrounds":
                            response += f"{idx}. {item['name']}\n"
                            response += f"   Location: {item['location']}\n"
                            response += f"   Distance: {item['distance']} miles\n"
                        else:
                            response += f"{idx}. {item['title']}\n"
                            response += f"   Merchant: {item['merchant']}\n"
                            response += f"   Location: {item['location']}\n"
                            response += f"   Distance: {item['distance']} miles\n"
                        response += "\n"
                else:
                    response = f"I couldn't find any {result_type} in that area. Would you like to try a different location?"

                # Update conversation history
                self.db_manager.add_message(session_id, "user", query)
                self.db_manager.add_message(session_id, "assistant", response)

                return {
                    "type": "conversation",
                    "content": {"response": response},
                    "trip_details": trip_details
                }
            # If we have an existing itinerary and it's just casual conversation
            if current_itinerary and message_type == "casual":
                response = self.conversation_chain.run(
                    query=query,
                    conversation_history=conversation_history,
                    trip_details=trip_details.dict()
                )
                
                self.db_manager.add_message(session_id, "user", query)
                self.db_manager.add_message(session_id, "assistant", response)
                return {
                    "type": "conversation",
                    "content": {"response": response},
                    "trip_details": trip_details
                }

            # Handle modifications to existing itinerary
            if current_itinerary and message_type == "modification":
                modifications = self._extract_modifications(query,trip_details)
                updated_trip_details = TripDetails(**modifications)
                
                # Save updated trip details
                self.db_manager.update_trip_details(session_id, updated_trip_details.dict())
                
                # Update conversation history
                self.db_manager.add_message(session_id, "user", query)
                self.db_manager.add_message(session_id, "assistant", response)
                
                return {
                    "type": "conversation",
                    "content": {"response": "Would you like to generate a new itinerary or travel tips for this modification?"},
                    "trip_details": updated_trip_details
                }


            # Regular conversation flow
            response = self.conversation_chain.run(
                query=query,
                conversation_history=conversation_history,
                trip_details=trip_details.dict()
            )

            # Only extract and update trip details if message is informational
            if message_type == "informational":
                extracted_info = self._extract_trip_details(query, session_id)
                trip_details.update(extracted_info)
                self.db_manager.update_trip_details(session_id, trip_details.dict())

            # Update conversation history
            self.db_manager.add_message(session_id, "user", query)
            self.db_manager.add_message(session_id, "assistant", response)

            # Only generate new itinerary if:
            # 1. Trip details are ready
            # 2. Message is informational
            # 3. We don't have an existing itinerary
            if (trip_details.check_readiness() and 
                message_type == "informational" and 
                not current_itinerary):
                itinerary = await self.generate_itinerary(trip_details)
                self.db_manager.save_itinerary(session_id, itinerary)
                return {
                    "type": "itinerary",
                    "content": itinerary,
                    "trip_details": trip_details
                }

            return {
                "type": "conversation",
                "content": {"response": response},
                "trip_details": trip_details
            }
                    
        except Exception as e:
            self.logger.error(f"Conversation handling error: {e}")
            return {
                "type": "conversation",
                "content": {
                    "response": "I'm having trouble processing your request. Could you rephrase or provide more details?"
                },
                "trip_details": TripDetails()
            }
    