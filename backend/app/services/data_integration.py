from typing import List, Dict, Optional, Any, Union, Tuple
from pydantic import BaseModel
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from geopy.distance import geodesic
from config import load_secrets, connect_to_db
from functools import lru_cache
import pandas as pd  # type: ignore
import asyncio
import logging
from pathlib import Path
from time import perf_counter
import pickle
import tempfile
from langchain.docstore.document import Document
tempfile.tempdir = "D:/mysql_temp"
secrets = load_secrets()
openai_api_key = secrets["OPENAI_API_KEY"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TravelService')

class ModifiedPromptTemplate(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        registered, not_registered = __pydantic_self__.filter_data(data)
        super().__init__(**registered)
        for k, v in not_registered.items():
            __pydantic_self__.__dict__[k] = v
    
    @classmethod
    def filter_data(cls, data):
        registered_attr = {}
        not_registered_attr = {}
        annots = cls.__annotations__
        for k, v in data.items():
            if k in annots:
                registered_attr[k] = v
            else:
                not_registered_attr[k] = v
        return registered_attr, not_registered_attr
class RVDetails(BaseModel):
    length: Optional[float] = None
    height: Optional[float] = None
    type: Optional[str] = None

class TripPreferences(BaseModel):
    looking_for: Optional[List[str]] = None
    travel_dates: Optional[Dict[str, str]] = None
    rv_details: Optional[RVDetails] = None
    route_type: Optional[List[str]] = None
    specific_needs: Optional[List[str]] = None
    explore: Optional[List[Union[str, Dict[str, str]]]] = None
    amenities: Optional[List[Dict[str, str]]] = None
    budget: Optional[str] = None
    
class TripDetails(BaseModel):
    start_location: Optional[str] = None
    destination: Optional[str] = None
    duration: Optional[str] = None
    budget: Optional[str] = None
    travel_style: Optional[str] = None
    interests: List[str] = []
    num_travelers: Optional[int] = None
    special_requirements: Optional[str] = None
    dates: Optional[str] = None
    ready_for_itinerary: bool = False

class Location(BaseModel):
    name: str
    lat: float
    lng: float

class RAGTravelDatabase:
    def __init__(self, openai_api_key: str, embeddings_dir: str = "app/embeddings"):
        logger.info("Initializing RAGTravelDatabase")
        start_time = perf_counter()
        
        self.embeddings_dir = Path(embeddings_dir).resolve()
        self.embeddings_dir.mkdir(exist_ok=True)
        self.camp_index_path = self.embeddings_dir / "campgrounds.faiss"
        self.camp_docstore_path = self.embeddings_dir / "campgrounds_docstore.pkl"
        self.offer_index_path = self.embeddings_dir / "offers.faiss"
        self.offer_docstore_path = self.embeddings_dir / "offers_docstore.pkl"
        self.conn = connect_to_db()
        self.cursor = self.conn.cursor(dictionary=True)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(openai_api_key=openai_api_key)
        self.vector_stores = {}
        self._initialize_vector_stores()
        self.campground_df = None
        self.offer_df = None
        
        end_time = perf_counter()
        logger.info(f"RAGTravelDatabase initialized in {end_time - start_time:.2f} seconds")

    def _initialize_vector_stores(self):
        """Initialize vector stores by loading existing FAISS index and docstore for campgrounds and offers."""
        start_time = perf_counter()
        logger.info("Loading existing vector stores")
        
        # Initialize campgrounds vector store
        try:
            logger.info(f"Checking for index at: {self.camp_index_path}")
            logger.info(f"Checking for docstore at: {self.camp_docstore_path}")
            
            if not (self.camp_index_path.exists() and self.camp_docstore_path.exists()):
                raise FileNotFoundError("Campground index or docstore files not found")
                
            # Load the docstore
            with open(self.camp_docstore_path, 'rb') as f:
                docstore = pickle.load(f)
            
            # Load the FAISS index and update its docstore
            vector_store = FAISS.load_local(
                folder_path=str(self.embeddings_dir),
                embeddings=self.embeddings,
                index_name="campgrounds",
                allow_dangerous_deserialization=True
            )
            vector_store.docstore = docstore
            
            self.vector_stores['campgrounds'] = vector_store
            logger.info("Successfully loaded campground vector store")
                
        except Exception as e:
            logger.error(f"Error loading campground vector store: {str(e)}")
            logger.warning("Falling back to creating new campground vector store")
            campground_data = self._get_campground_documents()
            self.vector_stores['campgrounds'] = self._create_vector_store(campground_data, 'campgrounds')
        
        # Initialize offers vector store
        try:
            logger.info(f"Checking for index at: {self.offer_index_path}")
            logger.info(f"Checking for docstore at: {self.offer_docstore_path}")
            
            if not (self.offer_index_path.exists() and self.offer_docstore_path.exists()):
                raise FileNotFoundError("Offers index or docstore files not found")
            
            # Load the docstore
            with open(self.offer_docstore_path, 'rb') as f:
                docstore = pickle.load(f)
            
            # Load the FAISS index and update its docstore
            vector_store = FAISS.load_local(
                folder_path=str(self.embeddings_dir),
                embeddings=self.embeddings,
                index_name="offers",
                allow_dangerous_deserialization=True
            )
            vector_store.docstore = docstore
            
            self.vector_stores['offers'] = vector_store
            logger.info("Successfully loaded offers vector store")
                
        except Exception as e:
            logger.error(f"Error loading offer vector store: {str(e)}")
            logger.warning("Falling back to creating new offer vector store")
            offer_data = self._get_offer_documents()
            self.vector_stores['offers'] = self._create_vector_store(offer_data, 'offers')

        end_time = perf_counter()
        logger.info(f"Vector stores initialized in {end_time - start_time:.2f} seconds")
    @lru_cache(maxsize=1)
    def _get_campground_documents(self) -> List[str]:
        start_time = perf_counter()
        logger.info("Fetching campground documents from database")
        
        query = """
        SELECT 
            c.*
        FROM campgrounds c
        """
        self.cursor.execute(query)
        campgrounds = self.cursor.fetchall()
        self.campground_df = pd.DataFrame(campgrounds)
        
        logger.info(f"Retrieved {len(campgrounds)} campground records")
        documents = []
        for camp in campgrounds:
            text = f"""CAMP: {camp['name']}
                    Description: {camp['description']}
                    Location: {camp['address']}, {camp['city']}, {camp['state']}
                    Coordinates: {camp['lat']}, {camp['lng']}
                    Phone: {camp['phone'] or 'No phone'}
                    Email: {camp['email'] or 'No email'}
                    Website: {camp['website'] or 'No website'}
                    """
            doc = Document(
                page_content=text,
                metadata={
                    "id": camp['id'],
                    "type": "campground"
                }
            )
            documents.append(doc)
        
        end_time = perf_counter()
        logger.info(f"Campground documents processed in {end_time - start_time:.2f} seconds")
        return documents

    @lru_cache(maxsize=1)
    def _get_offer_documents(self) -> List[str]:
        start_time = perf_counter()
        logger.info("Fetching offer documents from database")
        
        query = """
        SELECT 
            ao.*,
            am.name as merchant_name,
            apc.title as category,
            aol.city,
            aol.state,
            aol.lat,
            aol.lng
        FROM abenity_offers ao
        JOIN abenity_merchants am ON ao.abenity_merchant_id = am.id
        JOIN abenity_offer_locations aol ON ao.id = aol.abenity_offer_id
        JOIN abenity_perk_categories apc ON ao.abenity_perk_category_id = apc.id
        """
        self.cursor.execute(query)
        offers = self.cursor.fetchall()
        self.offer_df = pd.DataFrame(offers)
        
        logger.info(f"Retrieved {len(offers)} offer records")
        documents = []
        for offer in offers:
            text = f"""Offer: {offer['title']}
                    Merchant: {offer['merchant_name']}
                    Category: {offer['category']}
                    Location: {offer['city']}, {offer['state']}
                    Coordinates: {offer['lat']}, {offer['lng']}
                    Description: {offer['link']}
                    Expiration: {offer['exp_date']}
                    """
            doc = Document(
                page_content=text,
                metadata={
                    "id": offer['id'],
                    "type": "offer"
                }
            )
            documents.append(doc)
        
        end_time = perf_counter()
        logger.info(f"Offer documents processed in {end_time - start_time:.2f} seconds")
        return documents

    def _create_vector_store(self, documents: List[Document], store_name: str) -> FAISS:
        start_time = perf_counter()
        logger.info(f"Creating FAISS vector store for {store_name}")
        
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save the vector store after creation
        self._save_vector_store(store_name, vector_store)
        
        end_time = perf_counter()
        logger.info(f"Vector store created in {end_time - start_time:.2f} seconds")
        return vector_store

    def _save_vector_store(self, name: str, vector_store: FAISS):
        """Save the vector store and its docstore to disk."""
        logger.info(f"Saving vector store: {name}")
        try:
            # Save the FAISS index
            vector_store.save_local(
                folder_path=str(self.embeddings_dir),
                index_name=name
            )
            
            # Save the docstore separately
            docstore_path = self.embeddings_dir / f"{name}_docstore.pkl"
            with open(docstore_path, 'wb') as f:
                pickle.dump(vector_store.docstore, f)
                
            logger.info(f"Successfully saved vector store: {name}")
        except Exception as e:
            logger.error(f"Error saving vector store {name}: {str(e)}")

    def refresh_embeddings(self):
        logger.info("Refreshing all vector stores")
        campground_data = self._get_campground_documents()
        offer_data = self._get_offer_documents()
        
        self.vector_stores['campgrounds'] = self._create_vector_store(campground_data, 'campgrounds')
        self.vector_stores['offers'] = self._create_vector_store(offer_data, 'offers')
        
        self._save_vector_store('campgrounds', self.vector_stores['campgrounds'])
        self._save_vector_store('offers', self.vector_stores['offers'])
        
        logger.info("Vector stores refresh complete")

class EnhancedTravelFinder:
    def __init__(self, rag_db: RAGTravelDatabase):
        self.rag_db = rag_db
        self.similarity_threshold = 0.8
        self.logger = logging.getLogger('EnhancedTravelFinder')

    async def _parallel_similarity_search(self, store_name: str, query: str) -> List[Tuple]:
        start_time = perf_counter()
        logger.info(f"Starting similarity search for {store_name}")
        
        results = self.rag_db.vector_stores[store_name].similarity_search_with_score(
            query,
            k=50
        )
        
        end_time = perf_counter()
        logger.info(f"Similarity search for {store_name} completed in {end_time - start_time:.2f} seconds")
        return results

    async def find_attractions(
        self,
        trip_details: TripDetails,
        current_location: Location,
        max_distance: float = 120  # miles
    ) -> Dict[str, List[Dict]]:
        total_start_time = perf_counter()
        logger.info("Starting attraction search")
        
        search_query = self._construct_search_query(trip_details)
        logger.info(f"Constructed search query: {search_query}")
        
        # Parallel search for both campgrounds and offers
        campground_results, offer_results = await asyncio.gather(
            self._parallel_similarity_search('campgrounds', search_query),
            self._parallel_similarity_search('offers', search_query)
        )
        
        # Process results in parallel
        processing_tasks = []
        
        # Process campgrounds
        for doc, score in campground_results:
            if score <= self.similarity_threshold:
                processing_tasks.append(
                    self._process_campground(doc, score, current_location, max_distance)
                )
        
        # Process offers
        for doc, score in offer_results:
            if score <= self.similarity_threshold:
                processing_tasks.append(
                    self._process_offer(doc, score, current_location, max_distance)
                )
        
        # Gather all results
        all_results = await asyncio.gather(*processing_tasks)
        
        # Separate and filter results
        attractions = {
            'campgrounds': [],
            'offers': []
        }
        
        for result in all_results:
            if result and 'type' in result:
                if result['type'] == 'campground':
                    attractions['campgrounds'].append(result)
                elif result['type'] == 'offer':
                    attractions['offers'].append(result)
        
        # Sort and deduplicate results
        for key in attractions:
            if attractions[key]:
                df = pd.DataFrame(attractions[key]).drop_duplicates(
                    subset=['name'] if key == 'campgrounds' else ['title']
                )
                df = df.sort_values(['similarity_score', 'distance'])
                attractions[key] = df.to_dict('records')
        
        total_end_time = perf_counter()
        logger.info(f"""Search completed:
            - Total time: {total_end_time - total_start_time:.2f} seconds
            - Found {len(attractions['campgrounds'])} campgrounds and {len(attractions['offers'])} offers""")
        
        return attractions

    def _extract_coordinates(self, content: str) -> Optional[tuple]:
        try:
            coords_section = content.split('Coordinates:')[1].split('\n')[0].strip()
            lat, lng = map(float, coords_section.split(','))
            return (lat, lng)
        except:
            return None

    async def _process_campground(
        self,
        doc,
        score: float,
        current_location: Location,
        max_distance: float
    ) -> Optional[Dict]:
        start_time = perf_counter()
        
        camp_info = self._extract_campground_info(doc.page_content)
        coords = self._extract_coordinates(doc.page_content)
        
        if coords:
            distance = geodesic(
                (current_location.lat, current_location.lng),
                coords
            ).miles
            
            if distance <= max_distance:
                result = {
                    'type': 'campground',
                    'name': camp_info['name'],
                    'description': camp_info['description'],
                    'location': camp_info['location'],
                    'contact': {
                        'phone': camp_info['phone'],
                        'email': camp_info['email'],
                        'website': camp_info['website']
                    },
                    'distance': round(distance, 2),
                    'similarity_score': round(score, 4)
                }
                
                end_time = perf_counter()
                logger.debug(f"Processed campground {camp_info['name']} in {end_time - start_time:.4f} seconds")
                return result
        
        return None

    async def _process_offer(
        self,
        doc,
        score: float,
        current_location: Location,
        max_distance: float
    ) -> Optional[Dict]:
        start_time = perf_counter()
        
        offer_info = self._extract_offer_info(doc.page_content)
        coords = self._extract_coordinates(doc.page_content)
        
        if coords:
            distance = geodesic(
                (current_location.lat, current_location.lng),
                coords
            ).miles
            
            if distance <= max_distance:
                result = {
                    'type': 'offer',
                    'title': offer_info['title'],
                    'merchant': offer_info['merchant'],
                    'category': offer_info['category'],
                    'location': offer_info['location'],
                    'description': offer_info['description'],
                    'expiration': offer_info['expiration'],
                    'distance': round(distance, 2),
                    'similarity_score': round(score, 4)
                }
                
                end_time = perf_counter()
                logger.debug(f"Processed offer {offer_info['title']} in {end_time - start_time:.4f} seconds")
                return result
        
        return None

    def _extract_campground_info(self, content: str) -> dict:
        fields = {
            'name': ('CAMP:', 'Unknown Campground'),
            'description': ('Description:', 'No description available'),
            'location': ('Location:', 'Unknown Location'),
            'phone': ('Phone:', 'No phone available'),
            'email': ('Email:', 'No email available'),
            'website': ('Website:', 'No website available')
        }
        
        result = {}
        for field, (prefix, default) in fields.items():
            try:
                result[field] = content.split(prefix)[1].split('\n')[0].strip()
            except (IndexError, AttributeError):
                result[field] = default
        
        return result

    def _extract_offer_info(self, content: str) -> dict:
        fields = {
            'title': ('Offer:', 'Unknown Offer'),
            'merchant': ('Merchant:', 'Unknown Merchant'),
            'category': ('Category:', 'Unknown Category'),
            'location': ('Location:', 'Unknown Location'),
            'description': ('Description:', 'No description available'),
            'expiration': ('Expiration:', 'No expiration available')
        }
        
        result = {}
        for field, (prefix, default) in fields.items():
            try:
                result[field] = content.split(prefix)[1].split('\n')[0].strip()
            except (IndexError, AttributeError):
                result[field] = default
        
        return result

    def _construct_search_query(self, trip_details: TripDetails) -> str:
        if isinstance(trip_details, str):
        # If input is a string, use it directly as the query
            return trip_details.strip()
    
        query_parts = []

        # Add start location
        if trip_details.start_location:
            query_parts.append(f"Starting from: {trip_details.start_location}")

        # Add destination
        if trip_details.destination:
            query_parts.append(f"Destination: {trip_details.destination}")

        # Add travel style
        if trip_details.travel_style:
            query_parts.append(f"Travel style: {trip_details.travel_style}")

        # Add interests
        if trip_details.interests:
            query_parts.append(f"Interests: {', '.join(trip_details.interests)}")

        # Add special requirements
        if trip_details.special_requirements:
            query_parts.append(f"Special requirements: {trip_details.special_requirements}")

        # Combine all parts into a query string
        query_string = " | ".join(query_parts) if query_parts else "General search"
        return query_string

class TravelSearchService:
    def __init__(self, openai_api_key: str):
        start_time = perf_counter()
        
        self.rag_db = RAGTravelDatabase(openai_api_key,embeddings_dir="app/embeddings")
        self.travel_finder = EnhancedTravelFinder(self.rag_db)
        
        end_time = perf_counter()
        logger.info(f"TravelSearchService initialized in {end_time - start_time:.2f} seconds")

    async def search_attractions(
        self,
        trip_details: TripDetails,
        current_location: Location
    ) -> Dict[str, List[Dict]]:
        start_time = perf_counter()
        logger.info("Starting campgrounds and offers search")
        
        results = await self.travel_finder.find_attractions(
            trip_details,
            current_location
        )
        
        end_time = perf_counter()
        logger.info(f"Campgrounds and offers search completed in {end_time - start_time:.2f} seconds")
        return results
    
   
class PreferenceCampgroundFinder:
    def __init__(self, rag_db: RAGTravelDatabase):
        self.rag_db = rag_db
        self.similarity_threshold = 0.8  # Higher threshold since we're using <= now
        self.max_workers = 4
        self.logger = logging.getLogger('PreferenceCampgroundFinder Initialized')
    
    async def _parallel_similarity_search(self, query: str) -> List[Tuple]:
        start_time = perf_counter()
        logger.info("Starting similarity search")
        
        results = self.rag_db.vector_stores['campgrounds'].similarity_search_with_score(
            query,
            k=100
        )
        
        end_time = perf_counter()
        logger.info(f"Similarity search completed in {end_time - start_time:.2f} seconds")
        return results
    async def find_campgrounds_by_preferences(
        self, 
        preferences: TripPreferences, 
        current_location: Location, 
        max_distance: float = 120  # miles
    ) -> List[Dict]:
        total_start_time = perf_counter()
        logger.info("Starting campground search by preferences")
        
        preference_query = self._construct_preference_query(preferences)
        
        search_start_time = perf_counter()
        campground_results = await self._parallel_similarity_search(preference_query)
        logger.info(f"Retrieved {len(campground_results)} initial results")
        
        processing_start_time = perf_counter()
        tasks = [
            self._process_campground(doc, score, current_location, max_distance)
            for doc, score in campground_results
            if score <= self.similarity_threshold
        ]

        matching_campgrounds = await asyncio.gather(*tasks)
        matching_campgrounds = [c for c in matching_campgrounds if c]
        
        if matching_campgrounds:
            df = pd.DataFrame(matching_campgrounds).drop_duplicates(subset=['name'])
            df = df.sort_values(['similarity_score', 'distance'])
            matching_campgrounds = df.to_dict('records')
        
        total_end_time = perf_counter()
        logger.info(f"""Search completed:
            - Total time: {total_end_time - total_start_time:.2f} seconds
            - Search time: {processing_start_time - search_start_time:.2f} seconds
            - Processing time: {total_end_time - processing_start_time:.2f} seconds
            - Found {len(matching_campgrounds)} matching campgrounds""")
        
        return matching_campgrounds
    
        
    async def _process_campground(
        self,
        doc,
        score: float,
        current_location: Location,
        max_distance: float
    ) -> Optional[Dict]:
        start_time = perf_counter()
        
        camp_info = self.extract_campground_info(doc.page_content)
        coords = self._extract_coordinates(doc.page_content)
        
        if coords:
            distance = geodesic(
                (current_location.lat, current_location.lng),
                coords
            ).miles
            
            if distance <= max_distance:
                result = {
                    'name': camp_info['name'],
                    'description': {
                        'description': camp_info['description'],
                        'phone': camp_info['phone'],
                        'email': camp_info['email'],
                        'website': camp_info['website'],
                        'location': camp_info['location']
                    },
                    'distance': round(distance, 2),
                    'similarity_score': round(score, 4)
                }
                end_time = perf_counter()
                logger.debug(f"Processed campground {camp_info['name']} in {end_time - start_time:.4f} seconds")
                return result
                
        return None
    def _extract_coordinates(self, content: str) -> Optional[tuple]:
        try:
            coords_section = content.split('Coordinates:')[1].split('\n')[0].strip()
            lat, lng = map(float, coords_section.split(','))
            return (lat, lng)
        except:
            return None

    def extract_campground_info(self,content: str) -> dict:
        
        fields = {
            'name': ('CAMP:', 'Unknown Campground'),
            'description': ('Description:', 'No description available'),
            'location': ('Location:', 'Unknown Location'),
            'phone': ('Phone:', 'No phone available'),
            'email': ('Email:', 'No email available'),
            'website': ('Website:', 'No website available')
        }
        
        result = {}
        
        for field, (prefix, default) in fields.items():
            try:
                result[field] = content.split(prefix)[1].split('\n')[0].strip()
            except (IndexError, AttributeError):
                result[field] = default
        return result
    
    def _construct_preference_query(self, preferences: TripPreferences) -> str:
        query_parts = []

        # Add RV details (type and max length)
        if preferences.rv_details:
            rv_details = []
            if preferences.rv_details.type:
                rv_details.append(f"RVs only, {preferences.rv_details.type}")
            if preferences.rv_details.length:
                rv_details.append(f"{preferences.rv_details.length} ft max RV length")
            if rv_details:
                query_parts.append(" ".join(rv_details))

        # Add specific needs (like pet-friendly, full hookups)
        if preferences.specific_needs:
            query_parts.append(f"Specific needs: {', '.join(preferences.specific_needs)}")

        # Add exploration preferences (e.g., beach destinations)
        if preferences.explore:
            query_parts.append(f"Explore: {', '.join(preferences.explore)}")

        # Add amenities (like 20amp, Air conditioning, etc.)
        if preferences.amenities:
            amenity_list = []
            for amenity in preferences.amenities:
                for key, value in amenity.items():
                    if isinstance(value, list):
                        amenity_list.append(f"{key}: {', '.join(value)}")
                    else:
                        amenity_list.append(f"{key}: {value}")
            query_parts.append(f"Amenities: {', '.join(amenity_list)}")

        # Add budget
        if preferences.budget:
            query_parts.append(f"Budget: {preferences.budget}")

        # Combine all preference parts into a query string
        query_string = " | ".join(query_parts) if query_parts else "General search"
        return query_string


# Example usage in a service class
class CampgroundSearchService:
    def __init__(self, openai_api_key: str):
        start_time = perf_counter()
        
        self.rag_db = RAGTravelDatabase(openai_api_key,embeddings_dir="app/embeddings")
        self.preference_finder = PreferenceCampgroundFinder(self.rag_db)
        
        end_time = perf_counter()
        logger.info(f"CampgroundSearchService initialized in {end_time - start_time:.2f} seconds")

    async def search_campgrounds(
        self, 
        preferences: TripPreferences, 
        current_location: Location
    ) -> List[Dict]:
        start_time = perf_counter()
        logger.info("Starting campground search")
        
        results = await self.preference_finder.find_campgrounds_by_preferences(
            preferences, 
            current_location
        )
        
        end_time = perf_counter()
        logger.info(f"Campground search completed in {end_time - start_time:.2f} seconds")
        return results
        
