import json
from datetime import datetime
from typing import Optional, List, Dict
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_config: dict = None):
        default_config = {
            "host": "localhost",  # Replace with your DB host
            "database": "travel_agent_db",  # Replace with your DB name
            "user": "root",  # Replace with your DB user
            "password": "akshitha2903",  # Replace with your DB password
        }
        self.db_config = default_config if db_config is None else {**default_config, **db_config}

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = mysql.connector.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_database(self):
        """Create necessary tables if they don't exist"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        trip_details JSON 
                    )
                """)

                # Create conversation_history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255),
                        role VARCHAR(50),
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)

                # Create current_itinerary table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS current_itinerary (
                        session_id VARCHAR(255) PRIMARY KEY,
                        itinerary_data JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)

                # Create modification_history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS modification_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255),
                        modifications JSON,
                        result JSON,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)
                conn.commit()

    def create_or_update_session(self, session_id: str) -> None:
        """Create a new session or update last_active timestamp"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions (session_id)
                    VALUES (%s)
                    ON DUPLICATE KEY UPDATE last_active = CURRENT_TIMESTAMP
                """, (session_id,))
                conn.commit()

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history for a session"""
        with self.get_connection() as conn:
            with conn.cursor(dictionary=True) as cur:
                cur.execute("""
                    SELECT role, content, timestamp
                    FROM conversation_history
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                """, (session_id,))
                rows = cur.fetchall()
                return rows

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a new message to the conversation history"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_history (session_id, role, content)
                    VALUES (%s, %s, %s)
                """, (session_id, role, content))
                conn.commit()

    def get_trip_details(self, session_id: str) -> dict:
        """Get trip details for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT trip_details
                    FROM sessions
                    WHERE session_id = %s
                """, (session_id,))
                result = cur.fetchone()
                return json.loads(result[0]) if result and result[0] else {}

    def update_trip_details(self, session_id: str, details: dict) -> None:
        """Update trip details for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sessions
                    SET trip_details = JSON_MERGE_PATCH(trip_details, %s)
                    WHERE session_id = %s
                """, (json.dumps(details), session_id))
                conn.commit()

    def save_itinerary(self, session_id: str, itinerary_data: dict) -> None:
        """Save or update current itinerary"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO current_itinerary (session_id, itinerary_data)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE 
                        itinerary_data = VALUES(itinerary_data),
                        last_modified = CURRENT_TIMESTAMP
                """, (session_id, json.dumps(itinerary_data)))
                conn.commit()

    def get_current_itinerary(self, session_id: str) -> Optional[dict]:
        """Retrieve current itinerary for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT itinerary_data
                    FROM current_itinerary
                    WHERE session_id = %s
                """, (session_id,))
                result = cur.fetchone()
                return json.loads(result[0]) if result and result[0] else None

    def add_modification_history(self, session_id: str, modifications: dict, result: dict) -> None:
        """Add a modification history entry"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO modification_history 
                    (session_id, modifications, result)
                    VALUES (%s, %s, %s)
                """, (session_id, json.dumps(modifications), json.dumps(result)))
                conn.commit()

    def cleanup_old_sessions(self, days: int = 30) -> None:
        """Clean up sessions older than specified days"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM sessions
                    WHERE last_active < NOW() - INTERVAL %s DAY
                """, (days,))
                conn.commit()
