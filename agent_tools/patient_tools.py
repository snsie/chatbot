from smolagents import tool
from pymongo import MongoClient
import re

@tool
def get_patient_data(patient_id: str) -> str:
    """
    Fetches complete medical records for a patient from the hospital database.
    
    Args:
        patient_id: The patient's ID number (e.g., "101" or "patient_101")
    
    Returns:
        Formatted patient medical information including history, medications, and visit reason.
    """
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://admin:Rob123%21@localhost:27017/admin")
        db = client["patient_db"]
        collection = db["admitted_patient"]
        
        # Clean patient ID (extract numbers only)
        patient_id_clean = re.sub(r'\D', '', patient_id)
        print(f"Fetching data for Patient ID: {patient_id_clean}")
        # Try both string and integer ID formats
        # Try both string and integer ID formats
        query = {
            "$or": [
                {"patient_id": patient_id_clean},
                {"patient_id": int(patient_id_clean) if patient_id_clean.isdigit() else None}
            ]
        }
        print(f"[Tool] MongoDB query: {query}")
        
        patient = collection.find_one(query)
        print(f"[Tool] Found patient: {patient is not None}")
        
        if not patient:
            client.close()
            return f"❌ Patient ID {patient_id} not found in database."
        
        # Format patient data
        name = patient.get("name", "Unknown")
        age = patient.get("age", "N/A")
        first_visit = patient.get("first_visit_reason", "N/A")
        history = patient.get("history", [])
        medications = patient.get("medications", [])
        allergies = patient.get("allergies", [])
        last_visit = patient.get("last_visit_date", "N/A")
        
        # Build formatted response
        history_str = "\n  • ".join(history) if history else "None recorded"
        meds_str = "\n  • ".join(medications) if medications else "None"
        allergies_str = ", ".join(allergies) if allergies else "None known"
        
        result = f"""Patient #{patient_id_clean} - {name}

    Basic Info:
  • Age: {age}
  • Last Visit: {last_visit}

    First Visit Reason:
  {first_visit}

    Medical History:
  • {history_str}

    Current Medications:
  • {meds_str}

    Allergies: {allergies_str}
"""
        print(f"Patient Data from tool - {result}")
        client.close()
        return result
    
    except Exception as e:
        return f"❌ Error fetching patient data: {str(e)}"