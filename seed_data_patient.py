from pymongo import MongoClient

client = MongoClient("mongodb://admin:Rob123%21@localhost:27017/admin")
db = client["patient_db"]
collection = db["admitted_patient"]

# Sample patients
patients = [
    {
        "patient_id": "101",
        "name": "Patient 101",
        "age": 45,
        "first_visit_reason": "Chest pain and shortness of breath",
        "history": ["Hypertension diagnosed 2019", "Type 2 Diabetes since 2020", "Cholesterol management"],
        "medications": ["Metformin 500mg twice daily", "Lisinopril 10mg daily", "Aspirin 81mg daily"],
        "allergies": ["Penicillin"],
        "last_visit_date": "2024-10-15"
    },
    {
        "patient_id": "205",
        "name": "Patient 205",
        "age": 32,
        "first_visit_reason": "Severe migraine headaches",
        "history": ["Chronic migraines since 2018", "Anxiety disorder"],
        "medications": ["Sumatriptan 50mg as needed", "Propranolol 40mg daily"],
        "allergies": [],
        "last_visit_date": "2024-11-01"
    }
]

# Insert or update
for patient in patients:
    collection.update_one(
        {"patient_id": patient["patient_id"]},
        {"$set": patient},
        upsert=True
    )

print(f"✅ Inserted {len(patients)} sample patients")
client.close()