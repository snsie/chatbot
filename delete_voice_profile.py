#!/usr/bin/env python3
"""
Complete Voice Profile Deletion Script
=====================================

This script completely removes a voice profile from both storage systems:
1. File-based storage (data/ folder and enrollments.npz)
2. MongoDB database storage

Usage: python delete_voice_profile.py <username>
Example: python delete_voice_profile.py scott
"""

import sys
import shutil
from pathlib import Path
import subprocess
from pymongo import MongoClient

# MongoDB connection (same as in chatbot)
MONGO_URI = "mongodb://admin:Rob123%21@localhost:27017/admin"
ENROLL_PATH = "enrollments.npz"

def delete_voice_profile(username: str):
    """Completely delete a voice profile from both file and database storage."""
    
    print(f"🗑️ Deleting voice profile for: {username}")
    
    # 1. Delete from file system
    data_folder = Path("data") / username
    if data_folder.exists():
        print(f"📁 Removing folder: {data_folder}")
        shutil.rmtree(data_folder)
        print("✅ Folder deleted")
    else:
        print(f"📁 Folder {data_folder} not found (already deleted?)")
    
    # 2. Rebuild enrollments.npz to remove from file-based voice ID
    print("🔄 Rebuilding enrollments.npz...")
    try:
        subprocess.run(
            ["python", "build_enrollments.py", "--root", "data", "--out", ENROLL_PATH],
            check=True
        )
        print("✅ enrollments.npz rebuilt")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to rebuild enrollments.npz: {e}")
    
    # 3. Delete from MongoDB
    print("🗄️ Removing from MongoDB...")
    try:
        mongo = MongoClient(MONGO_URI)
        people = mongo["voice_db"]["people"]
        
        result = people.delete_one({"name": username})
        
        if result.deleted_count > 0:
            print(f"✅ Deleted {result.deleted_count} record(s) from MongoDB")
        else:
            print(f"📄 No MongoDB record found for {username}")
            
        mongo.close()
        
    except Exception as e:
        print(f"❌ MongoDB deletion failed: {e}")
    
    print(f"🎉 Voice profile deletion complete for {username}")
    print("The chatbot should no longer recognize this voice.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_voice_profile.py <username>")
        print("Example: python delete_voice_profile.py scott")
        sys.exit(1)
    
    username = sys.argv[1].strip()
    if not username:
        print("Error: Username cannot be empty")
        sys.exit(1)
    
    delete_voice_profile(username)