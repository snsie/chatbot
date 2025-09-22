# MongoDB Integration for Voice Chatbot

This voice chatbot now includes MongoDB integration to store all conversations persistently.

## Features

- **Automatic Conversation Storage**: All conversations are automatically saved to MongoDB
- **Message Tracking**: Every user and assistant message is stored with timestamps
- **Conversation History**: Retrieve past conversations and messages
- **Persistent Storage**: Conversations persist across chatbot restarts

## MongoDB Configuration

The MongoDB settings are configurable in the main script:

```python
# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27018/"
MONGODB_DATABASE = "chatbot_conversations"
MONGODB_COLLECTION = "conversations"
```

## Database Structure

### Collection: `conversations`

Each conversation document contains:

```json
{
  "_id": ObjectId("..."),
  "conversation_id": "1758292648626261",
  "started_at": ISODate("2025-09-19T14:37:28.626Z"),
  "system_prompt": "Your name is Cora. You are an autonomous AI assistant...",
  "messages": [
    {
      "role": "user",
      "content": "Hello, this is a test message.",
      "timestamp": ISODate("2025-09-19T14:37:28.630Z")
    },
    {
      "role": "assistant", 
      "content": "Hello! I received your test message...",
      "timestamp": ISODate("2025-09-19T14:37:28.635Z")
    }
  ],
  "last_updated": ISODate("2025-09-19T14:37:28.635Z")
}
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install pymongo==4.8.0
```

### 2. Start MongoDB

```bash
# Start MongoDB on port 27018 (avoiding conflicts with default port)
mongod --dbpath ~/mongodb_data --port 27018 --bind_ip 127.0.0.1 --logpath ~/mongodb.log --fork
```

### 3. Verify MongoDB is Running

```bash
# Check MongoDB processes
ps aux | grep mongod

# Test connection (optional)
python test_mongodb_integration.py
```

### 4. Run the Voice Chatbot

```bash
python streaming_voice_chatbot_connect_mongo.py
```

## MongoDB Operations

### View Stored Conversations

Using Python:
```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27018/')
db = client['chatbot_conversations']
collection = db['conversations']

# Get all conversations
for conv in collection.find():
    print(f"Conversation {conv['conversation_id']}: {len(conv['messages'])} messages")

client.close()
```

### Query Specific Conversations

```python
# Find conversations with specific content
conversations = collection.find({
    "messages.content": {"$regex": "hello", "$options": "i"}
})

# Find recent conversations
import datetime
recent = collection.find({
    "started_at": {"$gte": datetime.datetime.now() - datetime.timedelta(days=1)}
}).sort("started_at", -1)
```

## Benefits

1. **Conversation History**: Review past interactions for analysis or debugging
2. **Data Analytics**: Analyze conversation patterns and user behavior
3. **Backup & Recovery**: Conversations are safely stored and can be recovered
4. **Multi-Session Support**: Access conversation history across different sessions
5. **Scalability**: MongoDB provides robust scaling options for large conversation volumes

## MongoDB Management

### Backup Conversations

```bash
# Export conversations to JSON
mongoexport --port 27018 --db chatbot_conversations --collection conversations --out conversations_backup.json
```

### Restore Conversations

```bash
# Import conversations from JSON
mongoimport --port 27018 --db chatbot_conversations --collection conversations --file conversations_backup.json
```

### Clear All Conversations

```python
# WARNING: This deletes all stored conversations
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27018/')
db = client['chatbot_conversations']
db['conversations'].delete_many({})
client.close()
```

## Troubleshooting

### MongoDB Connection Issues

1. **Check if MongoDB is running**:
   ```bash
   ps aux | grep mongod
   ```

2. **Check port availability**:
   ```bash
   netstat -an | grep 27018
   ```

3. **Review MongoDB logs**:
   ```bash
   tail -f ~/mongodb.log
   ```

### Common Issues

- **Port already in use**: Change the port in both MongoDB startup and the script configuration
- **Permission errors**: Ensure the MongoDB data directory is writable
- **Connection timeouts**: Check firewall settings and MongoDB bind IP configuration

## Performance Considerations

- MongoDB automatically indexes the `_id` field
- Consider adding indexes for frequently queried fields:
  ```python
  collection.create_index("conversation_id")
  collection.create_index("started_at")
  ```

- For high-volume usage, consider MongoDB replica sets for redundancy
- Monitor disk space usage in the MongoDB data directory