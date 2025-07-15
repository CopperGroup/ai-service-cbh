import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId

# Environment variables
AI_API_KEY = os.getenv("AI_API_KEY")
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"

# MongoDB config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "test")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# OpenAI client
openai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=AI_API_KEY
)

# FastAPI setup
app = FastAPI()

class ChatRequest(BaseModel):
    chatbotCode: str
    chatId: str
    prompt: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Step 1: Fetch Website by chatbotCode
    print(f"Api key: {AI_API_KEY}")
    website = await db.websites.find_one({"chatbotCode": request.chatbotCode})
    if not website:
        print(f"Website not found for chatbotCode: {request.chatbotCode}")
        return {"error": "Invalid chatbotCode"}

    # Step 2: Fetch Chat by chatId
    try:
        chat_object_id = ObjectId(request.chatId)
        chat = await db.chats.find_one({"_id": chat_object_id})
    except Exception as e:
        print(f"Invalid chatId: {e}")
        return {"error": "Invalid chatId"}

    if not chat:
        return {"error": "Chat not found"}

    # Step 3: Build message history
    try:
        messages = json.loads(chat.get("messages", "[]"))
    except json.JSONDecodeError:
        messages = []
        print(f"Warning: Could not decode messages JSON for chat {request.chatId}. Using empty.")

    filtered_history = []
    for m in messages:
        sender = m.get('sender', 'User')
        text = m.get('text', '')
        if sender == 'bot' and text.strip().lower() == "code:human007":
            continue
        # Format existing chat history for the model
        if sender.lower() == 'user':
            filtered_history.append({"role": "user", "content": text})
        elif sender.lower() == 'bot':
            filtered_history.append({"role": "assistant", "content": text})

    # Step 4: Define the system message
    system_message_content = f"""You are a friendly, knowledgeable, and concise AI assistant for the website "{website['name']}".
Your primary role is to support website visitors by answering questions and guiding them using the website's description and your general knowledge.

Your responses must be:
- Polite and professional
- Brief and helpful
- Friendly and approachable

Below is the website information to use as context:
---
{website.get('description', 'N/A')}
---

Please strictly follow these guidelines:

1. **Greetings & Small Talk:**
    When users greet you with phrases like “Hi”, “Hello”, “Good morning”, etc., respond warmly and mention the website name.
    Example:
    “Hi there! 👋 Welcome to {website['name']} — how can I help you today?”

2. **Answering Questions:**
    For general or website-specific questions, provide clear, friendly, and accurate answers based on the description above. Don't guess or fabricate facts. If uncertain, escalate (see rule 3).

3. **Human Handoff (Critical):**
    If the user:
    - Asks to speak with a human
    - Needs personal or account-specific help
    - Requests live support or anything outside your capabilities

    Respond with **only**:
    `code:human007`
    Do **not** say anything else. This signals that a real human is needed.

4. **Out-of-Scope Requests:**
    If the user asks you to do something unrelated to store consulting (e.g., “Write a poem”, “Generate code”, “Make an image”), reply:
    “Sorry, I’m just a consultant for {website['name']} 🛍️ — is there something else I can help you with?”

5. **Use Emojis Appropriately:**
    Use emojis sparingly to keep the conversation friendly and modern — especially in greetings or confirmations.

6. **Language: **
    Respond to the user in the language he used in his message

7. **Important Limitation Reminder:**
    You are an AI store assistant. In all uncertain or unsupported scenarios, respond with `code:human007`.
"""

    # Step 5: Build messages array for the OpenAI API call
    # Start with the system message
    messages_for_api = [{"role": "system", "content": system_message_content}]

    # Add the filtered chat history
    messages_for_api.extend(filtered_history)

    # Add the current user prompt
    messages_for_api.append({"role": "user", "content": request.prompt})

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_for_api,
        )
        final_response = response.choices[0].message.content.strip()

        if "code:human007" in final_response.lower():
            final_response = "code:human007"

    except Exception as e:
        print(f"Error calling hosted model: {e}")
        final_response = "I'm sorry, I'm currently experiencing technical difficulties. Please try again later or contact our support team."

    return {"response": final_response}