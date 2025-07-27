import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId

# --- Environment Variables ---
# It's good practice to provide default values or clearly indicate required ones
# Existing AI_API_KEY for the default chat model
AI_API_KEY = os.getenv("AI_API_KEY")
if not AI_API_KEY:
    raise ValueError("AI_API_KEY environment variable not set.")

# New API Keys for specific models
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
if not GEMMA_API_KEY:
    raise ValueError("GEMMA_API_KEY environment variable not set.")

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
if not KIMI_API_KEY:
    raise ValueError("KIMI_API_KEY environment variable not set.")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME_CHAT = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528:free") # Renamed for clarity

# Specific models for summary and merging
MODEL_NAME_SUMMARIZATION = "google/gemma-3n-e2b-it:free"
MODEL_NAME_MERGING = "moonshotai/kimi-k2:free"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "test")

# --- MongoDB Setup ---
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# --- OpenAI Client Setup (using different clients for different APIs might be cleaner, but for OpenRouter, one client with different API keys can work if OpenRouter supports it this way) ---
# For OpenRouter, typically you'd just change the model name and the API key would be passed if different for each model.
# However, if OpenRouter requires a *different* API key for different *provider prefixes* (e.g., one for Google, one for MoonshotAI),
# then you might need separate OpenAI client instances. Assuming for now that a single base_url and changing api_key per request is acceptable.

# Client for the default chat model
openai_client_chat = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=AI_API_KEY
)

# Clients for specific tasks (if different API keys are needed, otherwise reuse openai_client_chat)
# If OpenRouter accepts different API keys for different models with the same base URL, these clients are good.
openai_client_summarization = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=GEMMA_API_KEY
)

openai_client_merging = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=KIMI_API_KEY
)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Microservice",
    description="Provides AI capabilities for chat, content summarization, and summary merging.",
    version="1.0.0"
)

# --- Pydantic Models for Request Bodies ---

class ChatRequest(BaseModel):
    """Model for the /chat endpoint request."""
    chatbotCode: str = Field(..., description="Unique code for the chatbot associated with a website.")
    chatId: str = Field(..., description="ID of the chat session in MongoDB.")
    prompt: str = Field(..., description="User's current message/query.")

class ScrapedData(BaseModel):
    """Model for the extracted data from a scraped page."""
    text: str = Field(..., description="All visible text extracted from the page.")
    forms: list = Field(default_factory=list, description="List of detected forms and their details.")
    buttons: list = Field(default_factory=list, description="List of detected buttons and their details.")
    path: str = Field(..., description="The path of the page being summarized (e.g., '/about-us').") # Added path field

class SummaryRequest(BaseModel):
    """Model for the /summary endpoint request."""
    data: ScrapedData = Field(..., description="The scraped data object to summarize.")

class MergeRequest(BaseModel):
    """Model for the /merge endpoint request."""
    freshSummary: str = Field(..., description="The newly generated summary from the current scrape.")
    previousSummary: str | None = Field(None, description="The previous AI summary for the website, if available.")
    scrapedData: ScrapedData = Field(..., description="The raw scraped data for context during merging.")


# --- AI Service Endpoints ---

@app.post("/summary", summary="Generate a summary from scraped data")
async def generate_summary(request: SummaryRequest):
    """
    Generates a concise summary from provided scraped web page data (text, forms, buttons).
    """
    if not request.data.text:
        raise HTTPException(status_code=400, detail="No text provided for summarization.")

    # Combine all scraped data into a single context for the AI
    scraped_content = f"Website Text:\n{request.data.text}\n\n"
    if request.data.forms:
        scraped_content += f"Forms Detected:\n{json.dumps(request.data.forms, indent=2)}\n\n"
    if request.data.buttons:
        scraped_content += f"Buttons Detected:\n{json.dumps(request.data.buttons, indent=2)}\n\n"

    messages_for_api = [
        {"role": "system", "content": f"You are a highly skilled summarization AI, that summarizes the info about pages of different websites. You do it for another AI, that will use this info to guide and help users. Your task is to extract the key information from the provided web page content and condense it into a concise, factual, and understandable summary. Focus on the main purpose and content of the page, including any interactive elements like forms or prominent buttons, and text. At the beginning of your summary include this path of the page, that is being summarized: {request.data.path}"},
        {"role": "user", "content": f"Summarize the following website content:\n\n{scraped_content}"}
    ]

    try:
        response = openai_client_summarization.chat.completions.create(
            model=MODEL_NAME_SUMMARIZATION, # Using GEMMA_API_KEY implicitly through openai_client_summarization
            messages=messages_for_api,
            max_tokens=400, # Limit summary length
            temperature=0.3, # Aim for less creative, more factual summaries
        )
        summary = response.choices[0].message.content.strip()
        print(f"Generated summary: {summary[:100]}...")
        return {"summary": summary}
    except Exception as e:
        print(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")

@app.post("/merge", summary="Merge fresh and previous summaries")
async def merge_summaries(request: MergeRequest):
    """
    Merges a fresh summary with a previous summary, providing the raw scraped data as context.
    Creates a new, comprehensive summary that incorporates updates or new information.
    """
    # Build the prompt for merging
    messages_for_api = [
        {"role": "system", "content": "You are an AI assistant specialized in creating and structuring information about websites for other AI, so they can be helpful to users. Your goal is to combine a 'fresh' summary of recent content with a 'previous' summary of older content related to the same website. Create a single, detailed, and updated summary. If the previous summary is empty, just refine the fresh summary."},
        {"role": "user", "content": f"Fresh Summary:\n{request.freshSummary}\n\n"}
    ]

    if request.previousSummary:
        messages_for_api.append({"role": "user", "content": f"Previous Summary:\n{request.previousSummary}\n\n"})

    # Add scraped data as additional context (can be useful for the merge model to understand the original content if needed)
    # This part of the prompt remains the same as requested, to avoid changing existing prompts.
    messages_for_api.append({"role": "user", "content": "Please provide a single, merged, and updated summary based on the provided information. If there's no previous summary, simply refine the fresh summary."})

    try:
        response = openai_client_merging.chat.completions.create(
            model=MODEL_NAME_MERGING, # Using KIMI_API_KEY implicitly through openai_client_merging
            messages=messages_for_api,
            max_tokens=2000, # Allow more tokens for merged summary
            temperature=0.5, # Slightly more creative for merging logic
        )
        merged_summary = response.choices[0].message.content.strip()
        print(f"Generated merged summary: {merged_summary[:100]}...")
        return {"mergedSummary": merged_summary}
    except Exception as e:
        print(f"Error merging summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to merge summaries: {e}")


# --- Existing Chat Endpoint ---
@app.post("/chat", summary="Handle AI chatbot conversations")
async def chat_endpoint(request: ChatRequest):
    """
    Processes a user's chat prompt for a specific chatbot on a website,
    fetches chat history, builds context, and generates an AI response.
    """
    # Step 1: Fetch Website by chatbotCode
    print(f"Api key: {'***' + AI_API_KEY[-4:] if AI_API_KEY else 'Not Set'}") # Mask API key in logs
    website = await db.websites.find_one({"chatbotCode": request.chatbotCode})
    if not website:
        print(f"Website not found for chatbotCode: {request.chatbotCode}")
        raise HTTPException(status_code=404, detail="Invalid chatbotCode")

    # Step 2: Fetch Chat by chatId
    try:
        chat_object_id = ObjectId(request.chatId)
        chat = await db.chats.find_one({"_id": chat_object_id})
    except Exception as e:
        print(f"Invalid chatId: {e}")
        raise HTTPException(status_code=400, detail="Invalid chatId")

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

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
        # The 'bot' role is typically for the AI assistant in OpenAI models
        if sender.lower() == 'user':
            filtered_history.append({"role": "user", "content": text})
        elif sender.lower() == 'bot': # Assuming 'bot' is the AI's previous responses
            filtered_history.append({"role": "assistant", "content": text})
        elif sender.lower() == "ai": # If 'ai' is also used for assistant
            filtered_history.append({"role": "assistant", "content": text})
        elif sender.lower() == "owner": # Owner messages often represent system/AI context
            filtered_history.append({"role": "assistant", "content": text})
        elif "staff" in sender.lower(): # Staff messages too
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

Collected Info on Website
{website.get('aiSummary', 'N/A')}
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
    - Wants product recomendations but there was no list of products provided

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
        response = openai_client_chat.chat.completions.create( # Use the client for chat
            model=MODEL_NAME_CHAT, # Use the model name for chat
            messages=messages_for_api,
            # Using default max_tokens and temperature from model if not specified,
            # or you can set them explicitly like in the summary/merge endpoints.
        )
        final_response = response.choices[0].message.content.strip()

        if "code:human007" in final_response.lower():
            final_response = "code:human007"

    except Exception as e:
        print(f"Error calling hosted model: {e}")
        # Provide a user-friendly message for AI errors
        final_response = "I'm sorry, I'm currently experiencing technical difficulties. Please try again later or contact our support team."
        raise HTTPException(status_code=500, detail=f"AI model error: {e}") # Raise HTTPException for proper API error response

    return {"response": final_response}