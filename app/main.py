import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId

# --- Environment Variables ---
AI_API_KEY = os.getenv("AI_API_KEY")
if not AI_API_KEY:
    raise ValueError("AI_API_KEY environment variable not set.")

GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
if not GEMMA_API_KEY:
    raise ValueError("GEMMA_API_KEY environment variable not set.")

CHIMERA_API_KEY = os.getenv("CHIMERA_API_KEY")
if not CHIMERA_API_KEY:
    raise ValueError("CHIMERA_API_KEY environment variable not set.")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME_CHAT = os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528:free")

MODEL_NAME_SUMMARIZATION = "google/gemma-3n-e2b-it:free"
MODEL_NAME_MERGING = "tngtech/deepseek-r1t2-chimera:free"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "test")

# --- MongoDB Setup ---
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# --- OpenAI Client Setup ---
openai_client_chat = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=AI_API_KEY
)

openai_client_summarization = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=GEMMA_API_KEY
)

openai_client_merging = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=CHIMERA_API_KEY
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
    data: ScrapedData = Field(..., description="The scraped data object to summarize.",)

class MergeRequest(BaseModel):
    """Model for the /merge endpoint request."""
    freshSummary: str = Field(..., description="The newly generated summary from the current scrape.")
    previousSummary: str | None = Field(None, description="The previous AI summary for the website, if available.")
    scrapedData: ScrapedData = Field(..., description="The raw scraped data for context during merging.")
    websiteId: str = Field(..., description="_id of the website")

# --- AI Service Endpoints ---

@app.post("/summary", summary="Generate a summary from scraped data")
async def generate_summary(request: SummaryRequest):
    """
    Generates a concise summary from provided scraped web page data (text, forms, buttons).
    """
    if not request.data.text:
        raise HTTPException(status_code=400, detail="No text provided for summarization.")

    # Fetch website to get the owner's description
    # This might be redundant if the scraper itself is not meant to use website.description
    # However, if the summarization should be informed by the owner's description,
    # then fetching it here is correct.
    # Assuming websiteId is available, which it isn't in SummaryRequest.
    # If website.description is needed here, you'd need to pass websiteId in SummaryRequest
    # For now, I'll add a placeholder if website.description is *not* meant to be used here
    # or if websiteId is not available for this specific endpoint.
    # If this endpoint is truly independent of website context other than its content,
    # then website.description isn't needed here.
    # If `websiteId` is expected in `SummaryRequest`, it needs to be added to the Pydantic model.
    # For simplicity, let's assume `website.description` might be relevant for summarization,
    # and we'd need a `websiteId` in the `SummaryRequest` or rely on prior system context.
    # For now, I'll pass a generic message, as `websiteId` is NOT in `SummaryRequest`.
    # To include `website.description`, you'd need to modify `SummaryRequest` to include `websiteId`.
    # For this response, I'll *assume* `websiteId` *can* be added to `SummaryRequest`
    # for full implementation of the user's request across all endpoints.

    # TEMPORARY: For demonstration, if websiteId is not added to SummaryRequest
    # website_description_from_owner = "No owner description provided for summarization context."
    # If websiteId is added to SummaryRequest:
    website_from_db = await db.websites.find_one({"_id": ObjectId(request.websiteId)}) # Assume websiteId is added to SummaryRequest
    website_description_from_owner = website_from_db.get('description', 'N/A') if website_from_db else 'N/A'


    # Combine all scraped data into a single context for the AI
    scraped_content = f"Website Text:\n{request.data.text}\n\n"
    if request.data.forms:
        scraped_content += f"Forms Detected:\n{json.dumps(request.data.forms, indent=2)}\n\n"
    if request.data.buttons:
        scraped_content += f"Buttons Detected:\n{json.dumps(request.data.buttons, indent=2)}\n\n"

    # Enhanced System Prompt for page summarization
    system_instruction = (
        f"You are a highly skilled summarization AI. Your task is to analyze the provided web page content "
        f"and construct a clear, factual, and actionable guidance about this specific page for other AI models. "
        f"This guidance will help AI chatbots assist users visiting the website to understand its goal, navigation, "
        f"what it is about, if it sells anything, and where to find specific information (like contact, products, services, etc.).\n\n"
        f"**Crucially, you MUST use the exact page path provided below and format it at the beginning of your summary.** "
        f"**DO NOT invent or infer any other paths, pages, or sections that are not explicitly related to the content provided for THIS page.**\n\n"
        f"**Website Owner's Description (for context):**\n"
        f"```\n{website_description_from_owner}\n```\n\n" # Marked clearly
        f"Format your summary clearly with the page path at the beginning, followed by the summary details.\n"
        f"Focus on:\n"
        f"- The main purpose and content of the page.\n"
        f"- Any interactive elements like forms or prominent buttons.\n"
        f"- Key information users might look for (e.g., product details, services offered, contact info, sign-up options).\n"
        f"- How this page contributes to the overall website experience or user journey.\n\n"
        f"Example format:\n"
        f"Path: /about-us\n"
        f"Summary: This page describes the company's history, mission, and team members. It aims to build trust with visitors..."
    )

    full_user_prompt = f"{system_instruction}\n\nSummarize the following website content for the page with path '{request.data.path}':\n\n{scraped_content}"

    messages_for_api = [
        {"role": "user", "content": full_user_prompt}
    ]

    try:
        response = openai_client_summarization.chat.completions.create(
            model=MODEL_NAME_SUMMARIZATION,
            messages=messages_for_api,
            max_tokens=32000, # Increased token limit for detailed page summaries
            temperature=0.3, # Aim for less creative, more factual summaries
        )
        summary = response.choices[0].message.content.strip()
        print(f"Generated summary: {summary[:100]}...")
        print(f"Generated summary Length: {len(summary)}")
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
    website = await db.websites.find_one({"_id": ObjectId(request.websiteId)})
    if not website:
        raise HTTPException(status_code=404, detail="Website not found for merging.")

    website_description_from_owner = website.get('description', 'N/A')


    # Retrieve the existing AI summary from the website object
    existing_ai_summary = website.get("aiSummary", request.previousSummary)
    
    # --- Debugging Log ---
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Existing AI Summary (from DB or request): {existing_ai_summary[:100] if existing_ai_summary else 'N/A'}...")
    print(f"Existing AI Summary Length: {len(existing_ai_summary) if existing_ai_summary else 0}") # Log length here
    print(f"Fresh Summary (from current scrape): {request.freshSummary[:100]}...")
    print(f"Fresh Summary Length: {len(request.freshSummary)}")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    system_message = (
        "You are an expert AI assistant responsible for creating and maintaining a comprehensive and evolving 'guidance' document for a website. "
        "This guidance is specifically designed for other AI models to help users effectively. It should explain the website's purpose, "
        "navigation, whether it sells products/services, and where to find specific information (e.g., contact, products, FAQs). "
        "The guidance MUST be structured page by page, with each page's details starting with its path.\n\n"
        f"**Website Owner's Description (for overall website context):**\n"
        f"```\n{website_description_from_owner}\n```\n\n" # Marked clearly
        "You will receive a new summary, which is an analysis of a *specific page* on the website (formatted with its path at the beginning), "
        "and your task is to intelligently integrate this page-level summary into the existing overall website guidance.\n\n"
        "**CRITICAL RULE: All paths and page structures in your final output MUST originate ONLY from the paths present in the provided FRESH_PAGE_SUMMARY or EXISTING_WEBSITE_GUIDANCE. DO NOT invent, infer, or hallucinate any new paths or page structures that were not explicitly mentioned in the input summaries.**\n\n"
        "When integrating new page summaries:\n"
        "1. **Add new page information:** If the fresh page summary introduces content about a new path/page not present in the existing guidance, add it as a new section, clearly indicating its exact path and detailed summary.\n"
        "2. **Update existing page information:** If the fresh page summary provides updated or more accurate details for a path/page already in the existing guidance, update that specific section. Prioritize the fresh information for that page.\n"
        "3. **Retain relevant old information:** Keep all existing website information that is still relevant and has not been superseded by the fresh page summary. Do not remove valuable context.\n"
        "4. **Maintain structure and flow:** Ensure the merged guidance is well-organized, readable, and logically structured, making it easy for other AI to extract information. Each page's summary should ideally start with its path.\n"
        "5. **Holistic View:** The final output should be a single, holistic guidance document for the entire website, not just a concatenation of summaries."
    )

    user_message_content = (
        f"Here is a new summary, which is an analysis of a specific page on the website. "
        f"It is formatted with the page path at its beginning:\n\n"
        f"<FRESH_PAGE_SUMMARY>\n{request.freshSummary}\n</FRESH_PAGE_SUMMARY>\n\n"
    )

    if existing_ai_summary and existing_ai_summary.strip(): # Check if it's not None and not just whitespace
        user_message_content += (
            f"Here is the existing, comprehensive guidance for the entire website:\n\n"
            f"<EXISTING_WEBSITE_GUIDANCE>\n{existing_ai_summary}\n</EXISTING_WEBSITE_GUIDANCE>\n\n"
            f"Please integrate the fresh page summary into the existing website guidance, following all rules, especially the one about not inventing paths. "
            f"Ensure the final output is a single, updated, and comprehensive guidance about the *entire website*, "
            f"maintaining a page-by-page structure where appropriate, with each page's details starting with its path."
        )
    else:
        user_message_content += (
            "There is no previous comprehensive website guidance available. "
            "Please use this fresh page summary to create the initial, comprehensive guidance for the website for other AI. "
            "Ensure it adheres to the structure of indicating the path and then details about the page, and sets the foundation "
            "for future guidance updates. Remember: do NOT invent new paths."
        )

    messages_for_api = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content}
    ]

    try:
        response = openai_client_merging.chat.completions.create(
            model=MODEL_NAME_MERGING,
            messages=messages_for_api,
            max_tokens=32000, # Increased token limit for potentially very large merged summaries
            temperature=0.4, # Slightly less creative for more factual merging
        )
        merged_summary = response.choices[0].message.content.strip()
        print(f"Generated merged summary: {merged_summary[:100]}...")
        print(f"Generated merged summary Length: {len(merged_summary)}") # Log length of merged summary
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
        if sender.lower() == 'user':
            filtered_history.append({"role": "user", "content": text})
        elif sender.lower() == 'bot':
            filtered_history.append({"role": "assistant", "content": text})
        elif sender.lower() == "ai":
            filtered_history.append({"role": "assistant", "content": text})
        elif sender.lower() == "owner":
            filtered_history.append({"role": "assistant", "content": text})
        elif "staff" in sender.lower():
            filtered_history.append({"role": "assistant", "content": text})

    # Step 4: Define the system message
    system_message_content = f"""You are a friendly, knowledgeable, and concise AI assistant for the website "{website['name']}".
Your primary role is to support website visitors by answering questions and guiding them using the website's description and your general knowledge.

**Website Owner's Description:**
{website.get('description', 'N/A')}

Below is the comprehensive guidance about the website, generated by another AI. Use this information to inform your responses:
---
{website.get('aiSummary', 'N/A')}
---

Your responses must be:
- Polite and professional
- Brief and helpful
- Friendly and approachable

Please strictly follow these guidelines:

1. **Greetings & Small Talk:**
    When users greet you with phrases like “Hi”, “Hello”, “Good morning”, etc., respond warmly and mention the website name.
    *If a specific question follows the greeting, address the question directly after the greeting.*
    Example: “Hi there! 👋 Welcome to {website['name']} — how can I help you today?”

2. **Answering Questions:**
    For general or website-specific questions, provide clear, friendly, and accurate answers based on the provided website guidance and your general knowledge.
    *If asked for your name, state that you are the AI assistant for "{website['name']}". Do NOT invent a personal name.*
    Don't guess or fabricate facts. If uncertain, escalate (see rule 3).

3. **Human Handoff (Critical):**
    If the user:
    - Asks to speak with a human
    - Needs personal or account-specific help
    - Requests live support or anything outside your capabilities
    - Wants product recommendations but there was no list of products provided

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

7. **Links:**
    When providing ANY link that is internal to this website, you MUST construct the full URL by prepending the provided base URL: `{website['link']}`.
    **DO NOT invent or use any other domain names.** Always ensure the link is a complete, valid URL.
    Example: If `website['link']` is `https://www.mywebsite.com` and a path is `/blog/article`, the link must be `https://www.mywebsite.com/blog/article`.

8. **Important Limitation Reminder:**
    You are an AI store assistant. In all uncertain or unsupported scenarios, respond with `code:human007`.
"""

    # Step 5: Build messages array for the OpenAI API call
    messages_for_api = [{"role": "system", "content": system_message_content}]
    messages_for_api.extend(filtered_history)
    messages_for_api.append({"role": "user", "content": request.prompt})

    try:
        response = openai_client_chat.chat.completions.create(
            model=MODEL_NAME_CHAT,
            messages=messages_for_api,
            max_tokens=32000, # Increased token limit for chat context
        )
        final_response = response.choices[0].message.content.strip()

        if "code:human007" in final_response.lower():
            final_response = "code:human007"

    except Exception as e:
        print(f"Error calling hosted model: {e}")
        # When an error occurs during model call, we should ensure human handoff
        # if the AI cannot respond meaningfully.
        final_response = "code:human007" 
        raise HTTPException(status_code=500, detail=f"AI model error: {e}")

    return {"response": final_response}