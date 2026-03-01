import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

# 1. Setup AIPIPE Client
# This pulls the token from your Render Environment Variables
token = os.getenv("AIPIPE_TOKEN")

# Fallback to prevent startup crash if token is missing
if not token:
    print("⚠️ WARNING: AIPIPE_TOKEN is not set in Environment Variables!")
    token = "missing_key"

client = OpenAI(
    base_url="https://aipipe.org/openai/v1", 
    api_key=token
)

# 2. Initialize FastAPI
app = FastAPI()

# 3. Add CORS Middleware (Crucial for the assignment website to reach your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows the assignment site to access your API
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)

# 4. Define the Data Models (The Schema)
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

class UserRequest(BaseModel):
    comment: str

# 5. The Analysis Endpoint
@app.post("/comment")
async def analyze_sentiment(request: UserRequest):
    try:
        # Call AIPIPE using the gpt-4o-mini model
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a sentiment analysis tool. "
                               "Analyze the user comment and return a JSON object with "
                               "'sentiment' (positive, negative, or neutral) and "
                               "'rating' (an integer from 1 to 5)."
                },
                {"role": "user", "content": request.comment}
            ],
            response_format=SentimentResponse,
        )

        # Extract the structured data
        return response.choices[0].message.parsed

    except Exception as e:
        # Handle errors gracefully as per requirements
        raise HTTPException(status_code=500, detail=f"AI Analysis Error: {str(e)}")

# 6. Optional Root Health Check (To see if the server is awake)
@app.get("/")
async def root():
    return {"status": "Server is running!", "endpoint": "/comment"}
