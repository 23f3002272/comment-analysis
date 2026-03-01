import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from google import genai
from google.genai import types

# 1. Setup the New Gemini Client
# It will automatically look for an environment variable named 'GEMINI_API_KEY'
# or you can pass it manually like this:
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# 2. Define the Schema (The "Shape" of your data)
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

class UserRequest(BaseModel):
    comment: str

# 3. Initialize FastAPI
app = FastAPI()

@app.post("/comment")
async def analyze_sentiment(request: UserRequest):
    try:
        # 4. Generate content using the new 'models' interface
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=f"Analyze the sentiment of this comment: {request.comment}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SentimentResponse, # Pass the class directly!
            ),
        )

        # 5. Magic Step: 'response.parsed' gives us the object directly!
        # No more manual JSON parsing needed.
        return response.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
