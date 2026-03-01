import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

# 1. Setup AIPIPE Client
# AIPIPE uses the OpenAI format but points to a different "base_url"
client = OpenAI(
    base_url="https://aipipe.org/openai/v1", 
    api_key=os.getenv("AIPIPE_TOKEN") # Set this in Render!
)

app = FastAPI()

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

class UserRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_sentiment(request: UserRequest):
    try:
        # 2. Call the model (AIPIPE supports many, gpt-4o-mini is standard)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze sentiment. Return JSON with 'sentiment' and 'rating'."},
                {"role": "user", "content": request.comment}
            ],
            response_format=SentimentResponse,
        )

        return response.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AIPIPE Error: {str(e)}")
