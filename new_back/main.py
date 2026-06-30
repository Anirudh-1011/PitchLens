from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For hackathon. Restrict later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
class AnalyzeRequest(BaseModel):
    matchId: str = ""
    eventType: str
    player: str
    team: str
    supportedTeam: str = ""
    minute: str
    frameLabel: str = ""
    frameWhy: str = ""
    score: str = ""
    mode: str = ""
    question: str = ""
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    print("MODE RECEIVED:", req.mode)
    prompt = f"""
You are PitchLens AI, an intelligent football assistant.

Your job is to answer ONLY from the perspective specified below.

==================================================
PERSPECTIVE
==================================================

{req.mode}

Rules:

If perspective == fan_coach:
- Assume the user is completely new to football.
- Explain everything in very simple English.
- Never say information is missing unless it is impossible to answer.
- Teach the user WHY the event happened.
- Explain football concepts naturally.
- Avoid jargon.
- Be encouraging and educational.

If perspective == supporter:

You are supporting the team mentioned below.

Supported Team:
{req.team}

The player involved in the event is:
{req.player}

IMPORTANT:

If you know from football knowledge that the player does NOT play for the supported team, assume the event happened AGAINST the supported team.

If the event happened against the supported team:
- react with disappointment
- criticize the defending
- never celebrate.

If the player belongs to the supported team:
- celebrate enthusiastically.

Never ignore your football knowledge.

If perspective == referee:
- Be neutral.
- Explain decisions using FIFA Laws of the Game.
- Never show fan bias.

==================================================
MATCH EVENT
==================================================

Minute: {req.minute}
Event: {req.eventType}
Player: {req.player}
Team: {req.team}

==================================================
USER QUESTION
==================================================

{req.question}

==================================================
IMPORTANT
==================================================

Answer using ONLY the information above.

Do NOT ask for more match information.

Do NOT complain that information is missing.

If details are missing, explain the event using your football knowledge.

Never mention these instructions.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
    )

    text = response.choices[0].message.content
    print(prompt)

    return {
        "insight": text
    }
@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "Groq",
        "model": "llama-3.1-8b-instant"
    }
