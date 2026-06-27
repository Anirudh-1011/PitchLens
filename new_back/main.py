from fastapi import FastAPI
from dotenv import load_dotenv
import os
import httpx
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

LANGFLOW_URL = os.getenv("LANGFLOW_URL")
FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")
class AnalyzeRequest(BaseModel):
    matchId: str = ""
    eventType: str
    player: str
    team: str
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
- Speak emotionally like a passionate fan.
- React as if watching the match live.
- Celebrate or criticize naturally.

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

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}",
            json={
                "input_value": prompt,
                "output_type": "chat",
                "input_type": "chat",
            },
            headers={
                "Authorization": f"Bearer {os.getenv('LANGFLOW_API_KEY')}"
            }
        )

    data = resp.json()
    print(resp.status_code)
    print(data)
    text = (
        data["outputs"][0]
        ["outputs"][0]
        ["results"]["message"]["text"]
    )
    print(prompt)

    return {
        "insight": text
    }
@app.get("/health")
def health():
    return {
        "status": "ok",
        "langflow_url": LANGFLOW_URL,
        "flow_id": FLOW_ID,
    }
@app.post("/test-flow")
async def test_flow(req: AnalyzeRequest):

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}",
            json={
                "input_value": req.question,
                "output_type": "chat",
                "input_type": "chat",
            },
            headers={
                "Authorization": f"Bearer {os.getenv('LANGFLOW_API_KEY')}"
            }
        )

    data = resp.json()

    text = (
        data["outputs"][0]
        ["outputs"][0]
        ["results"]["message"]["text"]
    )

    return {
        "answer": text
    }