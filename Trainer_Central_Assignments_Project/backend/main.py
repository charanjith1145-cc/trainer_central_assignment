#!/usr/bin/env python3
"""
Simple FastAPI backend for the Assignment Chatbot demo.

Flow:
  1. React frontend POSTs the conversation to /api/chat.
  2. This backend fetches fresh data from the Text Dashboard API
     (entries, learners, stats).
  3. That data + the conversation is handed to Claude, which answers
     the user's question (e.g. "how many times did Rohith submit the
     assignment") using only the fetched data.

Setup:
    pip install fastapi uvicorn anthropic requests python-dotenv
    (reuses the ANTHROPIC_API_KEY / TRAINER_API_KEY already in the
    project's .env file)

Run:
    uvicorn main:app --reload --port 8000
"""

import json
import os
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
TRAINER_API_KEY = os.environ.get("TRAINER_API_KEY")

MODEL = "claude-sonnet-5-5"
DASHBOARD_API_BASE = "https://trainer-central-assigments-api.onrender.com"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a TrainerCentral
assignment-submission dashboard.

You will be given the current dashboard data as JSON (entries, learners, stats) followed by
a conversation with a user. Answer the user's question using ONLY the data provided -
don't make anything up. Count, filter, and summarize the data yourself as needed
(e.g. counting how many entries mention a particular learner). Be concise and
conversational. If the data doesn't let you answer, say so plainly.
"""

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project root.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="Assignment Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


def fetch_dashboard_data() -> dict:
    headers = {"x-api-key": TRAINER_API_KEY} if TRAINER_API_KEY else {}
    data = {}

    endpoints = {
        "entries": ("/api/entries", {"limit": 500}),
        "learners": ("/api/learners", None),
        "stats": ("/api/stats", None),
    }
    for key, (path, params) in endpoints.items():
        try:
            resp = requests.get(f"{DASHBOARD_API_BASE}{path}", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data[key] = resp.json()
        except requests.RequestException as e:
            data[key] = {"error": str(e)}
    return data


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    dashboard_data = fetch_dashboard_data()

    system = SYSTEM_PROMPT + "\n\nCurrent dashboard data (JSON):\n" + json.dumps(dashboard_data, indent=2)

    claude_messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            messages=claude_messages,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}") from e

    reply = "".join(block.text for block in response.content if block.type == "text").strip()
    return {"reply": reply}
