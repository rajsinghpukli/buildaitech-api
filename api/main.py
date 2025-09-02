# api/main.py
import os, secrets
from typing import List, Dict
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from openai import OpenAI

# ---------- settings ----------
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://buildaitech.com,https://www.buildaitech.com"
    ).split(",")
]

BASIC_USER = os.getenv("BASIC_AUTH_USER", "admin")
BASIC_PASS = os.getenv("BASIC_AUTH_PASS", "change-me")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)
# If you want to use your vector store later, read it from env:
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")  # optional for now

# ---------- app ----------
app = FastAPI(title="buildaitech-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

def require_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(credentials.username, BASIC_USER)
    ok_pass = secrets.compare_digest(credentials.password, BASIC_PASS)
    if not (ok_user and ok_pass):
        # www-auth header lets the browser show a login dialog if you want that UX
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username  # not used, but handy to return

class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []  # [{"role":"user"/"assistant","content":"..."}]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/login")
def login(_: str = Depends(require_basic_auth)):
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatRequest, _: str = Depends(require_basic_auth)):
    """
    Minimal call to OpenAI. Later we can switch to Responses API + file_search
    and attach your VECTOR_STORE_ID. For now, this returns a plain answer.
    """
    messages = [{"role": "system",
                 "content": "You are RegulAIte — a regulatory assistant. "
                            "Answer clearly, and use tables when helpful."}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.question})

    # Use a lightweight model; you can change to gpt-4o or your preferred one
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return {"answer": answer, "citations": []}
