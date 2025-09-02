from datetime import datetime, timedelta
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ---- SETTINGS (all of these come from Railway Variables) ----
SECRET_KEY = os.getenv("JWT_SECRET", "change-me")     # set in Railway
ALGORITHM = "HS256"
ACCESS_TOKEN_MIN = int(os.getenv("TOKEN_MIN", "180"))

API_USER = os.getenv("BASIC_USER", "admin")           # set in Railway
API_PASS = os.getenv("BASIC_PASS", "password")        # set in Railway

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "")
RESPONSES_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4o-mini")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://buildaitech.com").split(",")
]

# ---- APP + CORS ----
app = FastAPI(title="RegulAIte API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

# ---- Auth helpers ----
def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_MIN):
    expire = datetime.utcnow() + timedelta(minutes=minutes)
    to_encode = {**data, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_user(username: str, password: str) -> bool:
    return username == API_USER and password == API_PASS

def require_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") != API_USER:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# ---- OpenAI (Responses API) ----
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(question: str) -> str:
    # Uses your vector store if provided
    attachments = [{"vector_store_id": VECTOR_STORE_ID}] if VECTOR_STORE_ID else None
    resp = client.responses.create(
        model=RESPONSES_MODEL,
        input=question,
        attachments=attachments,
        temperature=0.2,
    )
    # SDK has .output_text helper:
    return resp.output_text

# ---- Schemas ----
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

# ---- Routes ----
@app.post("/api/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if not verify_user(form.username, form.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": API_USER})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, _: str = Depends(require_user)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    answer = ask_openai(payload.question)
    return AskResponse(answer=answer)

@app.get("/healthz")
def health():
    return {"ok": True}
