import os
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from groq import AsyncGroq
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="SMAI Backend",
    description="Private AI backend for TheSickMenu.",
    version="0.1.0",
)

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_HISTORY_MESSAGES = 8
MAX_QUESTION_CHARS = 1500
MAX_HISTORY_CHARS = 6000
MAX_COMPLETION_TOKENS = 900

SYSTEM_PROMPT = (
    "Eres SMAI, un asistente de IA integrado en un chat de videojuego. "
    "Responde en el idioma del usuario, normalmente español. "
    "Sé útil, preciso y claro. Si usas búsqueda web, distingue los datos "
    "encontrados de las inferencias y no inventes fuentes. "
    "Tus respuestas se mostrarán en un chat con espacio limitado: "
    "prioriza respuestas concisas, pero conserva los datos importantes. "
    "No solicites contraseñas, tokens ni información personal sensible."
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AIRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    web_search: bool = True

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La pregunta no puede estar vacía.")
        return value

    @field_validator("history")
    @classmethod
    def limit_history_size(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        total = sum(len(message.content) for message in value)
        if total > MAX_HISTORY_CHARS:
            raise ValueError("El historial es demasiado largo.")
        return value


class AIResponse(BaseModel):
    answer: str
    model: str
    web_search: bool


def verify_bot_token(x_bot_token: str | None) -> None:
    expected = os.getenv("BOT_API_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="El backend todavía no tiene configurado BOT_API_TOKEN.",
        )
    if not x_bot_token or x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Token de bot inválido.")


@app.get("/")
async def root():
    return {"name": "SMAI Backend", "status": "online", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ai", response_model=AIResponse)
async def ask_ai(
    payload: AIRequest,
    x_bot_token: str | None = Header(default=None),
):
    verify_bot_token(x_bot_token)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Falta configurar GROQ_API_KEY.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": item.role, "content": item.content}
        for item in payload.history
    )
    messages.append({"role": "user", "content": payload.question})

    tools = [{"type": "browser_search"}] if payload.web_search else None

    try:
        client = AsyncGroq(api_key=api_key, timeout=45.0, max_retries=1)
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else "none",
            reasoning_effort="low",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        answer = completion.choices[0].message.content or ""
        answer = answer.strip()
        if not answer:
            raise HTTPException(status_code=502, detail="Groq devolvió una respuesta vacía.")

        return AIResponse(
            answer=answer,
            model=MODEL,
            web_search=payload.web_search,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose API keys or raw provider errors to the game client.
        print(f"Groq request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="No se pudo obtener una respuesta de Groq. Inténtalo nuevamente.",
        ) from exc
