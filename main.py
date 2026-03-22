"""
Weather AI Agent API - Valencia
================================
API REST con FastAPI, autenticacion por API key,
rate limiting, LangSmith y Langfuse 4.0.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from langfuse import get_client
from langsmith import traceable
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv(dotenv_path=".env", override=True)

# ---------------------------------------------------------------------------
# Logging estructurado
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/agent.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("weather-agent")

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

VALENCIA_MUNICIPIO = "46250"
AEMET_ESTACION     = "8414A"
AEMET_BASE_URL     = "https://opendata.aemet.es/opendata/api"
MODEL              = "gpt-4o"

# Rate limiting: max peticiones por minuto por API key
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "10"))

_openai_key     = os.getenv("OPENAI_API_KEY")
_aemet_key      = os.getenv("AEMET_API_KEY")
_langsmith_key  = os.getenv("LANGCHAIN_API_KEY")
_langsmith_proj = os.getenv("LANGCHAIN_PROJECT", "weather-agent-valencia")
_langfuse_pub   = os.getenv("LANGFUSE_PUBLIC_KEY")
_langfuse_sec   = os.getenv("LANGFUSE_SECRET_KEY")
_langfuse_host  = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# API keys validas — separadas por coma en la variable de entorno
# Ejemplo: API_KEYS=key-abc123,key-def456,key-ghi789
_raw_api_keys = os.getenv("API_KEYS", "")
VALID_API_KEYS: set[str] = {
    k.strip() for k in _raw_api_keys.split(",") if k.strip()
}

if not _openai_key:
    _openai_key = ""
    logger.warning("OPENAI_API_KEY no configurada")
if not _aemet_key:
    _aemet_key = ""
    logger.warning("AEMET_API_KEY no configurada")
if not VALID_API_KEYS:
    logger.warning("API_KEYS no configurada — autenticacion desactivada")

# Activa LangSmith
if _langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = _langsmith_key
    os.environ["LANGCHAIN_PROJECT"]    = _langsmith_proj
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://eu.api.smith.langchain.com"
    logger.info("LangSmith activado — proyecto: %s", _langsmith_proj)
else:
    logger.warning("LANGCHAIN_API_KEY no configurada — LangSmith desactivado")

# Inicializa Langfuse 4.0
langfuse_enabled = False
if _langfuse_pub and _langfuse_sec:
    os.environ["LANGFUSE_PUBLIC_KEY"] = _langfuse_pub
    os.environ["LANGFUSE_SECRET_KEY"] = _langfuse_sec
    os.environ["LANGFUSE_HOST"]       = _langfuse_host
    langfuse_enabled = True
    logger.info("Langfuse 4.0 activado — host: %s", _langfuse_host)
else:
    logger.warning("LANGFUSE keys no configuradas — Langfuse desactivado")

client = AsyncOpenAI(api_key=_openai_key)

# ---------------------------------------------------------------------------
# Rate limiting en memoria
# ---------------------------------------------------------------------------

# Almacena {api_key: [(timestamp), ...]}
_request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(api_key: str) -> bool:
    """Devuelve True si la key ha superado el limite de peticiones por minuto."""
    now = time.time()
    window = 60.0
    timestamps = _request_log[api_key]

    # Elimina timestamps fuera de la ventana
    _request_log[api_key] = [t for t in timestamps if now - t < window]

    if len(_request_log[api_key]) >= RATE_LIMIT_RPM:
        return True

    _request_log[api_key].append(now)
    return False

# ---------------------------------------------------------------------------
# Autenticacion por API key
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Verifica que la API key sea valida."""
    if not VALID_API_KEYS:
        return "anonymous"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key requerida. Incluye el header X-API-Key en tu peticion.",
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="API key invalida.",
        )

    return api_key

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agente Meteorologico Valencia",
    description=(
        "API REST para consultar el tiempo en Valencia usando AEMET + OpenAI GPT-4o.\n\n"
        "**Autenticacion:** incluye el header `X-API-Key` en todas las peticiones.\n\n"
        "**Rate limit:** 10 peticiones por minuto por API key."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Middleware de logging por request
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start      = time.time()
    logger.info("REQUEST  [%s] %s %s", request_id, request.method, request.url.path)
    response   = await call_next(request)
    duration   = round((time.time() - start) * 1000, 1)
    logger.info("RESPONSE [%s] %s %s %dms", request_id, request.method, request.url.path, duration)
    return response

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    pregunta: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"pregunta": "Que tiempo hace ahora en Valencia?"},
                {"pregunta": "Cual es la prevision para manana?"},
                {"pregunta": "Necesito paraguas esta semana?"},
            ]
        }
    }


class WeatherResponse(BaseModel):
    respuesta: str
    pregunta:  str


class HealthResponse(BaseModel):
    status:  str
    version: str

# ---------------------------------------------------------------------------
# Herramientas AEMET
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Obtiene la observacion meteorologica actual en Valencia, Espana, "
                "desde la API oficial de AEMET."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Obtiene la prediccion oficial de AEMET para los proximos dias en Valencia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "description": "Numero de dias de prevision (1-7).",
                    }
                },
                "required": [],
            },
        },
    },
]


async def _aemet_fetch(endpoint: str) -> Any:
    headers = {"api_key": _aemet_key}
    async with httpx.AsyncClient(timeout=15.0, verify=False) as http:
        r1 = await http.get(f"{AEMET_BASE_URL}{endpoint}", headers=headers)
        r1.raise_for_status()
        meta = r1.json()
        if meta.get("estado") != 200:
            raise ValueError(f"AEMET error {meta.get('estado')}: {meta.get('descripcion')}")
        r2 = await http.get(meta["datos"], headers=headers)
        r2.raise_for_status()
        return json.loads(r2.content.decode("latin-1"))


async def get_current_weather() -> dict[str, Any]:
    logger.info("Consultando observacion actual AEMET — estacion %s", AEMET_ESTACION)
    data = await _aemet_fetch(f"/observacion/convencional/datos/estacion/{AEMET_ESTACION}")
    obs  = data[-1] if isinstance(data, list) else data
    return {
        "fuente":           "AEMET - Agencia Estatal de Meteorologia",
        "estacion":         obs.get("nombre", "Valencia/Aeropuerto"),
        "fecha":            obs.get("fint",   "-"),
        "temperatura":      f"{obs.get('ta',   '-')} C",
        "humedad":          f"{obs.get('hr',   '-')} %",
        "viento_velocidad": f"{obs.get('vv',   '-')} m/s",
        "viento_direccion": f"{obs.get('dv',   '-')} grados",
        "presion":          f"{obs.get('pres', '-')} hPa",
        "precipitacion":    f"{obs.get('prec', '0')} mm",
    }


async def get_weather_forecast(days: int = 3) -> dict[str, Any]:
    days = max(1, min(days, 7))
    logger.info("Consultando prevision AEMET — %d dias", days)
    data = await _aemet_fetch(f"/prediccion/especifica/municipio/diaria/{VALENCIA_MUNICIPIO}")
    forecast = []
    for dia in data[0]["prediccion"]["dia"][:days]:
        cielo_list = dia.get("estadoCielo", [])
        cielo = next(
            (c.get("descripcion", "-") for c in cielo_list if c.get("periodo") == "1"),
            cielo_list[0].get("descripcion", "-") if cielo_list else "-",
        )
        temp   = dia.get("temperatura", {})
        viento = (dia.get("viento") or [{}])[0]
        forecast.append({
            "fecha":              dia.get("fecha", "-")[:10],
            "temp_max":           f"{temp.get('maxima', '-')} C",
            "temp_min":           f"{temp.get('minima', '-')} C",
            "estado_cielo":       cielo,
            "prob_precipitacion": f"{(dia.get('probPrecipitacion') or [{}])[0].get('value', '0')} %",
            "viento":             f"{viento.get('velocidad', '-')} km/h - {viento.get('direccion', '-')}",
        })
    return {
        "fuente":     "AEMET - Agencia Estatal de Meteorologia",
        "municipio":  "Valencia capital",
        "dias":       days,
        "prediccion": forecast,
    }


async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "get_current_weather":
        result = await get_current_weather()
    elif tool_name == "get_weather_forecast":
        result = await get_weather_forecast(**tool_input)
    else:
        result = {"error": f"Herramienta desconocida: {tool_name}"}
    return json.dumps(result, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Bucle del agente
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Eres un asistente meteorologico oficial que usa datos de AEMET para Valencia, Espana. "
    "Responde siempre en espanol, de forma clara y amigable. "
    "Menciona siempre que los datos provienen de AEMET. "
    "Usa las herramientas disponibles para obtener datos reales antes de responder."
)


@traceable(name="weather-agent-run")
async def run_agent(user_message: str, api_key: str = "anonymous") -> str:
    logger.info("Agente iniciado — pregunta: %.80s", user_message)
    start        = time.time()
    tools_used:  list[str] = []
    total_tokens = 0

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    lf = get_client() if langfuse_enabled else None

    try:
        async def _run():
            nonlocal total_tokens
            while True:
                response = await client.chat.completions.create(
                    model=MODEL,
                    tools=TOOLS,
                    messages=messages,
                )
                choice = response.choices[0]

                if response.usage:
                    total_tokens += response.usage.total_tokens

                if choice.finish_reason == "stop":
                    duration = round(time.time() - start, 2)
                    logger.info(
                        "Agente finalizado — tools: %s | tokens: %d | tiempo: %ss",
                        tools_used, total_tokens, duration,
                    )
                    return choice.message.content or ""

                if choice.finish_reason == "tool_calls":
                    messages.append(choice.message)
                    for tool_call in choice.message.tool_calls:
                        tool_name  = tool_call.function.name
                        tool_input = json.loads(tool_call.function.arguments)
                        tools_used.append(tool_name)
                        logger.info("Tool call: %s | input: %s", tool_name, tool_input)
                        result = await execute_tool(tool_name, tool_input)
                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tool_call.id,
                            "content":      result,
                        })
                else:
                    break

            return "No se pudo obtener una respuesta del agente."

        if lf:
            with lf.start_as_current_observation(
                as_type="span",
                name="weather-agent-run",
                input={"pregunta": user_message},
            ) as span:
                result = await _run()
                span.update(
                    output={"respuesta": result},
                    metadata={
                        "tools_used":   tools_used,
                        "total_tokens": total_tokens,
                        "api_key":      api_key[:8] + "...",
                    },
                )
                return result
        else:
            return await _run()

    except Exception as e:
        logger.error("Error en el agente: %s", e)
        raise
    finally:
        if lf:
            lf.flush()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Sirve la interfaz web."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/consulta", response_model=WeatherResponse)
async def consulta(
    request: QuestionRequest,
    api_key: str = Security(verify_api_key),
):
    """Consulta meteorologica en lenguaje natural."""
    if check_rate_limit(api_key):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit superado. Maximo {RATE_LIMIT_RPM} peticiones por minuto.",
        )
    try:
        respuesta = await run_agent(request.pregunta, api_key=api_key)
        return WeatherResponse(respuesta=respuesta, pregunta=request.pregunta)
    except httpx.HTTPError as e:
        logger.error("Error AEMET: %s", e)
        raise HTTPException(status_code=503, detail=f"Error conectando con AEMET: {e}")
    except Exception as e:
        logger.error("Error interno: %s", e)
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@app.get("/tiempo/ahora", response_model=dict)
async def tiempo_ahora(api_key: str = Security(verify_api_key)):
    """Observacion actual de AEMET."""
    if check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit superado.")
    try:
        return await get_current_weather()
    except Exception as e:
        logger.error("Error observacion AEMET: %s", e)
        raise HTTPException(status_code=503, detail=f"Error obteniendo datos de AEMET: {e}")


@app.get("/tiempo/prevision/{dias}", response_model=dict)
async def prevision(
    dias: int = 3,
    api_key: str = Security(verify_api_key),
):
    """Prevision para los proximos N dias (1-7)."""
    if check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit superado.")
    try:
        return await get_weather_forecast(days=dias)
    except Exception as e:
        logger.error("Error prevision AEMET: %s", e)
        raise HTTPException(status_code=503, detail=f"Error obteniendo prevision de AEMET: {e}")