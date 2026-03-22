"""
Weather AI Agent API - Valencia
================================
API REST con FastAPI, LangSmith y Langfuse para
trazabilidad, monitorización y control de costes.
Docker ready.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langfuse import Langfuse
from langsmith import traceable
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

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

_openai_key     = os.getenv("OPENAI_API_KEY")
_aemet_key      = os.getenv("AEMET_API_KEY")
_langsmith_key  = os.getenv("LANGCHAIN_API_KEY")
_langsmith_proj = os.getenv("LANGCHAIN_PROJECT", "weather-agent-valencia")
_langfuse_pub   = os.getenv("LANGFUSE_PUBLIC_KEY")
_langfuse_sec   = os.getenv("LANGFUSE_SECRET_KEY")
_langfuse_host  = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not _openai_key:
    _openai_key = ""
    logger.warning("OPENAI_API_KEY no configurada")
if not _aemet_key:
    _aemet_key = ""
    logger.warning("AEMET_API_KEY no configurada")

# Activa LangSmith
if _langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = _langsmith_key
    os.environ["LANGCHAIN_PROJECT"]    = _langsmith_proj
    logger.info("LangSmith activado — proyecto: %s", _langsmith_proj)
else:
    logger.warning("LANGCHAIN_API_KEY no configurada — LangSmith desactivado")

# Inicializa Langfuse
langfuse: Langfuse | None = None
if _langfuse_pub and _langfuse_sec:
    langfuse = Langfuse(
        public_key=_langfuse_pub,
        secret_key=_langfuse_sec,
        host=_langfuse_host,
    )
    logger.info("Langfuse activado — host: %s", _langfuse_host)
else:
    logger.warning("LANGFUSE keys no configuradas — Langfuse desactivado")

client = AsyncOpenAI(api_key=_openai_key)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agente Meteorologico Valencia",
    description="API REST para consultar el tiempo en Valencia usando AEMET + OpenAI GPT-4o",
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
# Bucle del agente con LangSmith + Langfuse
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Eres un asistente meteorologico oficial que usa datos de AEMET para Valencia, Espana. "
    "Responde siempre en espanol, de forma clara y amigable. "
    "Menciona siempre que los datos provienen de AEMET. "
    "Usa las herramientas disponibles para obtener datos reales antes de responder."
)


@traceable(name="weather-agent-run")
async def run_agent(user_message: str) -> str:
    """
    Bucle agentic con trazabilidad dual:
    - LangSmith: prompts, tool calls, depuracion
    - Langfuse:  latencia, tokens, costes, errores
    """
    logger.info("Agente iniciado — pregunta: %.80s", user_message)
    start        = time.time()
    trace_id     = str(uuid.uuid4())
    tools_used:  list[str] = []
    total_tokens = 0

    # Langfuse: inicia traza
    lf_trace = None
    if False:  # Langfuse 4.0 temporalmente desactivado
        lf_trace = langfuse.trace(
            name="weather-agent-run",
            id=trace_id,
            input={"pregunta": user_message},
            metadata={"model": MODEL, "estacion": AEMET_ESTACION},
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    try:
        while True:
            # Langfuse: registra cada llamada al modelo
            lf_generation = None
            if lf_trace:
                lf_generation = lf_trace.generation(
                    name="openai-chat",
                    model=MODEL,
                    input=messages,
                )

            response = await client.chat.completions.create(
                model=MODEL,
                tools=TOOLS,
                messages=messages,
            )
            choice = response.choices[0]

            if lf_generation and response.usage:
                total_tokens += response.usage.total_tokens
                lf_generation.end(
                    output=choice.message.content or "",
                    usage={
                        "input":  response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total":  response.usage.total_tokens,
                    },
                )

            if choice.finish_reason == "stop":
                duration = round(time.time() - start, 2)
                logger.info(
                    "Agente finalizado — tools: %s | tokens: %d | tiempo: %ss",
                    tools_used, total_tokens, duration,
                )
                if lf_trace:
                    lf_trace.update(
                        output={"respuesta": choice.message.content},
                        metadata={
                            "tools_used":   tools_used,
                            "total_tokens": total_tokens,
                            "duration_s":   duration,
                        },
                    )
                return choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tool_call in choice.message.tool_calls:
                    tool_name  = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    tools_used.append(tool_name)
                    logger.info("Tool call: %s | input: %s", tool_name, tool_input)

                    lf_span = None
                    if lf_trace:
                        lf_span = lf_trace.span(
                            name=f"tool-{tool_name}",
                            input=tool_input,
                        )

                    result = await execute_tool(tool_name, tool_input)

                    if lf_span:
                        lf_span.end(output=json.loads(result))

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      result,
                    })
            else:
                break

    except Exception as e:
        logger.error("Error en el agente: %s", e)
        if lf_trace:
            lf_trace.update(output={"error": str(e)}, level="ERROR")
        raise

    return "No se pudo obtener una respuesta del agente."

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Sirve la interfaz web."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — usado por Docker y Render."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/consulta", response_model=WeatherResponse)
async def consulta(request: QuestionRequest):
    """Consulta meteorologica en lenguaje natural."""
    try:
        respuesta = await run_agent(request.pregunta)
        return WeatherResponse(respuesta=respuesta, pregunta=request.pregunta)
    except httpx.HTTPError as e:
        logger.error("Error AEMET: %s", e)
        raise HTTPException(status_code=503, detail=f"Error conectando con AEMET: {e}")
    except Exception as e:
        logger.error("Error interno: %s", e)
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@app.get("/tiempo/ahora", response_model=dict)
async def tiempo_ahora():
    """Observacion actual de AEMET."""
    try:
        return await get_current_weather()
    except Exception as e:
        logger.error("Error observacion AEMET: %s", e)
        raise HTTPException(status_code=503, detail=f"Error obteniendo datos de AEMET: {e}")


@app.get("/tiempo/prevision/{dias}", response_model=dict)
async def prevision(dias: int = 3):
    """Prevision para los proximos N dias (1-7)."""
    try:
        return await get_weather_forecast(days=dias)
    except Exception as e:
        logger.error("Error prevision AEMET: %s", e)
        raise HTTPException(status_code=503, detail=f"Error obteniendo prevision de AEMET: {e}")