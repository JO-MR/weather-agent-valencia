# Agente Meteorologico Valencia

API REST con interfaz web que consulta el tiempo en Valencia usando datos oficiales de AEMET y OpenAI GPT-4o como agente de lenguaje natural, con trazabilidad LLMOps via LangSmith y Langfuse.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)
![AEMET](https://img.shields.io/badge/Datos-AEMET%20oficial-blue)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![LangSmith](https://img.shields.io/badge/LangSmith-tracing-purple)
![Langfuse](https://img.shields.io/badge/Langfuse-monitoring-green)

## Demo

Interfaz web accesible desde el navegador con:
- Temperatura y estado del tiempo actual
- Prevision para los proximos 5 dias
- Chat con agente IA en lenguaje natural
- Trazabilidad completa con LangSmith y Langfuse

## Stack

| Tecnologia | Uso |
|------------|-----|
| FastAPI | API REST y servidor web |
| OpenAI GPT-4o | Agente de lenguaje natural |
| AEMET API | Datos meteorologicos oficiales |
| httpx | Peticiones HTTP asincronas |
| LangSmith | Trazabilidad de prompts y depuracion |
| Langfuse | Monitorización, tokens y costes |
| Docker | Contenedor portable |
| Render | Despliegue en produccion |

## Endpoints

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/` | Interfaz web |
| GET | `/health` | Health check |
| POST | `/consulta` | Pregunta en lenguaje natural |
| GET | `/tiempo/ahora` | Observacion actual AEMET |
| GET | `/tiempo/prevision/{dias}` | Prevision N dias (1-7) |

## Ejemplo de uso
```bash
curl -X POST https://weather-agent-valencia.onrender.com/consulta \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "Que tiempo hace ahora en Valencia?"}'
```

Respuesta:
```json
{
  "pregunta": "Que tiempo hace ahora en Valencia?",
  "respuesta": "Segun los datos de AEMET, en Valencia hay actualmente 19 C con cielo despejado..."
}
```

## Instalacion local
```bash
# 1. Clona el repositorio
git clone https://github.com/JO-MR/weather-agent-valencia.git
cd weather-agent-valencia

# 2. Crea el entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Crea el archivo .env
OPENAI_API_KEY=sk-...
AEMET_API_KEY=eyJ...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=weather-agent-valencia
LANGCHAIN_TRACING_V2=true

# 5. Ejecuta sin Docker
uvicorn main:app --host 127.0.0.1 --port 8000

# 6. O ejecuta con Docker
docker-compose up --build
```

Abre `http://127.0.0.1:8000` en el navegador.

## Despliegue en Render

1. Sube el proyecto a GitHub
2. Crea cuenta en [render.com](https://render.com)
3. New → Web Service → conecta el repositorio
4. Añade las variables de entorno
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Clic en Deploy

## Variables de entorno

| Variable | Descripcion | Donde obtenerla |
|----------|-------------|-----------------|
| `OPENAI_API_KEY` | API key de OpenAI | platform.openai.com/api-keys |
| `AEMET_API_KEY` | API key de AEMET | opendata.aemet.es |
| `LANGFUSE_PUBLIC_KEY` | Public key de Langfuse | cloud.langfuse.com |
| `LANGFUSE_SECRET_KEY` | Secret key de Langfuse | cloud.langfuse.com |
| `LANGFUSE_HOST` | Host de Langfuse | https://cloud.langfuse.com |
| `LANGCHAIN_API_KEY` | API key de LangSmith | smith.langchain.com |
| `LANGCHAIN_PROJECT` | Nombre del proyecto | weather-agent-valencia |
| `LANGCHAIN_TRACING_V2` | Activa trazabilidad | true |

## Estructura del proyecto
```
weather-agent-valencia/
├── static/
│   └── index.html      # Interfaz web
├── main.py             # API FastAPI + agente + LLMOps
├── Dockerfile          # Contenedor Docker
├── docker-compose.yml  # Orquestacion local
├── requirements.txt    # Dependencias
├── render.yaml         # Configuracion Render
├── runtime.txt         # Version de Python
├── .gitignore
└── README.md
```

## LLMOps

Cada consulta al agente queda registrada en:
- **LangSmith** → smith.langchain.com — prompts, tool calls, depuracion
- **Langfuse** → cloud.langfuse.com — tokens, costes, latencia, errores

## Licencia

MIT
