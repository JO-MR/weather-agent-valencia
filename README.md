# Agente Meteorologico Valencia

API REST con interfaz web que consulta el tiempo en Valencia usando datos oficiales de AEMET y OpenAI GPT-4o como agente de lenguaje natural.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)
![AEMET](https://img.shields.io/badge/Datos-AEMET%20oficial-blue)

## Demo

Interfaz web accesible desde el navegador con:
- Temperatura y estado del tiempo actual
- Prevision para los proximos 5 dias
- Chat con agente IA en lenguaje natural

## Stack

| Tecnologia | Uso |
|------------|-----|
| FastAPI | API REST y servidor web |
| OpenAI GPT-4o | Agente de lenguaje natural |
| AEMET API | Datos meteorologicos oficiales |
| httpx | Peticiones HTTP asincronas |
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
curl -X POST https://tu-url.onrender.com/consulta \
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

# 5. Ejecuta
uvicorn main:app --reload
```

Abre `http://127.0.0.1:8000` en el navegador.

## Despliegue en Render

1. Sube el proyecto a GitHub
2. Crea cuenta en [render.com](https://render.com)
3. New → Web Service → conecta el repositorio
4. Añade las variables de entorno `OPENAI_API_KEY` y `AEMET_API_KEY`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Clic en Deploy

## Variables de entorno

| Variable | Descripcion | Donde obtenerla |
|----------|-------------|-----------------|
| `OPENAI_API_KEY` | API key de OpenAI | platform.openai.com/api-keys |
| `AEMET_API_KEY` | API key de AEMET | opendata.aemet.es |

## Estructura del proyecto

```
weather-agent-valencia/
├── static/
│   └── index.html      # Interfaz web
├── main.py             # API FastAPI + agente
├── requirements.txt    # Dependencias
├── render.yaml         # Configuracion Render
├── .gitignore
└── README.md
```

## Licencia

MIT
