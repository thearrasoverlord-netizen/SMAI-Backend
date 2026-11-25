# SMAI Backend

Backend de IA para integrar Groq con TheSickMenu.

## Funciones iniciales

- `POST /ai`: responde preguntas usando Groq.
- Browser Search opcional en cada petición.
- Historial corto enviado por el cliente.
- Autenticación mediante el encabezado `X-Bot-Token`.
- La clave de Groq permanece en el servidor, nunca en la DLL.

## Variables de entorno

Configura estas variables en Render:

| Variable | Obligatoria | Descripción |
| --- | --- | --- |
| `GROQ_API_KEY` | Sí | API key de Groq |
| `BOT_API_TOKEN` | Sí | Secreto propio para autenticar el mod |
| `GROQ_MODEL` | No | Por defecto `openai/gpt-oss-20b` |

No subas archivos `.env`, claves ni tokens al repositorio.

## Despliegue en Render

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

## API

### Comprobar estado

`GET /health`

### Preguntar a la IA

`POST /ai`

Headers:

```http
Content-Type: application/json
X-Bot-Token: TU_BOT_API_TOKEN
```

Body:

```json
{
  "question": "¿Qué novedades hay sobre Geometry Dash?",
  "history": [],
  "web_search": true
}
```

Respuesta:

```json
{
  "answer": "Respuesta generada por Groq...",
  "model": "openai/gpt-oss-20b",
  "web_search": true
}
```

El historial se envía en cada petición y no se guarda permanentemente en este prototipo.

## Seguridad

- Nunca incluyas `GROQ_API_KEY` en el código del mod.
- `BOT_API_TOKEN` también debe mantenerse privado. Un secreto incluido dentro de una DLL distribuida puede extraerse; no debe considerarse protección absoluta.
- Usa HTTPS y rota los secretos si se exponen.
- Las preguntas enviadas al endpoint se procesan mediante Groq.
