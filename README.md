# SMAI Backend

Backend de IA para integrar Groq con TheSickMenu.

## Funciones

- \`POST /ai\`: responde preguntas usando Groq.
- Browser Search opcional en cada petición.
- Respuestas moderadas por \`openai/gpt-oss-safeguard-20b\`; si la moderación falla o marca contenido inapropiado, la API devuelve un error y no entrega un mensaje.
- Formato fijo: \`(AI): <mensaje> - Comando: /ai\`.
- Máximo de 100 caracteres en el mensaje completo, sin cortar palabras.
- Elimina emojis de la respuesta antes de entregarla.
- Memoria JSON separada por \`player_id\`, con campos permitidos: \`level\`, \`device\`, \`game\` y \`notes\`.
- Historial corto enviado por el cliente.
- Autenticación mediante \`X-Bot-Token\`.

## Variables de entorno

| Variable | Obligatoria | Descripción |
| --- | --- | --- |
| \`GROQ_API_KEY\` | Sí | API key de Groq |
| \`BOT_API_TOKEN\` | Sí | Secreto propio para autenticar el mod |
| \`GROQ_MODEL\` | No | Por defecto \`openai/gpt-oss-20b\` |
| \`PLAYER_MEMORY_FILE\` | No | Ruta del JSON de memoria; por defecto \`data/player_memories.json\` |

No subas archivos \`.env\`, claves, tokens ni el JSON de memoria al repositorio.

## Despliegue en Render

- Runtime: Python
- Build command: \`pip install -r requirements.txt\`
- Start command: \`uvicorn main:app --host 0.0.0.0 --port $PORT\`
- Health check path: \`/health\`

### Persistencia de memoria en Render

La memoria se guarda en un archivo JSON local. En Render Free, el sistema de archivos del servicio es efímero: el archivo puede perderse al reiniciar o desplegar. Para conservarla entre despliegues, configura un disco persistente en el servicio y establece \`PLAYER_MEMORY_FILE\` en una ruta dentro de su punto de montaje, por ejemplo \`/var/data/player_memories.json\`. El disco persistente requiere un plan de Render que lo admita.

No se guardan conversaciones completas; solo los campos de perfil enviados por el cliente.

## API

### Comprobar estado

\`GET /health\`

### Preguntar a la IA

\`POST /ai\`

Headers:

\`\`\`http
Content-Type: application/json
X-Bot-Token: TU_BOT_API_TOKEN
\`\`\`

Body:

\`\`\`json
{
  "player_id": "player_123",
  "question": "¿Qué novedades hay sobre Geometry Dash?",
  "history": [],
  "player_memory": {
    "level": "Through The Decay",
    "device": "PC",
    "game": "Geometry Dash"
  },
  "web_search": true
}
\`\`\`

\`player_id\` debe ser un identificador estable y opaco para cada jugador. El mod debe enviar el mismo identificador en cada petición. \`player_memory\` es opcional; los campos no enviados conservan su valor anterior.

Respuesta:

\`\`\`json
{
  "answer": "(AI): ¡Puedo buscar novedades de Geometry Dash! - Comando: /ai",
  "model": "openai/gpt-oss-20b",
  "web_search": true
}
\`\`\`

Si la moderación bloquea la respuesta, la API devuelve HTTP 422 con el código \`AI_RESPONSE_REJECTED\`. El cliente debe enviar el mensaje al chat únicamente cuando reciba HTTP 200. Si el filtro no está disponible, se devuelve HTTP 503 y tampoco se envía el mensaje.

## Coherencia de conversación

La pregunta actual y el historial reciente tienen prioridad. Los datos de perfil guardados solo se usan cuando son relevantes; por ejemplo, el nivel del jugador no se menciona al responder una pregunta de matemáticas.

## Seguridad

- Nunca incluyas \`GROQ_API_KEY\` en el código del mod.
- \`BOT_API_TOKEN\` también debe mantenerse privado. Un secreto incluido dentro de una DLL distribuida puede extraerse; no debe considerarse protección absoluta.
- Usa HTTPS y rota los secretos si se exponen.
- La moderación automática reduce el riesgo, pero no garantiza detectar absolutamente todo contenido inapropiado.
- Evita guardar datos personales reales en la memoria del jugador.
