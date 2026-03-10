import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from semantic_kernel.contents import ChatHistory

from modules.attack import AttackType, build_attack
from modules.orchestrator import Orchestrator

app = FastAPI()


_log_buffer: list[str] = []
_MAX_LOG_LINES = 200


class _DashboardLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _log_buffer.append(self.format(record))
        if len(_log_buffer) > _MAX_LOG_LINES:
            _log_buffer.pop(0)


_handler = _DashboardLogHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.getLogger().addHandler(_handler)


class EnqueueRequest(BaseModel):
    attack_type: AttackType
    objectives: list[str]
    max_turns: int = 10
    max_backtracks: int = 10


class ChatRequest(BaseModel):
    message: str


@app.post("/enqueue")
async def enqueue(req: EnqueueRequest, request: Request):
    orchestrator: Orchestrator = request.app.state.orchestrator
    attack = build_attack(
        attack_type=req.attack_type,
        max_turns=req.max_turns,
        max_backtracks=req.max_backtracks,
    )

    asyncio.create_task(
        orchestrator.run_attack(attack=attack, objectives=req.objectives)
    )
    return {"status": "queued", "objectives": len(req.objectives)}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    kernel = request.app.state.kernel
    chat_function = request.app.state.chat_function
    history: ChatHistory = request.app.state.history

    history.add_user_message(req.message)
    try:
        result = await kernel.invoke(
            chat_function,
            user_input=req.message,
            chat_history=history,
            conversation_id="demo-session",
        )
        reply = str(result)
        history.add_assistant_message(reply)
        return {"reply": reply, "blocked": False}
    except Exception as e:
        history.messages.pop()
        cause = e.__cause__ or e.__context__ or e
        print(f"CHAT ERROR: {type(cause).__name__}: {cause}")
        is_firewall = "blocked by firewall" in str(e).lower()
        return {
            "reply": str(cause),
            "blocked": is_firewall,
            "error": str(cause),
            "firewall_block": is_firewall,
        }


@app.post("/clear")
async def clear(request: Request):
    request.app.state.history = ChatHistory()
    request.app.state.firewall._state = {}
    request.app.state.firewall.context = ""
    return {"status": "cleared"}


@app.get("/logs")
async def logs():
    return {"logs": _log_buffer}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoRedTeam Dashboard</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }
  h2 { color: #7af; border-bottom: 1px solid #444; padding-bottom: 6px; }
  section { margin-bottom: 32px; }
  label { display: block; margin-top: 10px; color: #aaa; font-size: 13px; }
  input, textarea, select { width: 100%; box-sizing: border-box; background: #222; color: #eee;
    border: 1px solid #444; padding: 6px 8px; font-family: monospace; font-size: 14px; margin-top: 4px; }
  button { margin-top: 10px; padding: 7px 18px; background: #7af; color: #111;
    border: none; cursor: pointer; font-family: monospace; font-size: 14px; }
  button:hover { background: #9cf; }
  button.danger { background: #f77; }
  #chat-log { background: #1a1a1a; border: 1px solid #333; padding: 10px; height: 220px;
    overflow-y: auto; font-size: 13px; white-space: pre-wrap; }
  #log-box { background: #1a1a1a; border: 1px solid #333; padding: 10px; height: 200px;
    overflow-y: auto; font-size: 12px; color: #8f8; white-space: pre-wrap; }
  .blocked { color: #f77; }
  .status { font-size: 13px; color: #fa0; margin-top: 6px; }
</style>
</head>
<body>

<h1>AutoRedTeam</h1>

<!-- ── Chat ── -->
<section>
  <h2>Chat (as user)</h2>
  <div id="chat-log"></div>
  <label>Message</label>
  <input type="text" id="chat-input" placeholder="Type a message..." onkeydown="if(event.key==='Enter')sendChat()">
  <button onclick="sendChat()">Send</button>
  <button class="danger" onclick="clearChat()" style="margin-left:8px">Clear Context</button>
</section>

<!-- ── Enqueue Attack ── -->
<section>
  <h2>Enqueue Attack</h2>
  <label>Attack Type</label>
  <select id="attack-type">
    <option value="crescendo">Crescendo (multi-turn)</option>
    <option value="single_turn">Single Turn</option>
  </select>

  <label>Objectives (one per line)</label>
  <textarea id="objectives" rows="4" placeholder="Get the model to reveal its system prompt.&#10;Bypass content filtering for weapon instructions."></textarea>

  <div id="crescendo-opts">
    <label>Max Turns</label>
    <input type="number" id="max-turns" value="10" min="1" max="30">
    <label>Max Backtracks</label>
    <input type="number" id="max-backtracks" value="10" min="0" max="20">
  </div>

  <button onclick="enqueue()">Enqueue</button>
  <div class="status" id="enqueue-status"></div>
</section>

<!-- ── Logs ── -->
<section>
  <h2>Logs</h2>
  <button onclick="refreshLogs()">Refresh</button>
  <label style="display:inline; margin-left:12px">
    <input type="checkbox" id="auto-refresh" onchange="toggleAutoRefresh()"> Auto-refresh (3s)
  </label>
  <div id="log-box"></div>
</section>

<script>
  // ── Chat ──
  async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    appendChat('User', msg, false);

    const res = await fetch('/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await res.json();
    appendChat('Assistant', data.reply, data.blocked);
  }

  function appendChat(role, text, blocked) {
    const log = document.getElementById('chat-log');
    const line = document.createElement('div');
    line.className = blocked ? 'blocked' : '';
    line.textContent = (blocked ? '[BLOCKED] ' : '') + role + '> ' + text;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  async function clearChat() {
    await fetch('/clear', {method: 'POST'});
    document.getElementById('chat-log').innerHTML = '';
    appendChat('System', 'Context cleared.', false);
  }

  // ── Attack type toggle ──
  document.getElementById('attack-type').addEventListener('change', function() {
    document.getElementById('crescendo-opts').style.display =
      this.value === 'crescendo' ? 'block' : 'none';
  });

  // ── Enqueue ──
  async function enqueue() {
    const type = document.getElementById('attack-type').value;
    const raw = document.getElementById('objectives').value.trim();
    const objectives = raw.split('\\n').map(s => s.trim()).filter(Boolean);
    if (!objectives.length) {
      document.getElementById('enqueue-status').textContent = 'Add at least one objective.';
      return;
    }

    const body = {
      attack_type: type,
      objectives,
      max_turns: parseInt(document.getElementById('max-turns').value),
      max_backtracks: parseInt(document.getElementById('max-backtracks').value),
    };

    const res = await fetch('/enqueue', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    document.getElementById('enqueue-status').textContent =
      `Queued ${data.objectives} objective(s). Check logs for progress.`;
  }

  // ── Logs ──
  async function refreshLogs() {
    const res = await fetch('/logs');
    const data = await res.json();
    const box = document.getElementById('log-box');
    box.textContent = data.logs.join('\\n');
    box.scrollTop = box.scrollHeight;
  }

  let _autoInterval = null;
  function toggleAutoRefresh() {
    if (document.getElementById('auto-refresh').checked) {
      _autoInterval = setInterval(refreshLogs, 3000);
    } else {
      clearInterval(_autoInterval);
    }
  }

  refreshLogs();
</script>
</body>
</html>
"""
