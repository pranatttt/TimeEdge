/* === STATE === */
let sessions = [];
let currentSessionId = null;
let abortController = null;

/* === DOM === */
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");
const chatContainer = document.getElementById("chatContainer");
const chatHistoryDiv = document.getElementById("chatHistory");
const sessionTitleBar = document.getElementById("sessionTitleBar");
const sessionTitle = document.getElementById("sessionTitle");

/* === CONSTANTS === */
const API_URL = "http://localhost:8000/chat";   // FIXED

/* === LOCAL STORAGE === */
function loadSessions() {
  const data = localStorage.getItem("sarimax_sessions");
  sessions = data ? JSON.parse(data) : [];
}
function saveSessions() {
  localStorage.setItem("sarimax_sessions", JSON.stringify(sessions));
}

/* === SESSION MGMT === */
function startNewSession() {
  const newSession = {
    id: Date.now().toString(),
    name: "New SARIMAX Forecast",
    messages: [],
  };
  sessions.unshift(newSession);
  currentSessionId = newSession.id;
  saveSessions();
  updateSidebar();
  renderSession();
}

function updateSidebar() {
  chatHistoryDiv.innerHTML = "";
  if (!sessions.length) {
    chatHistoryDiv.innerHTML = `<div class="chat-item" style="color:#999;">No sessions yet.</div>`;
    return;
  }
  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = "chat-item" + (s.id === currentSessionId ? " active" : "");
    item.textContent = s.name;
    item.onclick = () => {
      currentSessionId = s.id;
      updateSidebar();
      renderSession();
    };
    chatHistoryDiv.appendChild(item);
  });
}

/* === RENDER CHAT === */
function renderSession() {
  const session = sessions.find((s) => s.id === currentSessionId);
  sessionTitleBar.style.display = session ? "" : "none";
  sessionTitle.textContent = session ? session.name : "";
  chatContainer.innerHTML = "";

  if (!session || !session.messages.length) {
    chatContainer.innerHTML = `
      <div class="welcome-message">
        <h1 class="welcome-title">SARIMAX Forecasting Assistant</h1>
        <p class="welcome-subtitle">
          Ask me about sales trends, November 2012 revenue, forecast range, or seasonal insights.
        </p>
      </div>`;
    return;
  }
  session.messages.forEach((m) => addMessageToDOM(m.content, m.role));
}

/* === ADD MESSAGE TO DOM === */
function addMessageToDOM(content, type) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${type}`;

  const avatar = document.createElement("div");
  avatar.className = `message-avatar ${type}-avatar`;
  avatar.textContent = type === "user" ? "You" : "SMX";

  const body = document.createElement("div");
  body.className = "message-content";

  if (type === "assistant") body.innerHTML = content;
  else body.textContent = content;

  wrapper.append(avatar, body);
  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* === BUTTON ACTIONS === */
function showSendButton() {
  sendButton.textContent = "➤";
  sendButton.onclick = sendMessage;
}

function showStopButton() {
  sendButton.textContent = "■";
  sendButton.onclick = stopFetch;
}

/* === SEND MESSAGE === */
function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  let session = sessions.find((s) => s.id === currentSessionId);
  if (!session) {
    startNewSession();
    session = sessions.find((s) => s.id === currentSessionId);
  }

  session.messages.push({ role: "user", content: text });
  const loadingMsg = { role: "assistant", content: "⏳ Connecting to SARIMAX engine..." };
  session.messages.push(loadingMsg);

  chatInput.value = "";
  saveSessions();
  renderSession();
  showStopButton();

  fetchResponse(text, session.id, loadingMsg);
}

/* === FETCH RESPONSE === */
async function fetchResponse(userInput, sessionId, loadingMsg) {
  try {
    abortController = new AbortController();

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userInput }),
      signal: abortController.signal,
    });

    if (!res.ok) throw new Error("Server error");

    const data = await res.json();

    finalizeAssistantMsg(sessionId, loadingMsg, data.response || "⚠️ No response.");
  } catch (err) {
    finalizeAssistantMsg(
      sessionId,
      loadingMsg,
      "❌ Assistant offline. Ensure FastAPI is running on **localhost:8000**."
    );
  }
}

/* === STOP FETCH === */
function stopFetch() {
  if (abortController) abortController.abort();
  showSendButton();
}

/* === FINALIZE === */
function finalizeAssistantMsg(sessionId, loadingMsg, newContent) {
  const session = sessions.find((s) => s.id === sessionId);
  if (!session) return;
  loadingMsg.content = newContent;
  saveSessions();
  renderSession();
  showSendButton();
}

/* === INIT === */
loadSessions();
if (!sessions.length) startNewSession();
else {
  currentSessionId = sessions[0].id;
  updateSidebar();
  renderSession();
}
