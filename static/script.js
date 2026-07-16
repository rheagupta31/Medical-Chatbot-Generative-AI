const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const chips = document.getElementById("chips");

let sessionId = null;

const BOT_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2.5-6 4 12 2.5-6h5"/></svg>';

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeWelcome() {
  const welcome = chatWindow.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = role === "user" ? "You" : BOT_ICON;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatWindow.appendChild(msg);
  scrollToBottom();
  return { msg, bubble };
}

function addUserMessage(text) {
  const { bubble } = addMessage("user");
  bubble.textContent = text;
}

function addTypingIndicator() {
  const { msg, bubble } = addMessage("bot");
  bubble.innerHTML =
    '<span class="typing-dots"><span></span><span></span><span></span></span>';
  return msg;
}

function typewriterReveal(bubble, text, sources) {
  bubble.classList.add("typing-reveal");
  const durationMs = Math.min(2000, text.length * 14);
  const start = Date.now();

  const timer = setInterval(() => {
    // time-based so browser timer throttling can't stretch the animation
    const progress = Math.min(1, (Date.now() - start) / durationMs);
    const i = Math.round(progress * text.length);
    bubble.textContent = text.slice(0, i);
    scrollToBottom();

    if (i >= text.length) {
      clearInterval(timer);
      bubble.classList.remove("typing-reveal");
      if (sources && sources.length) {
        const wrap = document.createElement("div");
        wrap.className = "sources";
        for (const src of sources) {
          const pill = document.createElement("span");
          pill.className = "source-pill";
          pill.textContent = src.split("/").pop();
          wrap.appendChild(pill);
        }
        bubble.appendChild(wrap);
        scrollToBottom();
      }
    }
  }, 16);
}

function setBusy(busy) {
  chatInput.disabled = busy;
  sendBtn.disabled = busy;
  if (!busy) chatInput.focus();
}

async function sendMessage(message) {
  removeWelcome();
  addUserMessage(message);
  setBusy(true);

  const typing = addTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();
    sessionId = data.session_id;
    typing.remove();

    const { bubble } = addMessage("bot");
    typewriterReveal(bubble, data.answer, data.sources);
  } catch (err) {
    typing.remove();
    const { msg, bubble } = addMessage("bot");
    msg.classList.add("error");
    bubble.textContent = `Something went wrong: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  sendMessage(message);
});

if (chips) {
  chips.addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (chip) sendMessage(chip.textContent.trim());
  });
}
