/* ─── State ──────────────────────────────────────────────────────────────── */
const THREAD_STORAGE_KEY = 'sidekick-thread-id';

const state = {
  threadId: localStorage.getItem(THREAD_STORAGE_KEY) || null,
  isProcessing: false,
  pendingFiles: [],
};

/* ─── DOM References ─────────────────────────────────────────────────────── */
const chatArea       = document.getElementById('chatArea');
const emptyState     = document.getElementById('emptyState');
const statusBar      = document.getElementById('statusBar');
const statusText     = document.getElementById('statusText');
const msgInput       = document.getElementById('msgInput');
const criteriaInput  = document.getElementById('criteriaInput');
const sendBtn        = document.getElementById('sendBtn');
const resetBtn       = document.getElementById('resetBtn');
const themeToggle    = document.getElementById('themeToggle');
const fileInput      = document.getElementById('fileInput');
const fileList       = document.getElementById('fileList');

/* ─── Theme ──────────────────────────────────────────────────────────────── */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeToggle.textContent = theme === 'light' ? '🌙' : '☀️';
  themeToggle.title = theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode';
  localStorage.setItem('sidekick-theme', theme);
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

// Restore theme on load
applyTheme(localStorage.getItem('sidekick-theme') || 'dark');

/* ─── Status Bar ─────────────────────────────────────────────────────────── */
function setStatus(text, active = true) {
  statusText.textContent = text;
  statusBar.classList.toggle('active', active);
}

function clearStatus() {
  setStatus('Ready', false);
}

/* ─── Chat Rendering ─────────────────────────────────────────────────────── */
function hideEmptyState() {
  if (emptyState) emptyState.style.display = 'none';
}

function renderMessage(role, content) {
  const group = document.createElement('div');
  group.className = 'message-group';

  const roleLabel = document.createElement('div');
  roleLabel.className = 'message-role';
  roleLabel.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'Sidekick' : 'Evaluator';

  const bubble = document.createElement('div');
  bubble.className = `message message-${role}`;
  bubble.textContent = content;

  group.appendChild(roleLabel);
  group.appendChild(bubble);
  chatArea.appendChild(group);
  chatArea.scrollTop = chatArea.scrollHeight;
  return bubble;
}

function renderFileChips() {
  fileList.innerHTML = '';
  for (const file of state.pendingFiles) {
    const chip = document.createElement('span');
    chip.className = 'file-chip';
    chip.textContent = file.name;
    fileList.appendChild(chip);
  }
}

async function uploadPendingFiles() {
  if (!state.pendingFiles.length) return;
  const formData = new FormData();
  for (const file of state.pendingFiles) {
    formData.append('files', file);
  }
  if (state.threadId) {
    formData.append('thread_id', state.threadId);
  }

  setStatus('📎 Uploading files...', true);
  const res = await fetch('/api/upload', { method: 'POST', body: formData });
  if (!res.ok) throw new Error(`File upload failed (${res.status})`);
  const data = await res.json();
  state.threadId = data.thread_id || state.threadId;
  if (state.threadId) localStorage.setItem(THREAD_STORAGE_KEY, state.threadId);
  const chunksAdded = data.chunks_added || 0;
  const ignoredFiles = data.ignored_files || [];
  if (chunksAdded === 0) {
    // Keep chips so the user can switch to file types that yield extractable text.
    const details = ignoredFiles
      .slice(0, 3)
      .map((f) => `${f.filename}: ${f.reason}`)
      .join(' | ');
    setStatus(
      ignoredFiles.length
        ? `⚠️ Uploaded files ignored (no text to index): ${details || ''}`
        : '⚠️ No extractable text from uploaded files.',
      true
    );
    renderFileChips();
    return;
  }

  state.pendingFiles = [];
  renderFileChips();
  setStatus(`📚 Uploaded ${data.files?.length || 0} file(s)`, true);
}

/* ─── Fetch + SSE Chat ───────────────────────────────────────────────────── */
async function sendMessage() {
  const message = msgInput.value.trim();
  const criteria = criteriaInput.value.trim();

  if (!message || state.isProcessing) return;
  if (!criteria) {
    criteriaInput.focus();
    criteriaInput.style.borderColor = 'var(--btn-danger-text)';
    setTimeout(() => (criteriaInput.style.borderColor = ''), 1500);
    setStatus('⚠️ Please provide success criteria.', true);
    setTimeout(clearStatus, 2000);
    return;
  }

  // UI Lock
  state.isProcessing = true;
  sendBtn.disabled = true;
  fileInput.disabled = true;
  msgInput.disabled = true;
  criteriaInput.disabled = true;

  hideEmptyState();
  renderMessage('user', message);
  msgInput.value = '';
  criteriaInput.value = '';
  setStatus('🚀 Starting...', true);

  try {
    await uploadPendingFiles();
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        success_criteria: criteria,
        thread_id: state.threadId,
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    let liveAssistantBubble = null;

    const handleEventBlock = (block) => {
      if (!block.trim()) return;
      const lines = block.split('\n');
      let eventType = 'message';
      const dataLines = [];

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim());
        }
      }

      if (!dataLines.length) return;

      let payload;
      try {
        payload = JSON.parse(dataLines.join('\n'));
      } catch {
        return;
      }

      if (eventType === 'meta') {
        state.threadId = payload.thread_id;
        if (state.threadId) localStorage.setItem(THREAD_STORAGE_KEY, state.threadId);

      } else if (eventType === 'status') {
        setStatus(payload.text, true);

      } else if (eventType === 'assistant') {
        if (!liveAssistantBubble) {
          liveAssistantBubble = renderMessage('assistant', payload.text || '(No response)');
        } else {
          liveAssistantBubble.textContent = payload.text || '(No response)';
        }
        chatArea.scrollTop = chatArea.scrollHeight;

      } else if (eventType === 'done') {
        state.threadId = payload.thread_id;
        if (state.threadId) localStorage.setItem(THREAD_STORAGE_KEY, state.threadId);
        setStatus('✅ Done!', true);

        const finalAssistant = payload.assistant || '(No response)';
        if (liveAssistantBubble) {
          liveAssistantBubble.textContent = finalAssistant;
        } else {
          renderMessage('assistant', finalAssistant);
        }
        if (payload.evaluator) {
          renderMessage('evaluator', payload.evaluator);
        }
        setTimeout(clearStatus, 2000);

      } else if (eventType === 'error') {
        setStatus(`❌ Error: ${payload.message}`, true);
        renderMessage('assistant', `⚠️ Error: ${payload.message}`);
        setTimeout(clearStatus, 3000);
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r/g, '');
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';

      for (const block of blocks) {
        handleEventBlock(block);
      }
    }

    // Handle any final complete block left in buffer.
    if (buffer.includes('\n\n')) {
      const blocks = buffer.split('\n\n');
      for (const block of blocks) {
        handleEventBlock(block);
      }
    }

  } catch (err) {
    setStatus(`❌ Connection error: ${err.message}`, true);
    renderMessage('assistant', `⚠️ Network error: ${err.message}`);
    setTimeout(clearStatus, 3000);
  } finally {
    state.isProcessing = false;
    sendBtn.disabled = false;
    fileInput.disabled = false;
    msgInput.disabled = false;
    criteriaInput.disabled = false;
    msgInput.focus();
  }
}

/* ─── Reset ──────────────────────────────────────────────────────────────── */
async function resetSession() {
  if (state.isProcessing) return;
  try {
    const oldId = state.threadId;
    const url = oldId
      ? `/api/reset?thread_id=${encodeURIComponent(oldId)}`
      : '/api/reset';
    const res = await fetch(url, { method: 'POST' });
    const data = await res.json();
    state.threadId = data.thread_id;
    if (state.threadId) localStorage.setItem(THREAD_STORAGE_KEY, state.threadId);
  } catch {
    state.threadId = null;
    localStorage.removeItem(THREAD_STORAGE_KEY);
  }

  chatArea.innerHTML = '';
  chatArea.appendChild(emptyState);
  emptyState.style.display = '';
  msgInput.value = '';
  criteriaInput.value = '';
  state.pendingFiles = [];
  renderFileChips();
  clearStatus();
  msgInput.focus();
}

/* ─── Event Listeners ────────────────────────────────────────────────────── */
sendBtn.addEventListener('click', sendMessage);
resetBtn.addEventListener('click', resetSession);
fileInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files || []);
  state.pendingFiles = files;
  renderFileChips();
});

msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

criteriaInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textareas
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

msgInput.addEventListener('input', () => autoResize(msgInput));
criteriaInput.addEventListener('input', () => autoResize(criteriaInput));

// Init
clearStatus();
msgInput.focus();
