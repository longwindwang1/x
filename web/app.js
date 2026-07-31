// Parley browser dev client — reference implementation of docs/protocol.md.

const MIC_RATE = 16000;   // wire format for mic audio
const FRAME_MS = 40;      // mic frame size sent to the server

const els = {
  character: document.getElementById('character'),
  connect: document.getElementById('connect'),
  mic: document.getElementById('mic'),
  status: document.getElementById('status'),
  transcript: document.getElementById('transcript'),
  metrics: document.querySelector('#metrics tbody'),
  sessioninfo: document.getElementById('sessioninfo'),
  textinput: document.getElementById('textinput'),
  send: document.getElementById('send'),
  interrupt: document.getElementById('interrupt'),
  miclevel: document.getElementById('miclevel'),
};

let ws = null;
let audioCtx = null;
let ttsRate = 24000;

// --- playback ---------------------------------------------------------------

let playhead = 0;
let liveSources = [];

function playChunk(arrayBuffer) {
  if (!audioCtx) return;
  const pcm = new Int16Array(arrayBuffer);
  if (!pcm.length) return;
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) floats[i] = pcm[i] / 32768;
  const buf = audioCtx.createBuffer(1, floats.length, ttsRate);
  buf.copyToChannel(floats, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const startAt = Math.max(audioCtx.currentTime + 0.05, playhead);
  src.start(startAt);
  playhead = startAt + buf.duration;
  liveSources.push(src);
  src.onended = () => { liveSources = liveSources.filter(s => s !== src); };
}

function stopPlayback() {
  for (const src of liveSources) { try { src.stop(); } catch (_) {} }
  liveSources = [];
  playhead = 0;
}

// --- microphone -------------------------------------------------------------

let micOn = false;
let micStream = null;
let micNode = null;
let sendBuf = new Float32Array(0);

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  await audioCtx.audioWorklet.addModule('worklet.js');
  const source = audioCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(audioCtx, 'capture-processor');
  source.connect(micNode);
  micNode.port.onmessage = (ev) => onMicBlock(ev.data);
  micOn = true;
  els.mic.textContent = '🎙 Mic on';
  els.mic.classList.add('active');
}

function stopMic() {
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  if (micNode) micNode.disconnect();
  micStream = null; micNode = null; micOn = false;
  sendBuf = new Float32Array(0);
  els.mic.textContent = '🎙 Mic off';
  els.mic.classList.remove('active');
  els.miclevel.style.width = '0%';
}

function onMicBlock(block) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  // Level meter.
  let peak = 0;
  for (let i = 0; i < block.length; i++) peak = Math.max(peak, Math.abs(block[i]));
  els.miclevel.style.width = Math.min(100, peak * 300) + '%';

  // Downsample ctx rate -> 16 kHz (linear interpolation), accumulate, frame out.
  const down = downsample(block, audioCtx.sampleRate, MIC_RATE);
  const merged = new Float32Array(sendBuf.length + down.length);
  merged.set(sendBuf); merged.set(down, sendBuf.length);
  sendBuf = merged;

  const frameSamples = MIC_RATE * FRAME_MS / 1000;
  while (sendBuf.length >= frameSamples) {
    const frame = sendBuf.subarray(0, frameSamples);
    sendBuf = sendBuf.slice(frameSamples);
    const pcm = new Int16Array(frameSamples);
    for (let i = 0; i < frameSamples; i++) {
      pcm[i] = Math.max(-32768, Math.min(32767, Math.round(frame[i] * 32767)));
    }
    ws.send(pcm.buffer);
  }
}

function downsample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    out[i] = input[i0] + (input[i1] - input[i0]) * (pos - i0);
  }
  return out;
}

// --- transcript UI ----------------------------------------------------------

let npcLineEl = null;      // streaming NPC line for the active turn
let turnHasDeltas = false; // greeting turns carry text only in reply.sentence

function addLine(cls, who, text) {
  const div = document.createElement('div');
  div.className = 'line ' + cls;
  div.innerHTML = who ? `<div class="who">${who}</div><div class="body"></div>` : `<div class="body"></div>`;
  div.querySelector('.body').textContent = text;
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return div;
}

function appendNpcDelta(name, delta) {
  if (!npcLineEl) npcLineEl = addLine('npc', name, '');
  npcLineEl.querySelector('.body').textContent += delta;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function showMetrics(breakdown) {
  els.metrics.innerHTML = '';
  const labels = {
    asr_ms: 'ASR', llm_ttft_ms: 'LLM first token', llm_total_ms: 'LLM total',
    tts_first_chunk_ms: 'TTS first chunk', first_audio_ms: '⚡ first audio (e2e)',
    total_ms: 'turn total',
  };
  for (const [key, label] of Object.entries(labels)) {
    if (!(key in breakdown)) continue;
    const tr = document.createElement('tr');
    if (key === 'first_audio_ms') tr.className = 'headline';
    tr.innerHTML = `<td>${label}</td><td>${breakdown[key].toFixed(0)}</td>`;
    els.metrics.appendChild(tr);
  }
}

function setStatus(text, isErr) {
  els.status.textContent = text;
  els.status.className = isErr ? 'err' : '';
}

// --- websocket --------------------------------------------------------------

let characterName = 'NPC';

function connect() {
  const characterId = els.character.value;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    audioCtx = audioCtx || new AudioContext();
    audioCtx.resume();
    ws.send(JSON.stringify({
      type: 'session.start', character_id: characterId, sample_rate: MIC_RATE, format: 'pcm16',
    }));
    setStatus('connecting session…');
  };

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) { playChunk(ev.data); return; }
    const msg = JSON.parse(ev.data);
    handleMessage(msg);
  };

  ws.onclose = () => {
    setStatus('disconnected');
    stopMic();
    stopPlayback();
    setConnected(false);
    ws = null;
  };
  ws.onerror = () => setStatus('websocket error', true);
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'session.ready':
      characterName = msg.character.name;
      ttsRate = msg.tts_sample_rate;
      setStatus(`talking to ${characterName}`);
      els.sessioninfo.textContent =
        `${msg.session_id} · ${msg.character.name} (${msg.character.id}) · TTS ${ttsRate} Hz`;
      addLine('sys', '', `— session with ${characterName} (${msg.character.role}) —`);
      setConnected(true);
      break;
    case 'vad':
      if (msg.event === 'speech_start') setStatus('listening…');
      if (msg.event === 'speech_end') setStatus('thinking…');
      break;
    case 'turn.start':
      npcLineEl = null;
      turnHasDeltas = false;
      break;
    case 'asr.final':
      addLine('user', 'you (asr)', msg.text);
      break;
    case 'reply.delta':
      turnHasDeltas = true;
      appendNpcDelta(characterName, msg.text);
      break;
    case 'reply.sentence':
      // Turns without LLM streaming (greetings) surface text via sentences.
      if (!turnHasDeltas) appendNpcDelta(characterName, msg.text + ' ');
      break;
    case 'audio.start':
      setStatus(`${characterName} is speaking…`);
      break;
    case 'audio.end':
      setStatus(`talking to ${characterName}`);
      break;
    case 'turn.metrics':
      showMetrics(msg.breakdown);
      break;
    case 'turn.cancelled':
      stopPlayback();
      addLine('sys', '', `— interrupted (${msg.reason}) —`);
      npcLineEl = null;
      setStatus(`talking to ${characterName}`);
      break;
    case 'turn.end':
      npcLineEl = null;
      break;
    case 'error':
      addLine('sys', '', `error: ${msg.message}`);
      setStatus(msg.message, true);
      break;
  }
}

function setConnected(on) {
  els.mic.disabled = !on;
  els.textinput.disabled = !on;
  els.send.disabled = !on;
  els.interrupt.disabled = !on;
  els.connect.textContent = on ? 'Disconnect' : 'Connect';
  els.character.disabled = on;
}

// --- wiring -----------------------------------------------------------------

els.connect.onclick = () => {
  if (ws) { ws.send(JSON.stringify({ type: 'session.end' })); ws.close(); }
  else connect();
};

els.mic.onclick = async () => {
  if (micOn) stopMic();
  else {
    try { await startMic(); }
    catch (err) { setStatus('mic error: ' + err.message, true); }
  }
};

function sendText() {
  const text = els.textinput.value.trim();
  if (!text || !ws) return;
  stopPlayback();
  addLine('user', 'you (text)', text);
  ws.send(JSON.stringify({ type: 'input.text', text }));
  els.textinput.value = '';
}
els.send.onclick = sendText;
els.textinput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendText(); });

els.interrupt.onclick = () => {
  if (!ws) return;
  stopPlayback();
  ws.send(JSON.stringify({ type: 'barge_in' }));
};

(async function init() {
  try {
    const res = await fetch('/api/characters');
    const characters = await res.json();
    els.character.innerHTML = characters
      .map(c => `<option value="${c.id}">${c.name} — ${c.role}</option>`)
      .join('');
  } catch (_) {
    setStatus('failed to load characters — is the server running?', true);
  }
})();
