// sequence.js — motion-app frontend: renders the running sentence, live
// signing status, chip highlight, and speaks each new sign in the browser.

const subtitleInner = document.getElementById('subtitleInner');
const segStatus = document.getElementById('segStatus');
const segText = document.getElementById('segText');
const stateDisplay = document.getElementById('stateDisplay');
const handsDisplay = document.getElementById('handsDisplay');
const feedStatus = document.getElementById('feedStatus');
const videoFeed = document.getElementById('videoFeed');
const voiceBtn = document.getElementById('voiceBtn');
const voiceIcon = document.getElementById('voiceIcon');
const voiceLabel = document.getElementById('voiceLabel');
const clearBtn = document.getElementById('clearBtn');
const polishBtn = document.getElementById('polishBtn');
const polishLabel = document.getElementById('polishLabel');
const polishedBar = document.getElementById('polishedBar');
const polishedText = document.getElementById('polishedText');
const polishedSpeak = document.getElementById('polishedSpeak');
const suggestionsEl = document.getElementById('suggestions');
const debugBtn = document.getElementById('debugBtn');
const debugPanel = document.getElementById('debugPanel');
const debugVerdict = document.getElementById('debugVerdict');
const debugTop = document.getElementById('debugTop');
const debugConf = document.getElementById('debugConf');
const debugMargin = document.getElementById('debugMargin');
const debugHint = document.getElementById('debugHint');
// two-way conversation elements
const micBtn = document.getElementById('micBtn');
const micIcon = document.getElementById('micIcon');
const micLabel = document.getElementById('micLabel');
const sendBtn = document.getElementById('sendBtn');
const convoClearBtn = document.getElementById('convoClearBtn');
const convoCopyBtn = document.getElementById('convoCopyBtn');
const convoShareBtn = document.getElementById('convoShareBtn');
const convoExportBtn = document.getElementById('convoExportBtn');
const convoScroll = document.getElementById('convoScroll');
const convoEmpty = document.getElementById('convoEmpty');
const convoMessages = document.getElementById('convoMessages');
const convoInterim = document.getElementById('convoInterim');
const convoInterimText = document.getElementById('convoInterimText');
const speechCaption = document.getElementById('speechCaption');
const speechCaptionText = document.getElementById('speechCaptionText');

const HINT = '<span class="subtitle-hint">Sign something — your translation appears here</span>';

// --- browser text-to-speech (Web Speech API — offline, on this device) ---
const ttsSupported = 'speechSynthesis' in window;
let voiceOn = ttsSupported;
let lastSpeakSeq = null;   // set on first message so we don't replay history

function speak(text) {
  if (!voiceOn || !ttsSupported || !text) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  window.speechSynthesis.cancel();  // interrupt any queued speech
  window.speechSynthesis.speak(u);
}

// Robustly silence any in-progress OR queued speech. Chrome can leave the
// synth "paused" (its 15s keep-alive quirk); cancel() on a paused engine may
// not flush, so resume first, then cancel — otherwise a word that was already
// mid-sentence when Voice is turned off keeps talking.
function stopSpeaking() {
  if (!ttsSupported) return;
  try {
    if (window.speechSynthesis.paused) window.speechSynthesis.resume();
    window.speechSynthesis.cancel();
  } catch (_) {}
}

if (!ttsSupported) {
  voiceBtn.disabled = true;
  voiceLabel.textContent = 'No voice';
}

function applyVoiceButton() {
  voiceBtn.classList.toggle('voice-on', voiceOn);
  voiceBtn.classList.toggle('voice-off', !voiceOn);
  voiceIcon.querySelector('use').setAttribute('href', voiceOn ? '#i-volume' : '#i-mute');
  voiceLabel.textContent = voiceOn ? 'Voice On' : 'Voice Off';
}
applyVoiceButton();

voiceBtn.addEventListener('click', () => {
  if (!ttsSupported) return;
  voiceOn = !voiceOn;
  if (!voiceOn) stopSpeaking();   // kill the word that's mid-sentence right now
  applyVoiceButton();
});

clearBtn.addEventListener('click', () => {
  fetch('/clear', { method: 'POST' }).catch(() => {});
  hidePolished();
});

// --- LLM sentence polishing (offline, via local Ollama on the server) ---
let polishing = false;
let lastPolished = '';

function setPolishing(on) {
  polishing = on;
  polishBtn.disabled = on;
  polishBtn.classList.toggle('polishing', on);
  polishLabel.textContent = on ? 'Polishing…' : 'Polish';
}

function hidePolished() {
  polishedBar.hidden = true;
  polishedText.textContent = '';
  polishedText.classList.remove('is-error');
  polishedSpeak.hidden = true;
  lastPolished = '';
}

function showPolished(text, isError) {
  polishedText.textContent = text;
  polishedText.classList.toggle('is-error', !!isError);
  polishedSpeak.hidden = !!isError || !text;
  polishedBar.hidden = false;
  // retrigger the fade-in animation
  polishedBar.classList.remove('pop');
  void polishedBar.offsetWidth;
  polishedBar.classList.add('pop');
}

polishBtn.addEventListener('click', () => {
  if (polishing) return;                 // guard against double-clicks
  setPolishing(true);
  fetch('/polish', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data && data.ok) {
        lastPolished = data.polished;
        showPolished(data.polished, false);
        if (voiceOn) speak(data.polished);   // reuse the existing TTS helper
      } else {
        showPolished((data && data.error) || 'Could not polish.', true);
      }
    })
    .catch(() => showPolished('Could not reach the server.', true))
    .finally(() => setPolishing(false));
});

polishedSpeak.addEventListener('click', () => {
  if (lastPolished) speak(lastPolished);
});

// --- next-sign autocomplete chips (scaffold; grows with the vocabulary) ---
function renderSuggestions(suggestions) {
  const list = suggestions || [];
  if (!list.length) {
    suggestionsEl.hidden = true;
    suggestionsEl.innerHTML = '';
    return;
  }
  suggestionsEl.innerHTML = '';
  list.forEach(word => {
    const chip = document.createElement('button');
    chip.className = 'suggestion-chip';
    chip.textContent = word;
    chip.title = 'Likely next sign';
    // preview only for now — highlight the matching sign in the sidebar
    chip.addEventListener('click', () => highlightGesture(word));
    suggestionsEl.appendChild(chip);
  });
  suggestionsEl.hidden = false;
}

// --- diagnostics: show WHY the last sign did / didn't translate ---
let debugOn = false;
let lastDebugSeq = null;

debugBtn.addEventListener('click', () => {
  debugOn = !debugOn;
  debugBtn.classList.toggle('active', debugOn);
  debugPanel.hidden = !debugOn;
});

// --- practice / learn mode: name a target sign, score the attempt ---
// Grading is entirely client-side: it reuses the SAME `last_debug` the server
// already publishes per classification (top guess + accepted gate flag), so the
// shared motion pipeline is untouched. We just compare `top` to a target word.
const practiceBtn = document.getElementById('practiceBtn');
const practicePanel = document.getElementById('practicePanel');
const practiceTarget = document.getElementById('practiceTarget');
const practiceVerdict = document.getElementById('practiceVerdict');
const practiceVerdictText = document.getElementById('practiceVerdictText');
const practiceStreak = document.getElementById('practiceStreak');
const practiceBest = document.getElementById('practiceBest');
const practiceAcc = document.getElementById('practiceAcc');
const practiceNextBtn = document.getElementById('practiceNextBtn');
const practiceSkipBtn = document.getElementById('practiceSkipBtn');
const practiceExitBtn = document.getElementById('practiceExitBtn');

// full label set injected by the template; reject labels are filtered out in
// buildPracticeVocab() once /config tells us which ones they are.
let allLabels = [];
try {
  allLabels = JSON.parse(document.getElementById('vocabData').textContent) || [];
} catch (_) { allLabels = []; }
let ignoreLabels = new Set();   // populated from /config (see fetch below)

const practice = {
  active: false,
  target: '',
  queue: [],
  streak: 0,
  best: 0,
  attempts: 0,
  correct: 0,
  lastSeq: null,   // last graded last_debug.seq — skips stale/idle results
};

const normLabel = s => (s || '').toLowerCase().trim().replace(/[_\s]+/g, ' ');

function buildPracticeVocab() {
  return allLabels.filter(l => !ignoreLabels.has(normLabel(l)));
}

// Fisher–Yates shuffle into a fresh round queue (every sign appears once before
// any repeats). Uses Math.random — this is client-only, no determinism needed.
function refillQueue() {
  const v = buildPracticeVocab();
  for (let i = v.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [v[i], v[j]] = [v[j], v[i]];
  }
  practice.queue = v;
}

function updatePracticeStats() {
  practiceStreak.textContent = String(practice.streak);
  practiceBest.textContent = String(practice.best);
  practiceAcc.textContent = practice.attempts
    ? Math.round((practice.correct / practice.attempts) * 100) + '%'
    : '—';
}

// set the verdict line's text + state color. The dot lives in its own span, so
// text goes to practiceVerdictText while the state class rides the container.
function setVerdict(text, state) {
  practiceVerdictText.textContent = text;
  practiceVerdict.className = 'practice-verdict' + (state ? ' ' + state : '');
}

function flashTarget(state) {
  // brief pop on the target word so a result is felt, not just read
  practiceTarget.classList.remove('pop-ok', 'pop-bad');
  if (state) {
    void practiceTarget.offsetWidth;   // restart the CSS animation
    practiceTarget.classList.add(state === 'ok' ? 'pop-ok' : 'pop-bad');
  }
}

function nextTarget() {
  if (!practice.queue.length) refillQueue();
  practice.target = practice.queue.pop() || '';
  practiceTarget.textContent = practice.target || '—';
  setVerdict('Perform the sign above.', '');
  if (practice.target) speak(practice.target);   // reuse the existing TTS helper
}

function gradeAttempt(dbg) {
  // ignore captures that weren't a real classified sign (don't punish a miss)
  if (!dbg || !dbg.top || dbg.reason === 'not a sign' ||
      dbg.reason === 'lost tracking') {
    setVerdict('Didn’t catch that — try again.', 'neutral');
    return;
  }
  const hit = normLabel(dbg.top) === normLabel(practice.target);
  practice.attempts++;
  if (hit && dbg.accepted) {
    practice.correct++;
    practice.streak++;
    practice.best = Math.max(practice.best, practice.streak);
    setVerdict(`Correct! ${dbg.conf}% confident`, 'ok');
    flashTarget('ok');
    updatePracticeStats();
    setTimeout(() => { if (practice.active) nextTarget(); }, 1400);
    return;
  }
  practice.streak = 0;
  if (hit) {
    // right sign, but it didn't clear the live gates — encourage, don't fail hard
    setVerdict(`Close — right sign, sign it clearer (${dbg.conf}%)`, 'close');
    flashTarget('bad');
  } else {
    setVerdict(`Try again — that looked like ${dbg.top}`, 'bad');
    flashTarget('bad');
  }
  updatePracticeStats();
}

function enterPractice() {
  practice.active = true;
  practice.streak = practice.best = practice.attempts = practice.correct = 0;
  // seed lastSeq from the current state so a stale pre-entry result isn't graded
  practice.lastSeq = lastDebugSeq;
  document.body.classList.add('practice-mode');
  practiceBtn.classList.add('active');
  practicePanel.hidden = false;
  updatePracticeStats();
  refillQueue();
  nextTarget();
  // tell the server to grade on the raw recognizer (no context prior) and to
  // stop learning transitions while we're drilling isolated signs
  fetch('/practice', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active: true }),
  }).catch(() => {});
}

function exitPractice() {
  practice.active = false;
  document.body.classList.remove('practice-mode');
  practiceBtn.classList.remove('active');
  practicePanel.hidden = true;
  fetch('/practice', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active: false }),
  }).catch(() => {});
}

practiceBtn.addEventListener('click', () => {
  if (practice.active) exitPractice(); else enterPractice();
});
practiceNextBtn.addEventListener('click', () => nextTarget());
practiceSkipBtn.addEventListener('click', () => nextTarget());
practiceExitBtn.addEventListener('click', () => exitPractice());

function renderDebug(dbg) {
  if (!debugOn) return;
  if (!dbg) {
    debugVerdict.textContent = '—';
    debugVerdict.className = 'debug-verdict';
    debugTop.textContent = debugConf.textContent = debugMargin.textContent = '—';
    debugHint.textContent = 'Sign, then read the numbers here.';
    return;
  }
  debugTop.textContent = dbg.top;
  debugConf.textContent = dbg.conf + '%';
  debugMargin.textContent = dbg.margin + '%';
  if (dbg.accepted) {
    debugVerdict.textContent = '✓ translated';
    debugVerdict.className = 'debug-verdict ok';
    debugHint.textContent = 'Cleared both gates.';
  } else {
    debugVerdict.textContent = '✕ rejected — ' + (dbg.reason || 'gated');
    debugVerdict.className = 'debug-verdict bad';
    if (dbg.reason === 'low confidence') {
      debugHint.textContent = 'Model unsure — collect more clips of this sign.';
    } else if (dbg.reason === 'ambiguous') {
      debugHint.textContent = 'Two signs looked alike — make the shape more distinct.';
    } else if (dbg.reason === 'not a sign') {
      debugHint.textContent = 'Read as the background/"not a sign" class.';
    } else {
      debugHint.textContent = '';
    }
  }
}

// --- gesture chip highlight ---
function highlightGesture(word) {
  const norm = (word || '').toLowerCase().trim().replace(/\s+/g, ' ');
  document.querySelectorAll('.gesture-chip').forEach(chip => {
    const c = chip.dataset.gesture.toLowerCase().trim().replace(/[_\s]+/g, ' ');
    chip.classList.toggle('predicted', !!norm && c === norm);
  });
}

// --- render the running sentence ---
// The full sentence is preserved server-side (Send/Polish use it); the caption
// only ever shows the most recent words so it stays one clean line, like a
// modern live-caption strip, instead of an ever-growing wall of text.
const MAX_CAPTION_WORDS = 7;
function renderSentence(sentence, collecting) {
  subtitleInner.innerHTML = '';
  if ((!sentence || sentence.length === 0) && !collecting) {
    subtitleInner.innerHTML = HINT;
    return;
  }
  const words = sentence || [];
  const shown = words.slice(-MAX_CAPTION_WORDS);
  shown.forEach(item => {
    const span = document.createElement('span');
    span.className = 'word ' + (item.conf >= 90 ? 'high' : 'borderline');
    span.textContent = item.word;
    subtitleInner.appendChild(span);
  });
  if (collecting) {
    const cur = document.createElement('span');
    cur.className = 'word current';
    cur.textContent = '…';
    subtitleInner.appendChild(cur);
  }
}

// ===================================================================
//  Two-way conversation: sign -> text bubbles + speech -> text reply
// ===================================================================

let latestSentenceWords = [];   // most recent server sentence (for Send)
const transcript = [];          // {kind, text, time} — for copy/share/export

function timeStamp() {
  const d = new Date();
  let h = d.getHours();
  const m = d.getMinutes();
  const ampm = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  return h + ':' + String(m).padStart(2, '0') + ' ' + ampm;
}

function addMessage(kind, text) {
  text = (text || '').trim();
  if (!text) return;
  convoEmpty.hidden = true;

  const time = timeStamp();
  transcript.push({ kind, text, time });

  const msg = document.createElement('div');
  msg.className = 'msg ' + kind;               // 'signed' | 'spoken'

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = (kind === 'signed' ? 'Signed' : 'Spoken') + ' · ' + time;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  msg.appendChild(meta);
  msg.appendChild(bubble);
  convoMessages.appendChild(msg);
  convoScroll.scrollTop = convoScroll.scrollHeight;
}

// --- Copy / Share / Export the transcript ---
function transcriptToText() {
  return transcript
    .map(m => `[${m.time}] ${m.kind === 'signed' ? 'Signed' : 'Spoken'}: ${m.text}`)
    .join('\n');
}

function flashAct(btn, ok) {
  const prev = btn.innerHTML;
  btn.innerHTML = ok ? '&#10003;' : '&#10005;';
  btn.classList.toggle('done', ok);
  setTimeout(() => { btn.innerHTML = prev; btn.classList.remove('done'); }, 1000);
}

convoCopyBtn.addEventListener('click', async () => {
  if (!transcript.length) { flashAct(convoCopyBtn, false); return; }
  try {
    await navigator.clipboard.writeText(transcriptToText());
    flashAct(convoCopyBtn, true);
  } catch (_) {
    flashAct(convoCopyBtn, false);
  }
});

convoShareBtn.addEventListener('click', async () => {
  if (!transcript.length) { flashAct(convoShareBtn, false); return; }
  const text = transcriptToText();
  if (navigator.share) {
    try {
      await navigator.share({ title: 'Gestura conversation', text });
      flashAct(convoShareBtn, true);
    } catch (_) { /* user cancelled — no error state */ }
  } else {
    // no Web Share API (most desktops): fall back to copying
    try {
      await navigator.clipboard.writeText(text);
      flashAct(convoShareBtn, true);
    } catch (_) {
      flashAct(convoShareBtn, false);
    }
  }
});

convoExportBtn.addEventListener('click', () => {
  if (!transcript.length) { flashAct(convoExportBtn, false); return; }
  const header = 'Gestura conversation\n' +
                 '='.repeat(28) + '\n\n';
  const blob = new Blob([header + transcriptToText() + '\n'], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const d = new Date();
  const stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}` +
                `${String(d.getDate()).padStart(2, '0')}-${String(d.getHours()).padStart(2, '0')}` +
                `${String(d.getMinutes()).padStart(2, '0')}`;
  a.href = url;
  a.download = `gestura-${stamp}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  flashAct(convoExportBtn, true);
});

// --- Send: commit the current signed sentence into the conversation ---
function flashSend(label) {
  const prev = sendBtn.innerHTML;
  sendBtn.innerHTML = label;
  sendBtn.disabled = true;
  setTimeout(() => { sendBtn.innerHTML = prev; sendBtn.disabled = false; }, 900);
}

sendBtn.addEventListener('click', () => {
  // prefer the polished sentence if it's on screen, else the raw signs
  const text = ((!polishedBar.hidden && lastPolished)
                ? lastPolished : latestSentenceWords.join(' ')).trim();
  if (!text) { flashSend('Nothing yet'); return; }
  addMessage('signed', text);
  if (voiceOn) speak(text);                 // let the hearing person hear it
  fetch('/clear', { method: 'POST' }).catch(() => {});  // fresh next turn
  hidePolished();
  latestSentenceWords = [];
});

// --- Clear the whole conversation transcript ---
convoClearBtn.addEventListener('click', () => {
  convoMessages.innerHTML = '';
  transcript.length = 0;
  convoEmpty.hidden = false;
});

// ------------------------------------------------------------------
//  Speech-to-text reply lane (Web Speech API - in-browser, free)
// ------------------------------------------------------------------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let listening = false;
let captionTimer = null;

function showCaption(text, interim) {
  speechCaptionText.textContent = text;
  speechCaption.classList.toggle('interim', !!interim);
  speechCaption.hidden = !text;
  if (captionTimer) clearTimeout(captionTimer);
  if (!interim && text) {                   // linger final captions, then fade
    captionTimer = setTimeout(() => { speechCaption.hidden = true; }, 6000);
  }
}

function applyMicButton() {
  micBtn.classList.toggle('listening', listening);
  micLabel.textContent = listening ? 'Listening…' : 'Speak';
}

if (!SR) {
  micBtn.disabled = true;
  micLabel.textContent = 'No mic';
  micBtn.title = 'Speech recognition needs Chrome or Edge';
} else {
  recognition = new SR();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.continuous = true;

  recognition.onresult = (e) => {
    let interim = '', finalTxt = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) finalTxt += r[0].transcript;
      else interim += r[0].transcript;
    }
    if (interim) {
      convoInterim.hidden = false;
      convoInterimText.textContent = interim;
      showCaption(interim, true);
    }
    if (finalTxt.trim()) {
      addMessage('spoken', finalTxt);
      showCaption(finalTxt.trim(), false);
      convoInterim.hidden = true;
      convoInterimText.textContent = '';
    }
  };

  // Chrome ends recognition after a pause - restart while still toggled on
  recognition.onend = () => {
    if (listening) { try { recognition.start(); } catch (_) {} }
    else { convoInterim.hidden = true; }
  };

  recognition.onerror = (e) => {
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      listening = false;
      applyMicButton();
      convoInterim.hidden = true;
      showCaption('Microphone blocked - allow mic access to use Speak.', false);
    }
    // 'no-speech' / 'aborted' are transient; onend restarts if still listening
  };

  micBtn.addEventListener('click', () => {
    listening = !listening;
    applyMicButton();
    if (listening) {
      try { recognition.start(); } catch (_) {}
      convoInterim.hidden = false;
      convoInterimText.textContent = 'Listening…';
    } else {
      recognition.stop();
      convoInterim.hidden = true;
    }
  });
}
applyMicButton();

videoFeed.addEventListener('load', () => {
  feedStatus.style.display = 'none';
  cameraReady = true;
  camHud.hidden = false;
});
videoFeed.addEventListener('error', () => {
  feedStatus.querySelector('span').textContent = 'Camera unavailable';
  const ring = feedStatus.querySelector('.status-ring');
  if (ring) ring.remove();
});

const evtSource = new EventSource('/state');
evtSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  // speak only genuinely new words (skip the backlog on first connect)
  if (lastSpeakSeq === null) {
    lastSpeakSeq = data.speak_seq;
  } else if (data.speak_seq > lastSpeakSeq) {
    lastSpeakSeq = data.speak_seq;
    speak(data.speak_word);
  }

  renderSentence(data.sentence, data.collecting);
  renderSuggestions(data.suggestions);
  latestSentenceWords = (data.sentence || []).map(item => item.word);

  // diagnostics: render each NEW classification result (accepted or rejected)
  const dbg = data.last_debug;
  if (dbg && dbg.seq !== lastDebugSeq) {
    lastDebugSeq = dbg.seq;
    renderDebug(dbg);
    // practice mode grades each new classification against the target word
    if (practice.active && dbg.seq !== practice.lastSeq) {
      practice.lastSeq = dbg.seq;
      gradeAttempt(dbg);
    }
  }

  // if the sentence was cleared (idle auto-clear or Clear), drop the polished bar
  if ((!data.sentence || data.sentence.length === 0) && !polishing && !polishedBar.hidden) {
    hidePolished();
  }

  if (data.collecting) {
    segStatus.classList.add('signing');
    segText.textContent = `Signing…  ${data.seg_frames} frames`;
    stateDisplay.textContent = 'Signing';
  } else {
    segStatus.classList.remove('signing');
    segText.textContent = 'Ready — start signing';
    stateDisplay.textContent = 'Ready';
  }

  handsDisplay.textContent = data.hands > 0 ? String(data.hands) : '—';
  highlightGesture(data.last_word);
  renderCamera(data);
};

evtSource.onerror = () => {
  feedStatus.style.display = 'flex';
  feedStatus.querySelector('span').textContent = 'Reconnecting...';
};

// --- PWA: register the service worker (installable + offline app shell) ---
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js').catch(() => {});
  });
}

// ------------------------------------------------------------------
//  Accessibility settings (large captions, high contrast, size)
//  Applied via <body> classes + a CSS var; remembered in localStorage.
// ------------------------------------------------------------------
const a11yBtn = document.getElementById('a11yBtn');
const a11yMenu = document.getElementById('a11yMenu');
const a11yLargeCap = document.getElementById('a11yLargeCap');
const a11yContrast = document.getElementById('a11yContrast');
const a11ySize = document.getElementById('a11ySize');
const a11ySizeVal = document.getElementById('a11ySizeVal');
const a11yReset = document.getElementById('a11yReset');

const A11Y_KEY = 'signtranslate.a11y';
const a11y = { largeCap: false, contrast: false, size: 100 };

function applyA11y() {
  document.body.classList.toggle('a11y-large-cap', a11y.largeCap);
  document.body.classList.toggle('a11y-contrast', a11y.contrast);
  document.body.style.setProperty('--caption-scale', a11y.size / 100);
  a11yLargeCap.checked = a11y.largeCap;
  a11yContrast.checked = a11y.contrast;
  a11ySize.value = a11y.size;
  a11ySizeVal.textContent = a11y.size + '%';
}

function saveA11y() {
  try { localStorage.setItem(A11Y_KEY, JSON.stringify(a11y)); } catch (_) {}
}

function loadA11y() {
  try {
    const s = JSON.parse(localStorage.getItem(A11Y_KEY) || '{}');
    if (typeof s.largeCap === 'boolean') a11y.largeCap = s.largeCap;
    if (typeof s.contrast === 'boolean') a11y.contrast = s.contrast;
    if (typeof s.size === 'number') a11y.size = s.size;
  } catch (_) {}
  applyA11y();
}

a11yBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const open = a11yMenu.hidden;
  a11yMenu.hidden = !open;
  a11yBtn.setAttribute('aria-expanded', String(open));
});
document.addEventListener('click', (e) => {
  if (!a11yMenu.hidden && !a11yMenu.contains(e.target) && e.target !== a11yBtn) {
    a11yMenu.hidden = true;
    a11yBtn.setAttribute('aria-expanded', 'false');
  }
});

a11yLargeCap.addEventListener('change', () => { a11y.largeCap = a11yLargeCap.checked; applyA11y(); saveA11y(); });
a11yContrast.addEventListener('change', () => { a11y.contrast = a11yContrast.checked; applyA11y(); saveA11y(); });
a11ySize.addEventListener('input', () => { a11y.size = parseInt(a11ySize.value, 10); applyA11y(); saveA11y(); });
a11yReset.addEventListener('click', () => { a11y.largeCap = false; a11y.contrast = false; a11y.size = 100; applyA11y(); saveA11y(); });

loadA11y();

// ==================================================================
//  CAMERA FEATURES
//  - motion "listening" meter + recording ring (driven by SSE state)
//  - framing / coaching guide when idle with no hands
//  - live FPS + hands HUD
//  - snapshot, clip recording, Picture-in-Picture, fullscreen
//  All layered over the existing MJPEG feed — no pipeline changes.
// ==================================================================
const feedContainer = document.getElementById('feedContainer');
const motionFill = document.getElementById('motionFill');
const motionMark = document.getElementById('motionMark');
const framingGuide = document.getElementById('framingGuide');
const frameCoach = document.getElementById('frameCoach');
const camHud = document.getElementById('camHud');
const hudFps = document.getElementById('hudFps');
const hudHands = document.getElementById('hudHands');
const camTools = document.getElementById('camTools');
const snapBtn = document.getElementById('snapBtn');
const recBtn = document.getElementById('recBtn');
const pipBtn = document.getElementById('pipBtn');
const fsBtn = document.getElementById('fsBtn');
const recIndicator = document.getElementById('recIndicator');
const recTime = document.getElementById('recTime');

let cameraReady = false;

// motion thresholds come from the server so the meter uses the SAME numbers
// as the segmentation loop. Fallbacks match live_sequence.py defaults.
let camCfg = { motion_start: 0.02, motion_stop: 0.01, max_seg: 180, min_seg: 10 };
fetch('/config')
  .then(r => r.json())
  .then(cfg => {
    camCfg = Object.assign(camCfg, cfg || {});
    // reject labels for practice-mode vocab (same set the server's VOCAB uses)
    if (Array.isArray(cfg && cfg.ignore_labels)) {
      ignoreLabels = new Set(cfg.ignore_labels.map(normLabel));
    }
    // place the threshold marker at its spot on the meter (start = 62.5% of range)
    motionMark.style.left = (100 / 1.6) + '%';
  })
  .catch(() => { motionMark.style.left = (100 / 1.6) + '%'; });

function renderCamera(data) {
  // motion meter: scale so MOTION_START sits at ~62.5% (range = start * 1.6)
  const range = Math.max(camCfg.motion_start * 1.6, 1e-6);
  const pct = Math.min(100, ((data.motion || 0) / range) * 100);
  motionFill.style.width = pct.toFixed(1) + '%';
  feedContainer.classList.toggle('is-signing', !!data.collecting);

  // HUD
  if (typeof data.fps === 'number') hudFps.textContent = data.fps.toFixed(0);
  hudHands.textContent = String(data.hands || 0);

  // framing / coaching: only when idle, camera up, and nothing in frame
  const showGuide = cameraReady && !data.collecting && (data.hands || 0) === 0;
  framingGuide.hidden = !showGuide;
  if (showGuide) frameCoach.textContent = 'Center your hands in the frame';
}

// ---- shared draw loop: paints the MJPEG frame into a canvas so we can
//      record it and feed Picture-in-Picture. Ref-counted so it only runs
//      while something actually needs it. ----
const camCanvas = document.createElement('canvas');
const camCtx = camCanvas.getContext('2d');
let drawRAF = null;
let drawUsers = 0;

function drawFrame() {
  const w = videoFeed.naturalWidth || videoFeed.width;
  const h = videoFeed.naturalHeight || videoFeed.height;
  if (w && h) {
    if (camCanvas.width !== w) { camCanvas.width = w; camCanvas.height = h; }
    try { camCtx.drawImage(videoFeed, 0, 0, w, h); } catch (_) {}
  }
  drawRAF = requestAnimationFrame(drawFrame);
}
function startDraw() {
  drawUsers++;
  if (drawRAF === null) drawFrame();
}
function stopDraw() {
  drawUsers = Math.max(0, drawUsers - 1);
  if (drawUsers === 0 && drawRAF !== null) {
    cancelAnimationFrame(drawRAF);
    drawRAF = null;
  }
}

function camStamp() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
         `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ---- snapshot: one still of the annotated feed ----
snapBtn.addEventListener('click', () => {
  const w = videoFeed.naturalWidth, h = videoFeed.naturalHeight;
  if (!w || !h) return;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  try { c.getContext('2d').drawImage(videoFeed, 0, 0, w, h); } catch (_) { return; }
  c.toBlob(blob => { if (blob) downloadBlob(blob, `gestura-${camStamp()}.png`); }, 'image/png');
  // visual feedback: white flash + button pulse
  feedContainer.classList.remove('snap-flash');
  void feedContainer.offsetWidth;
  feedContainer.classList.add('snap-flash');
  snapBtn.classList.add('flash');
  setTimeout(() => snapBtn.classList.remove('flash'), 700);
});

// ---- clip recording via MediaRecorder over the canvas stream ----
let recorder = null;
let recChunks = [];
let recTimer = null;
let recStart = 0;
const recSupported = 'MediaRecorder' in window && !!HTMLCanvasElement.prototype.captureStream;

if (!recSupported) {
  recBtn.disabled = true;
  recBtn.title = 'Recording not supported in this browser';
}

function fmtDur(ms) {
  const s = Math.floor(ms / 1000);
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}

function startRecording() {
  if (!recSupported || recorder) return;
  startDraw();                       // ensure the canvas is being painted
  const stream = camCanvas.captureStream(30);
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mime = types.find(t => MediaRecorder.isTypeSupported(t)) || '';
  let rec;
  try {
    rec = mime ? new MediaRecorder(stream, { mimeType: mime })
               : new MediaRecorder(stream);
  } catch (_) { stopDraw(); return; }
  recorder = rec;
  recChunks = [];
  rec.ondataavailable = e => { if (e.data && e.data.size) recChunks.push(e.data); };
  // reference `rec` (closure local), not `recorder` — stopRecording() nulls the
  // outer handle synchronously, but onstop fires afterwards.
  rec.onstop = () => {
    const blob = new Blob(recChunks, { type: rec.mimeType || 'video/webm' });
    if (blob.size) downloadBlob(blob, `gestura-${camStamp()}.webm`);
    stopDraw();
  };
  rec.start();
  recStart = Date.now();
  recBtn.classList.add('recording');
  recBtn.querySelector('use').setAttribute('href', '#i-stop');
  recBtn.title = 'Stop recording';
  camTools.classList.add('pinned');
  recIndicator.hidden = false;
  recTime.textContent = '0:00';
  recTimer = setInterval(() => { recTime.textContent = fmtDur(Date.now() - recStart); }, 250);
}

function stopRecording() {
  if (!recorder) return;
  try { recorder.stop(); } catch (_) {}
  recorder = null;
  clearInterval(recTimer);
  recTimer = null;
  recBtn.classList.remove('recording');
  recBtn.querySelector('use').setAttribute('href', '#i-record');
  recBtn.title = 'Record a clip';
  camTools.classList.remove('pinned');
  recIndicator.hidden = true;
}

recBtn.addEventListener('click', () => {
  if (recorder) stopRecording(); else startRecording();
});

// ---- Picture-in-Picture (canvas -> hidden <video> -> PiP) ----
const pipVideo = document.createElement('video');
pipVideo.muted = true;
pipVideo.playsInline = true;
let pipActive = false;
const pipSupported = document.pictureInPictureEnabled &&
                     typeof pipVideo.requestPictureInPicture === 'function';

if (pipSupported) {
  pipBtn.hidden = false;
  pipVideo.addEventListener('leavepictureinpicture', () => {
    pipActive = false;
    pipVideo.pause();
    stopDraw();
    pipBtn.classList.remove('flash');
  });
}

pipBtn.addEventListener('click', async () => {
  if (!pipSupported) return;
  if (pipActive) { try { await document.exitPictureInPicture(); } catch (_) {} return; }
  startDraw();
  try {
    if (!pipVideo.srcObject) pipVideo.srcObject = camCanvas.captureStream(30);
    await pipVideo.play();
    await pipVideo.requestPictureInPicture();
    pipActive = true;
    pipBtn.classList.add('flash');
  } catch (_) {
    stopDraw();   // entering PiP failed — release the draw loop
  }
});

// ---- fullscreen ----
fsBtn.addEventListener('click', () => {
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  } else {
    feedContainer.requestFullscreen?.().catch(() => {});
  }
});

// ---- keyboard shortcuts (ignore while typing in a field) ----
document.addEventListener('keydown', (e) => {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (e.key === 'f' || e.key === 'F') { fsBtn.click(); }
  else if (e.key === 'p' || e.key === 'P') { snapBtn.click(); }
});
