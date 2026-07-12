# Sign Translator — Feature Roadmap & Ideas

Brainstorm of modern features and functionality to take the app to the next level.
Grouped by the kind of "next level" each pushes toward. ⭐ = top picks.

---

## 🤖 AI-powered (the "wow" tier)

- **⭐ LLM sentence polishing** — the app already builds a sentence of raw signs
  ("Hi / Good / Morning"). Pipe it through an LLM to produce natural language
  ("Good morning! How are you?"). Single most modern-feeling upgrade — turns
  word-salad into real conversation. Can run on a "Polish" button so it's cheap.
- **Context-aware prediction** — feed the recent sentence back as context so the
  model biases toward signs that make grammatical sense (a lightweight
  language-model prior over the sign vocabulary). Fewer nonsense sequences.
- **Auto-suggest / autocomplete** — as signs come in, show 2–3 likely next-word
  completions (like phone keyboards). Speeds up "conversation."

## 🗣️ Two-way conversation (translator → communicator)

- **⭐ Speech-to-text reply lane** — the hearing person taps the mic, their speech
  becomes on-screen text for the deaf user. Full loop, not one-directional.
  Web Speech API does this in-browser, free. — ✅ **DONE** (Speak button +
  big "They said" caption over the video)
- **Conversation view** — a chat-style transcript: signs on one side, spoken
  replies on the other, timestamped. Feels like a messaging app; very shareable
  in a demo. — ✅ **DONE** (sidebar transcript; Send commits the signed sentence)
- **Live shareable session** — a room link so a remote person sees your translated
  captions in real time (WebRTC/websocket). "Video call with live sign captions."
  — ⏳ not yet (needs a websocket/WebRTC server + rooms; bigger infra lift)

## 📱 Modern UX & PWA

- **⭐ Progressive Web App (PWA)** — installable to phone home screen, works
  offline, fullscreen. Makes it feel like a real product, not a localhost demo.
  Low effort, high perceived polish. — ✅ **DONE** (manifest + icons + service
  worker; app shell caches offline. Note: live translation still needs the
  server — true on-device offline needs in-browser inference below.)
- **Dark/light + accessibility modes** — large-caption mode, high-contrast,
  adjustable subtitle size (this audience genuinely needs it). — ✅ **DONE**
  (Aa menu: large captions, high contrast, caption-size slider, saved locally).
  Light theme not added (app is dark-first).
- **Copy / Share / Export** — copy the sentence, share the transcript, or export
  a session as text. — ✅ **DONE** (copy / Web-Share / export-.txt on the chat)
- **Responsive mobile layout** — current layout is desktop-first; a phone-friendly
  version massively widens who can try it. — ⏳ deferred (the camera + model run
  server-side, so a phone can't really run it until in-browser inference lands)

## 🎓 Product surfaces (a whole second audience)

- **⭐ Practice / Learn mode** — "Sign the word HELLO" → it scores you and gives
  feedback. Same tech, but now it's an ASL *teaching* app. Gamified streaks,
  accuracy %. This is how apps like this go viral.
- **In-app sign training** — let a user add their own sign by recording clips and
  retraining, all from the browser, with a live "how well does the model know
  this?" meter. Removes the terminal entirely.
- **Analytics dashboard** — most-used signs, accuracy over time, per-sign
  confusion. Makes it feel data-driven.

## ⚙️ Technical modernization (foundation for the above)

- **WebRTC in-browser inference** — run MediaPipe Tasks + a TF.js model in the
  browser instead of streaming frames to Python. Lower latency, scales to many
  users, deployable to the web for real.
- **Model confidence calibration** — temperature scaling / an "energy score" so
  "unknown" reads as unknown instead of "Hi 97%" (the recurring root issue).
- **TFLite / on-device mobile** — the real-world endgame: a phone app that works
  without a laptop.

---

## Recommended order (max modern-feel for the effort)

1. **⭐ LLM sentence polishing** — biggest "this is an AI app" moment, builds
   directly on what exists.
2. **⭐ Speech-to-text reply lane + conversation view** — flips it into two-way
   communication.
3. **⭐ PWA + Copy/Share/Export** — makes it feel like a shippable product.

That trio takes the app from "a working recognizer" to "a communication app
people would actually install."

---

## Already done

- ✅ Open-set rejection (confidence + margin gates, NONE/background class)
- ✅ Motion / sequence model with variable-length + continuous signing
- ✅ Motion-gated segmentation (auto sign start/stop detection)
- ✅ Subtitle system (word-wrap, auto-scroll, auto-clear, confidence colors)
- ✅ Dataset-health report in the trainer (per-class counts, warnings)
- ✅ Text-to-speech (offline, desktop app; Web Speech API in web app)
- ✅ Web app port of the motion model (`app_sequence.py` + modern UI)
- ✅ LLM sentence polishing (offline via local Ollama; "Polish" button)
- ✅ "Why didn't it translate?" debug overlay (top guess + confidence + margin)
- ✅ Two-way conversation: speech-to-text reply lane + chat transcript view
- ✅ Installable PWA (manifest, icons, offline app-shell service worker)
- ✅ Accessibility modes (large captions, high contrast, caption-size slider)
- ✅ Copy / Share / Export the conversation transcript
