# Gestura — Simple Demo Script

A plain, easy-to-follow script for showing **Gestura** (the sign-language
translator) to the Deaf and hard-of-hearing community.

**Two things to set up first:** have someone captioning or interpreting so
everyone can follow you, and make sure the room can read the screen. Everything
important shows up as **text on screen** — that's what people should watch.

---

## Before you start

- Good light on your hands, plain background, camera at chest height.
- Open the app: run `python app_sequence.py`, then open `http://localhost:5000`.
- Do a quick test: sign one sign and check a word appears. If it does, you're ready.
- Signs the app knows right now: **Hi, Hello, Good Morning, Good Afternoon,
  Thank you, Sorry, Excuse me, Yes, I love you, Take care, See you later,
  Well done, Congratulations, All done, I.**

---

## What to say at the start

> "This is Gestura. It uses a normal laptop camera to read sign language and turn
> it into words on the screen. It also lets a hearing person reply — so both
> people can talk. It runs on the laptop itself, offline, so your signing stays
> private and is never uploaded."

Then let them watch the camera track your hands for a few seconds before you say
anything else. Seeing it work says more than words.

---

## Show the features, one at a time

Go in this order. Each step is: **do the action → say the one point.**

**1. It sees your hands.**
Hold up both hands and move them. Dots follow your fingers.
→ "It finds both hands and follows every finger."

**2. It turns signs into words.**
Sign **Hello**, then **Thank you**, then **Good Morning**. Wait for each word.
→ "These are real signs with movement — not frozen poses. I sign, it writes."

**3. It's honest when it's unsure.**
Turn on the **Why** button. Sign something unclear on purpose.
→ "If it's not sure, it doesn't guess — it tells you why it waited."

**4. It builds a sentence and can speak it.**
Sign **Hi**, then **Thank you**. The words join into a sentence. Point to the
speaker button.
→ "The words gather into a sentence, and it can say it out loud for a hearing
person."

**5. It suggests the next sign.**
Sign **Hi**. A few suggestion chips appear. Tap one to add it.
→ "Like predictive text — it suggests what usually comes next, and gets smarter
with use."

**6. It cleans up the sentence.** *(only if Ollama is running)*
Sign **I**, **Thank you**, **Take care**. Press **Polish**.
→ "It turns the signs into a natural sentence, without changing your meaning."

**7. The hearing person can reply.**
Press **Speak** and say a short sentence out loud (or have a helper do it). It
appears as text.
→ "You sign, it becomes text. They speak, it becomes text. Both on one screen."

**8. It keeps a transcript.**
Point to the conversation list. Show Copy, Share, and Export.
→ "The whole conversation is saved — you can keep it or share it."

**9. It can teach you.**
Press **Practice**. It names a sign, you do it, it scores you.
→ "Practice mode names a sign and tells you how well you did — good for learners."

---

## What to say at the end

> "That's Gestura. A normal laptop, private and offline, and both people can talk
> on the same screen. It's not here to replace sign language or interpreters —
> it's a helper for moments when there isn't one. I'd love to hear from you: which
> signs matter most, and what would make this useful in your day?"

Then invite people to come and sign into it themselves.

---

## Quick answers if people ask

- **Is it recording or uploading me?** No. Everything runs on the laptop and works
  offline. It only saves something if you press save.
- **Which signs does it know?** The list above for now — and more can be added.
- **What if it gets a sign wrong?** It's built to wait rather than guess (that's
  the Why button). You're always in control.
- **Does it replace interpreters?** No. It's a helper for everyday moments where
  no interpreter is there.
