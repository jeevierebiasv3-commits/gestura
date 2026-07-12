const predictionText = document.getElementById('predictionText');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const predictionBox = document.getElementById('predictionBox');
const feedStatus = document.getElementById('feedStatus');
const videoFeed = document.getElementById('videoFeed');
const fpsDisplay = document.getElementById('fpsDisplay');
const handsDisplay = document.getElementById('handsDisplay');
const overlay = document.querySelector('.feed-overlay');

let currentPrediction = '';
let frameCount = 0;
let lastFpsUpdate = performance.now();

function highlightGesture(text) {
  if (!text) return;
  const normalized = text.toLowerCase().trim();
  document.querySelectorAll('.gesture-chip').forEach(chip => {
    const chipText = chip.dataset.gesture.toLowerCase().trim();
    chip.classList.toggle('predicted', chipText === normalized);
  });
}

function clearHighlight() {
  document.querySelectorAll('.gesture-chip.predicted').forEach(chip => {
    chip.classList.remove('predicted');
  });
}

videoFeed.addEventListener('load', () => {
  feedStatus.style.display = 'none';
});

videoFeed.addEventListener('error', () => {
  feedStatus.querySelector('span').textContent = 'Camera unavailable';
  feedStatus.querySelector('.status-ring').remove();
});

const evtSource = new EventSource('/prediction');
evtSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  const text = data.text || '';
  const conf = data.confidence || 0;

  frameCount++;

  if (text) {
    predictionText.textContent = text;
    confidenceFill.style.width = conf + '%';
    confidenceValue.textContent = conf + '%';
    overlay.classList.add('visible');
    highlightGesture(text);
    currentPrediction = text;
  } else {
    if (currentPrediction) {
      overlay.classList.remove('visible');
      clearHighlight();
      currentPrediction = '';
      predictionText.textContent = '—';
      confidenceFill.style.width = '0%';
      confidenceValue.textContent = '0%';
    }
  }

  const now = performance.now();
  if (now - lastFpsUpdate >= 1000) {
    fpsDisplay.textContent = frameCount;
    frameCount = 0;
    lastFpsUpdate = now;
  }

  const handsCount = (typeof data.hands === 'number') ? data.hands : 0;
  handsDisplay.textContent = handsCount > 0 ? String(handsCount) : '—';
};

evtSource.onerror = () => {
  feedStatus.style.display = 'flex';
  feedStatus.querySelector('span').textContent = 'Reconnecting...';
};
