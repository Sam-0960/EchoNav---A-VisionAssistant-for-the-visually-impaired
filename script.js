// Note: Assuming a JS version of YOLO from ultralytics or similar; in reality, ultralytics is Python, so this is placeholder.
// For actual implementation, use TensorFlow.js with a YOLO model, e.g., load from TF Hub.
// Here, we'll simulate with TensorFlow.js and COCO-SSD as proxy for YOLO, but adapt to YOLO API.

import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd'; // Using as stand-in for YOLO

// DOM elements
const video = document.getElementById('video');
const canvas = document.getElementById('captureCanvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalInput = document.getElementById('interval');
const speakToggle = document.getElementById('speakToggle');
const statusDiv = document.getElementById('status');
const logDiv = document.getElementById('log');
const modeRadios = document.querySelectorAll('input[name="mode"]');

// Variables
let model;
let stream;
let intervalId;
let isRunning = false;

// Load YOLO model (using COCO-SSD as proxy; replace with actual YOLO loading)
async function loadModel() {
  model = await cocoSsd.load(); // Placeholder: load actual YOLO model here, e.g., tf.loadGraphModel('yolo-url')
  console.log('Model loaded');
}

// Start webcam
async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    statusDiv.textContent = 'Camera active';
    statusDiv.style.color = 'var(--accent)'; // Access CSS variable
  } catch (err) {
    console.error('Error accessing webcam:', err);
    statusDiv.textContent = 'Camera error';
    statusDiv.style.color = 'red';
  }
}

// Stop webcam
function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  if (intervalId) clearInterval(intervalId);
  isRunning = false;
  statusDiv.textContent = 'Idle';
  statusDiv.style.color = 'var(--muted)'; // Access CSS
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// Run detection
async function detect() {
  if (!model || !isRunning) return;

  const currentMode = document.querySelector('input[name="mode"]:checked').value;

  if (currentMode === 'local') {
    // Local YOLO detection
    const predictions = await model.detect(video);
    // Draw on canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    predictions.forEach(pred => {
      ctx.strokeStyle = 'lime';
      ctx.lineWidth = 2;
      ctx.strokeRect(pred.bbox[0], pred.bbox[1], pred.bbox[2], pred.bbox[3]);
      ctx.fillStyle = 'lime';
      ctx.font = '16px Arial';
      ctx.fillText(pred.class, pred.bbox[0], pred.bbox[1] - 5);
    });
    // Update log
    const detections = predictions.map(p => `${p.class} (${Math.round(p.score * 100)}%)`).join(', ') || 'No objects detected';
    logDiv.textContent = detections;
    // Speak
    if (speakToggle.checked) {
      speechSynthesis.speak(new SpeechSynthesisUtterance(detections));
    }
  } else if (currentMode === 'server') {
    // Server mode: capture and send
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append('image', blob);
      try {
        const response = await fetch('/describe', { method: 'POST', body: formData });
        const result = await response.text();
        logDiv.textContent = result;
        if (speakToggle.checked) {
          speechSynthesis.speak(new SpeechSynthesisUtterance(result));
        }
      } catch (err) {
        console.error('Server error:', err);
        logDiv.textContent = 'Server error';
      }
    });
  }
}

// Event listeners
startBtn.addEventListener('click', async () => {
  if (!isRunning) {
    await startWebcam();
    isRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    intervalId = setInterval(detect, parseInt(intervalInput.value));
  }
});

stopBtn.addEventListener('click', () => {
  stopWebcam();
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

intervalInput.addEventListener('change', () => {
  if (isRunning) {
    clearInterval(intervalId);
    intervalId = setInterval(detect, parseInt(intervalInput.value));
  }
});

// Initialize
loadModel();
