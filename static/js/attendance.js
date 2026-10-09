// Face Recognition Live Attendance Client Script
// Handles WebRTC webcam streaming, client canvas HUD, periodic frame recognition, and audio alerts

(function () {
  let stream = null;
  let isScanning = false;
  let processingFrame = false;
  let scanInterval = null;

  const video = document.getElementById('liveVideo');
  const overlayCanvas = document.getElementById('liveOverlayCanvas');
  const overlayCtx = overlayCanvas.getContext('2d');
  
  const startBtn = document.getElementById('startCamBtn');
  const stopBtn = document.getElementById('stopCamBtn');
  const overlayStartBtn = document.getElementById('overlayStartBtn');
  const standbyOverlay = document.getElementById('standbyOverlay');
  const scannerRadar = document.getElementById('scannerRadar');
  const cameraPulse = document.getElementById('liveCameraPulse');
  const cameraStatusBadge = document.getElementById('cameraStatusBadge');
  const errorBox = document.getElementById('cameraErrorBox');
  const errorMessage = document.getElementById('cameraErrorMessage');

  const thresholdSlider = document.getElementById('thresholdSlider');
  const thresholdValue = document.getElementById('thresholdValue');
  const soundToggle = document.getElementById('soundToggle');

  const statusBanner = document.getElementById('statusBanner');
  const bannerIcon = document.getElementById('bannerIcon');
  const bannerTitle = document.getElementById('bannerTitle');
  const bannerMessage = document.getElementById('bannerMessage');

  const attendeesList = document.getElementById('attendeesList');
  const sessionCountBadge = document.getElementById('sessionCountBadge');
  const feedSearchInput = document.getElementById('feedSearchInput');

  // Track already announced students in current live session to prevent chime spam
  const sessionNotified = new Set();

  // Hidden capture canvas
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');

  // Audio Chime Synthesizer using Web Audio API (zero external assets needed)
  function playSuccessChime() {
    if (!soundToggle || !soundToggle.checked) return;
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc1 = audioCtx.createOscillator();
      const osc2 = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc1.type = 'sine';
      osc2.type = 'sine';
      osc1.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
      osc2.frequency.setValueAtTime(880.00, audioCtx.currentTime + 0.1); // A5

      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(audioCtx.destination);

      osc1.start();
      osc2.start(audioCtx.currentTime + 0.1);
      osc1.stop(audioCtx.currentTime + 0.35);
      osc2.stop(audioCtx.currentTime + 0.35);
    } catch (e) {
      console.warn('AudioContext not allowed or not supported:', e);
    }
  }

  // Threshold slider live display
  if (thresholdSlider && thresholdValue) {
    thresholdSlider.addEventListener('input', (e) => {
      thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Update Status Banner
  function setStatus(type, title, message) {
    statusBanner.className = 'recognition-alert';
    bannerIcon.className = 'bi fs-4';

    if (type === 'success') {
      statusBanner.classList.add('recognition-alert-success');
      bannerIcon.classList.add('bi-check-circle-fill', 'text-success');
    } else if (type === 'warning') {
      statusBanner.classList.add('recognition-alert-warning');
      bannerIcon.classList.add('bi-exclamation-circle-fill', 'text-warning');
    } else {
      statusBanner.classList.add('recognition-alert-info');
      bannerIcon.classList.add('bi-info-circle-fill', 'text-primary');
    }

    bannerTitle.textContent = title;
    bannerMessage.textContent = message;
  }

  // Start Camera Function
  async function startCamera() {
    errorBox.style.display = 'none';
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      video.srcObject = stream;
      await video.play();

      overlayCanvas.width = video.videoWidth || 640;
      overlayCanvas.height = video.videoHeight || 480;
      captureCanvas.width = overlayCanvas.width;
      captureCanvas.height = overlayCanvas.height;

      standbyOverlay.style.display = 'none';
      scannerRadar.style.display = 'block';
      cameraPulse.style.display = 'inline-block';
      cameraStatusBadge.className = 'badge bg-success';
      cameraStatusBadge.textContent = 'Camera Active';

      startBtn.disabled = true;
      stopBtn.disabled = false;
      isScanning = true;

      setStatus('info', 'Webcam Active', 'Looking for student faces. Please look directly into the camera lens.');

      // Start recognition loop (~every 650ms)
      scanInterval = setInterval(captureAndRecognize, 650);

    } catch (err) {
      console.error('Camera access error:', err);
      errorBox.style.display = 'block';
      errorMessage.textContent = `Camera Error: ${err.name} - ${err.message}`;
      setStatus('warning', 'Camera Unavailable', 'Please verify your camera connection and allow browser camera permissions.');
    }
  }

  // Stop Camera Function
  function stopCamera() {
    isScanning = false;
    if (scanInterval) {
      clearInterval(scanInterval);
      scanInterval = null;
    }

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    video.srcObject = null;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    standbyOverlay.style.display = 'flex';
    scannerRadar.style.display = 'none';
    cameraPulse.style.display = 'none';
    cameraStatusBadge.className = 'badge bg-secondary';
    cameraStatusBadge.textContent = 'Camera Stopped';

    startBtn.disabled = false;
    stopBtn.disabled = true;

    setStatus('info', 'Scanner Inactive', 'Click "Start Camera" to resume facial recognition.');
  }

  // Draw detection bounding box and HUD text on canvas
  function drawHUD(faces) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!faces || faces.length === 0) return;

    faces.forEach(face => {
      const [x, y, w, h] = face.box;
      const isRecognized = face.recognized;
      const primaryColor = isRecognized ? '#10b981' : '#f59e0b'; // Green vs Amber

      // Draw bounding box
      overlayCtx.strokeStyle = primaryColor;
      overlayCtx.lineWidth = 3;
      overlayCtx.strokeRect(x, y, w, h);

      // Draw corner brackets
      const bracketLen = 14;
      overlayCtx.lineWidth = 4;
      overlayCtx.beginPath();
      // Top-left
      overlayCtx.moveTo(x, y + bracketLen); overlayCtx.lineTo(x, y); overlayCtx.lineTo(x + bracketLen, y);
      // Top-right
      overlayCtx.moveTo(x + w - bracketLen, y); overlayCtx.lineTo(x + w, y); overlayCtx.lineTo(x + w, y + bracketLen);
      // Bottom-left
      overlayCtx.moveTo(x, y + h - bracketLen); overlayCtx.lineTo(x, y + h); overlayCtx.lineTo(x + bracketLen, y + h);
      // Bottom-right
      overlayCtx.moveTo(x + w - bracketLen, y + h); overlayCtx.lineTo(x + w, y + h); overlayCtx.lineTo(x + w, y + h - bracketLen);
      overlayCtx.stroke();

      // Label background banner
      const labelText = isRecognized
        ? `${face.full_name} (${face.score_pct}%)`
        : `Unknown (${face.score_pct}%)`;

      overlayCtx.font = 'bold 14px "Segoe UI", sans-serif';
      const textWidth = overlayCtx.measureText(labelText).width;
      const tagHeight = 24;
      const tagY = Math.max(0, y - tagHeight);

      overlayCtx.fillStyle = primaryColor;
      overlayCtx.fillRect(x, tagY, textWidth + 18, tagHeight);

      // Label text
      overlayCtx.fillStyle = '#ffffff';
      overlayCtx.fillText(labelText, x + 8, tagY + 17);
    });
  }

  // Capture current frame and post to recognition API
  async function captureAndRecognize() {
    if (!isScanning || processingFrame || !video.videoWidth) return;
    processingFrame = true;

    try {
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const b64Data = captureCanvas.toDataURL('image/jpeg', 0.82);

      const thresholdVal = thresholdSlider ? thresholdSlider.value : 0.38;

      const response = await fetch('/api/recognize_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: b64Data,
          threshold: parseFloat(thresholdVal)
        })
      });

      const data = await response.json();

      if (data.success) {
        drawHUD(data.faces);

        if (data.detected) {
          if (data.recognized && data.primary_result) {
            const student = data.primary_result.student;
            const att = data.primary_result.attendance;

            if (att.already_marked) {
              setStatus('info', 'Already Recorded', att.message);
            } else {
              // Newly marked attendance!
              setStatus('success', 'Attendance Marked!', `Verified: ${student.full_name} (${student.roll_number}) at ${att.attendance_time}`);
              
              if (!sessionNotified.has(student.student_id)) {
                sessionNotified.add(student.student_id);
                playSuccessChime();
                appendAttendeeToFeed(att);
              }
            }
          } else {
            setStatus('warning', 'Unknown Face Detected', 'Face does not match any consented enrolled student profile.');
          }
        }
      }
    } catch (err) {
      console.warn('Frame processing request error:', err);
    } finally {
      processingFrame = false;
    }
  }

  // Prepend newly marked attendee to the side activity list
  function appendAttendeeToFeed(att) {
    const emptyPlaceholder = attendeesList.querySelector('.empty-feed-placeholder');
    if (emptyPlaceholder) {
      emptyPlaceholder.remove();
    }

    const item = document.createElement('div');
    item.className = 'p-3 border-bottom d-flex align-items-center justify-content-between attendee-item animate-fadeIn';
    
    const initials = att.full_name.split(' ').map(p => p[0]).slice(0, 2).join('').toUpperCase();

    item.innerHTML = `
      <div class="d-flex align-items-center gap-3">
        <div class="rounded-circle bg-success text-white d-flex align-items-center justify-content-center fw-bold" style="width: 40px; height: 40px; font-size: 14px;">
          ${initials}
        </div>
        <div>
          <div class="fw-bold text-dark">${att.full_name}</div>
          <small class="text-muted">${att.roll_number} &bull; ${att.student_id}</small>
        </div>
      </div>
      <div class="text-end">
        <span class="badge badge-present mb-1 d-inline-block">
          <i class="bi bi-check2"></i> Present
        </span>
        <div class="text-muted small" style="font-size: 11px;">
          <i class="bi bi-clock me-1"></i>${att.attendance_time}
        </div>
      </div>
    `;

    attendeesList.prepend(item);

    // Update count badge
    if (sessionCountBadge) {
      const current = parseInt(sessionCountBadge.textContent) || 0;
      sessionCountBadge.textContent = `${current + 1} Present`;
    }
  }

  // Live filter for side feed
  if (feedSearchInput) {
    feedSearchInput.addEventListener('input', (e) => {
      const term = e.target.value.toLowerCase().trim();
      const items = attendeesList.querySelectorAll('.attendee-item');
      items.forEach(el => {
        const text = el.textContent.toLowerCase();
        el.style.display = text.includes(term) ? 'flex' : 'none';
      });
    });
  }

  // Event Listeners
  if (startBtn) startBtn.addEventListener('click', startCamera);
  if (overlayStartBtn) overlayStartBtn.addEventListener('click', startCamera);
  if (stopBtn) stopBtn.addEventListener('click', stopCamera);

  // Clean up on page unload
  window.addEventListener('beforeunload', () => {
    stopCamera();
  });

})();
