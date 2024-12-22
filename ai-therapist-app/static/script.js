
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const submitButton = document.getElementById('submit');
const messageInput = document.getElementById('message');
const responseDiv = document.getElementById('response');

// Start video stream
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    console.log("OWO");
    video.srcObject = stream;
  });

// Handle form submission
submitButton.addEventListener('click', async () => {
  // Take a snapshot from the video feed
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const snapshot = canvas.toDataURL('image/jpeg'); // Get image data as Base64

  const message = messageInput.value;

  // Send the message and snapshot to the server
  const response = await fetch('/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, snapshot })
  });

  const data = await response.json();
  responseDiv.textContent = `Therapist: ${data.response}`;
});
