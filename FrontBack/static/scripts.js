document.addEventListener('DOMContentLoaded', function() {
  const cameraButton = document.getElementById('cameraButton');
  const retakeButton = document.getElementById('retakeButton');
  const sendButton = document.getElementById('sendButton');
  const videoPreview = document.getElementById('videoPreview');
  const videoPlayback = document.getElementById('videoPlayback');
  const recordingProgress = document.getElementById('recordingProgress');

  let mediaRecorder;
  let videoChunks = [];
  let stream = null;

  function initCamera() {
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(str => {
        stream = str;
        videoPreview.srcObject = stream;
      }).catch(error => {
        console.error("Error accessing media devices:", error);
      });
  }

  cameraButton.addEventListener('click', function() {
    cameraButton.style.display = 'none'; // hide the camera while vid recording
    if (!stream) {
      console.error('Camera not initialized');
      return;
    }
    mediaRecorder = new MediaRecorder(stream);
    videoChunks = [];
    
    mediaRecorder.ondataavailable = function(e) {
      videoChunks.push(e.data);
    };
    
    mediaRecorder.onstop = function(e) {
      const videoBlob = new Blob(videoChunks, { 'type': 'video/mp4' });
      const videoUrl = URL.createObjectURL(videoBlob);
      videoPlayback.src = videoUrl;
      videoPlayback.style.display = 'block'; // Show the playback video
      videoPreview.style.display = 'none'; // Hide the preview video
      
      sendButton.style.display = 'block'; // Show send button
      retakeButton.style.display = 'block'; // Show the retake video button
      cameraButton.style.display = 'none';
      sendButton.onclick = function() {
        const formData = new FormData();
        formData.append('video', videoBlob, 'video.mp4');
        
        fetch('/upload-video', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          console.log(data.message);
          URL.revokeObjectURL(videoUrl); // Clean up the URL object
          videoPlayback.style.display = 'none';
          videoPreview.style.display = 'block';
          sendButton.style.display = 'none'; // Hide send button after sending
          videoPreview.srcObject = null;
          initCamera(); // Reinitialize camera for another recording if needed
        })
        .catch(error => {
          console.error(error);
        });
      };

      retakeButton.onclick = function(){
        URL.revokeObjectURL(videoUrl); // Clean up the URL object
        videoPlayback.style.display = 'none';
        videoPreview.style.display = 'block';
        sendButton.style.display = 'none'; // Hide send button after sending
        retakeButton.style.display = 'none'; // Hide retake button
        cameraButton.style.display = 'inline-block'; // Show the take video button
        videoPreview.srcObject = null;
        initCamera(); // Reinitialize camera for another recording if needed
      };

      stream.getTracks().forEach(track => track.stop());
    };
    
    mediaRecorder.start();
    recordingProgress.style.display = 'block';
    recordingProgress.value = 0;
    const interval = setInterval(() => {
      if (recordingProgress.value < recordingProgress.max) {
        recordingProgress.value += 1;
      } else {
        clearInterval(interval);
      }
    }, 1000);

    // Stop recording after 5 seconds
    setTimeout(() => {
      mediaRecorder.stop();
      recordingProgress.style.display = 'none';
      recordingProgress.value = 0;
    }, 5000);
  });

  initCamera();
});


