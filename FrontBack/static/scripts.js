


document.addEventListener('DOMContentLoaded', function() {

  console.log("Script loaded");
  if (document.getElementById('is-loading-page')) {
    console.log("Loading page detected, starting status check");
    checkProcessingStatus();
  } else {
    console.log("Not loading page, skipping status check");
  }

  const cameraButton = document.getElementById('cameraButton');
  const retakeButton = document.getElementById('retakeButton');
  const sendButton = document.getElementById('sendButton');
  const videoPreview = document.getElementById('videoPreview');
  const videoPlayback = document.getElementById('videoPlayback');
  const recordingProgress = document.getElementById('recordingProgress');
  const recordDescription = document.getElementById('recordDescription');

  const videoFormats = [
    'video/mp4; codecs="avc1.42E01E, mp4a.40.2"', // MP4 with H.264/AAC
    'video/webm; codecs="vp9,opus"', // WebM with VP9/Opus
    'video/webm; codecs="vp8,opus"', // WebM with VP8/Opus
    'video/webm' // Fallback to default WebM
  ];

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

  function findSupportedMimeType() {
    for (let format of videoFormats) {
      if (MediaRecorder.isTypeSupported(format)) {
        console.log("Using MIME type:", format); // Log the selected MIME type
        return format;
      }
    }
    return null; // No supported type found
  }

  function checkProcessingStatus() {
    fetch('/status')
    .then(response => response.json())
    .then(data => {
        console.log('Processing status:', data.status);
        if (data.status === 'completed') {
            window.location.href = "/visualize"; // Redirect when processing is completed
        } else if (data.status === 'error') {
            window.location.href = "/record-video"; // Redirect on error
        } else {
            // Continue polling if status is 'processing' or any other non-final state
            setTimeout(checkProcessingStatus, 2000); // Poll every 2 seconds
        }
    })
    .catch(error => {
        console.error('Failed to fetch processing status:', error);
        setTimeout(checkProcessingStatus, 2000); // Retry polling on fetch error
    });
}

  cameraButton.addEventListener('click', function() {
    cameraButton.style.display = 'none'; // hide the camera while vid recording
    if (!stream) {
      console.error('Camera not initialized');
      return;
    }
    let options = { mimeType: findSupportedMimeType() };
    if (!options.mimeType) {
        console.error('No supported video format found.');
        return;
    }

    mediaRecorder = new MediaRecorder(stream, options);
    videoChunks = [];
    
    mediaRecorder.ondataavailable = function(e) {
      videoChunks.push(e.data);
    };
    
    mediaRecorder.onstop = function(e) {
      const videoBlob = new Blob(videoChunks, { type: options.mimeType });
      const videoUrl = URL.createObjectURL(videoBlob);
      videoPlayback.src = videoUrl;
      videoPlayback.style.display = 'block'; // Show the playback video
      videoPreview.style.display = 'none'; // Hide the preview video
      
      sendButton.style.display = 'block'; // Show send button
      retakeButton.style.display = 'block'; // Show the retake video button
      cameraButton.style.display = 'none';


      sendButton.onclick = function() {
        // Collect the video data
        const formData = new FormData();
        formData.append('video', videoBlob, 'video.mp4');
    
        // Redirect to the loading page immediately
        window.location.href = "/loading";
    
        // After redirecting to the loading page, start the upload process
        fetch('/upload-video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message); // Logging the server response message
    
            // Conditional redirection based on the response from the server
            if (data.success) {
                window.location.href = "/visualize";
            } else {
                console.error('Error processing video:', data.error);
                window.location.href = "/record-video"; // Redirect back to recording if error
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            window.location.href = "/record-video"; // Redirect on fetch error
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


