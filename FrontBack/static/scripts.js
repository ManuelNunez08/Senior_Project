


document.addEventListener('DOMContentLoaded', function() {
  const cameraButton = document.getElementById('cameraButton');
  const retakeButton = document.getElementById('retakeButton');
  const sendButton = document.getElementById('sendButton');
  const videoPreview = document.getElementById('videoPreview');
  const videoPlayback = document.getElementById('videoPlayback');
  const recordingProgress = document.getElementById('recordingProgress');
  const recordDescription = document.getElementById('recordDescription');

  var socket = io.connect('http://localhost:5000');

  socket.on('processing_done', function(data) {
    console.log(data.message);  // Log the message
    // Hide the loading indicator and show the results button
    document.getElementById('loadingIndicator').style.display = 'none';
    document.getElementById('showResultsButton').style.display = 'block';
  });

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
        videoPlayback.style.display = 'none';
        videoPreview.style.display = 'none';
        sendButton.style.display = 'none';
        retakeButton.style.display = 'none';
        cameraButton.style.display = 'none';
        recordDescription.style.display = 'none';
        


        const formData = new FormData();
        formData.append('video', videoBlob, 'video.mp4');

        // Display the loading indicator
        document.getElementById("loadingScreen").style.display = "block";
        
        fetch('/upload-video', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          console.log(data.message);
          URL.revokeObjectURL(videoUrl); // Clean up the URL object

          if(data.success) {
            window.location.href = "/visualize";
          } else {
            // Handle error
            console.log(data.error);
          }
        })
        .catch(error => {
          console.error(error);
          document.getElementById('loadingIndicator').style.display = 'none';
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


