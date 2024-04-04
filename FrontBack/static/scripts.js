document.addEventListener('DOMContentLoaded', function() {
    const cameraButton = document.getElementById('cameraButton');
    
    cameraButton.addEventListener('click', function() {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          const mediaRecorder = new MediaRecorder(stream);
          const videoChunks = [];
          
          mediaRecorder.ondataavailable = function(e) {
            videoChunks.push(e.data);
          };
          
          mediaRecorder.onstop = function(e) {
            const videoBlob = new Blob(videoChunks, { 'type': 'video/mp4' });
            const formData = new FormData();
            formData.append('video', videoBlob, 'video.mp4');
            
            fetch('/upload-video', {
              method: 'POST',
              body: formData
            })
            .then(response => response.json())
            .then(data => {
              console.log(data.message);
              // Handle the response from the server here
            })
            .catch(error => {
              console.error(error);
              // Handle errors here
            });
            
            // Stop the user's camera stream
            stream.getTracks().forEach(track => track.stop());
          };
          
          mediaRecorder.start();
          
          // Stop recording after 5 seconds
          setTimeout(() => {
            mediaRecorder.stop();
          }, 5000);
        })
        .catch(error => {
          console.error("Error accessing media devices:", error);
        });
    });
  });
  