import React, { useEffect, useRef, useState } from 'react';
import * as blazeface from '@tensorflow-models/blazeface';
import '@tensorflow/tfjs';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [alert, setAlert] = useState('');

  // Load BlazeFace model
  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await blazeface.load();
      setModel(loadedModel);
      console.log("✅ BlazeFace model loaded");
    };
    loadModel();
  }, []);

  // Start webcam
  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };
    startVideo();
  }, []);

  // Detect faces
  useEffect(() => {
    const detectFaces = async () => {
      if (!model || !videoRef.current || videoRef.current.readyState !== 4) return;

      const predictions = await model.estimateFaces(videoRef.current, false);
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      if (predictions.length > 0) {
        setAlert(predictions.length > 1 ? `Multiple faces detected: ${predictions.length}` : '');
        predictions.forEach(pred => {
          const [x, y, width, height] = [
            pred.topLeft[0],
            pred.topLeft[1],
            pred.bottomRight[0] - pred.topLeft[0],
            pred.bottomRight[1] - pred.topLeft[1]
          ];
          ctx.strokeStyle = 'lime';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, width, height);
        });
      } else {
        setAlert('No face detected!');
      }
    };

    const interval = setInterval(detectFaces, 500); // Run detection every 0.5 sec
    return () => clearInterval(interval);
  }, [model]);

  return (
    <div style={{ textAlign: 'center', padding: 20 }}>
      <h1>🧠 Real-Time Face Detection</h1>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <video ref={videoRef} autoPlay muted playsInline style={{ width: 640, height: 480 }} />
        <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0 }} />
      </div>
      {alert && <h2 style={{ color: 'red', marginTop: 20 }}>{alert}</h2>}
    </div>
  );
}

export default App;
