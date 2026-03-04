import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './ModelBox.css'

const ModelBox = ({ modelId, modelInfo, apiUrl, selectedCamera }) => {
  const [isRunning, setIsRunning] = useState(false)
  const [frame, setFrame] = useState(null)
  const [detections, setDetections] = useState([])
  const [stats, setStats] = useState({
    detection_count: 0,
    class_counts: {},
    fps: 0
  })
  const [loading, setLoading] = useState(false)
  const frameIntervalRef = useRef(null)
  const fpsCounterRef = useRef({ count: 0, startTime: Date.now() })

  const modelColors = {
    model_1: '#FF6B6B',
    model_2: '#4ECDC4',
    model_3: '#45B7D1',
    model_4: '#FFA07A'
  }

  const startWebcam = async () => {
    try {
      setLoading(true)
      await axios.post(`${apiUrl}/webcam/${modelId}/start`, { camera_index: selectedCamera }, {
        headers: { 'Content-Type': 'application/json' }
      })
      setIsRunning(true)
      startFrameCapture()
    } catch (error) {
      console.error('Failed to start webcam:', error)
      alert('Failed to start webcam: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const stopWebcam = async () => {
    try {
      setLoading(true)
      await axios.post(`${apiUrl}/webcam/${modelId}/stop`, {}, {
        headers: { 'Content-Type': 'application/json' }
      })
      setIsRunning(false)
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current)
        frameIntervalRef.current = null
      }
      setFrame(null)
      setDetections([])
    } catch (error) {
      console.error('Failed to stop webcam:', error)
    } finally {
      setLoading(false)
    }
  }

  const startFrameCapture = () => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current)

    frameIntervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`${apiUrl}/webcam/${modelId}/frame`)
        const data = response.data

        setFrame(`data:image/jpeg;base64,${data.frame}`)
        setDetections(data.detections || [])
        setStats({
          detection_count: data.detection_count || 0,
          class_counts: data.class_counts || {},
          fps: calculateFPS()
        })
      } catch (error) {
        console.error('Failed to get frame:', error)
      }
    }, 250) // 250ms interval for 4 models = ~4 FPS per model, smoother overall
  }

  const calculateFPS = () => {
    const now = Date.now()
    const counter = fpsCounterRef.current

    counter.count++
    const elapsed = (now - counter.startTime) / 1000

    if (elapsed >= 1) {
      const fps = counter.count / elapsed
      counter.count = 0
      counter.startTime = now
      return Math.round(fps)
    }

    return 0
  }

  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current)
      }
    }
  }, [])

  if (!modelInfo) {
    return (
      <div className="model-box error">
        <div className="model-header">
          <h2>❌ Model Not Found</h2>
        </div>
      </div>
    )
  }

  const isLoaded = modelInfo.loaded
  const accentColor = modelColors[modelId] || '#64c8ff'

  return (
    <div className="model-box" style={{ borderLeftColor: accentColor }}>
      <div className="model-header">
        <div className="model-title">
          <h2>{modelInfo.name}</h2>
          <span className={`status-badge ${isLoaded ? 'loaded' : 'not-loaded'}`}>
            {isLoaded ? '✓ Ready' : '✗ Failed'}
          </span>
        </div>
        <p className="model-description">{modelInfo.description}</p>
        <p className="model-type">
          Type: <code>{modelInfo.type}</code>
        </p>
      </div>

      <div className="model-content">
        {!isLoaded ? (
          <div className="error-message">
            <p>⚠️ Model failed to load. Check console for errors.</p>
          </div>
        ) : (
          <>
            <div className="video-container">
              {frame ? (
                <img src={frame} alt={`${modelId} feed`} className="video-frame" />
              ) : (
                <div className="placeholder">
                  {isRunning ? (
                    <>
                      <div className="spinner-small"></div>
                      <p>Loading Frame...</p>
                    </>
                  ) : (
                    <>
                      <p>📷 Click Start Webcam</p>
                    </>
                  )}
                </div>
              )}
              <div className="fps-indicator">
                FPS: <span>{stats.fps}</span>
              </div>
            </div>

            <div className="controls">
              <button
                className={`btn ${isRunning ? 'btn-stop' : 'btn-start'}`}
                onClick={isRunning ? stopWebcam : startWebcam}
                disabled={loading}
              >
                {loading ? '⏳ Loading...' : isRunning ? '⏹️ Stop Webcam' : '▶️ Start Webcam'}
              </button>
            </div>

            <div className="detections-panel">
              <div className="detections-header">
                <h3>🎯 Detections</h3>
                <span className="count">{stats.detection_count}</span>
              </div>

              {stats.detection_count > 0 ? (
                <>
                  <div className="class-summary">
                    <p className="summary-title">Classes Detected:</p>
                    <div className="class-grid">
                      {Object.entries(stats.class_counts).map(([className, count]) => (
                        <div key={className} className="class-item" style={{ borderColor: accentColor }}>
                          <span className="class-name">{className}</span>
                          <span className="class-count">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="detections-list">
                    <p className="summary-title">All Detections:</p>
                    <div className="detections-scroll">
                      {detections.map((det, idx) => (
                        <div key={idx} className="detection-item">
                          <span className="det-class" style={{ color: accentColor }}>
                            {det.class}
                          </span>
                          <span className="det-conf">{(det.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="no-detections">
                  <p>No objects detected</p>
                </div>
              )}
            </div>

            <div className="model-classes">
              <p className="summary-title">Available Classes ({modelInfo.classes?.length || 0}):</p>
              <div className="classes-grid">
                {modelInfo.classes && modelInfo.classes.length > 0 ? (
                  modelInfo.classes.slice(0, 6).map((cls, idx) => (
                    <span key={idx} className="class-badge">
                      {cls}
                    </span>
                  ))
                ) : (
                  <span className="class-badge">Loading...</span>
                )}
                {modelInfo.classes && modelInfo.classes.length > 6 && (
                  <span className="class-badge more">+{modelInfo.classes.length - 6} more</span>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default ModelBox
