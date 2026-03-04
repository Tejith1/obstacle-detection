import React, { useEffect, useState } from 'react'
import axios from 'axios'
import ModelBox from './components/ModelBox'
import './App.css'

const API_URL = 'http://localhost:5000/api'

function App() {
  const [models, setModels] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState(0)
  const [selectedModel, setSelectedModel] = useState(null)
  const [camerasLoading, setCamerasLoading] = useState(false)

  useEffect(() => {
    fetchModels()
    fetchCameras()
  }, [])

  const fetchModels = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_URL}/models`)
      setModels(response.data)
      // Set first loaded model as default
      const firstModelKey = Object.keys(response.data)[0]
      setSelectedModel(firstModelKey)
      setError(null)
    } catch (err) {
      setError(`Failed to load models: ${err.message}`)
      console.error('Model loading error:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchCameras = async () => {
    try {
      setCamerasLoading(true)
      const response = await axios.get(`${API_URL}/cameras`)
      setCameras(response.data.cameras || [])
      if (response.data.cameras && response.data.cameras.length > 0) {
        setSelectedCamera(response.data.cameras[0].index)
      }
    } catch (err) {
      console.error('Failed to load cameras:', err)
      setCameras([{ index: 0, name: 'Camera 0', available: true }])
    } finally {
      setCamerasLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="app-container loading-container">
        <div className="spinner"></div>
        <p>Loading models...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app-container error-container">
        <div className="error-box">
          <h2>⚠️ Error</h2>
          <p>{error}</p>
          <button onClick={fetchModels}>Retry</button>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🤖 Ensemble Models Detection Dashboard</h1>
        <p>Real-time Multi-Model Object Detection System</p>
      </header>

      <div className="controller-bar">
        <div className="selector-group">
          <label htmlFor="model-select">🤖 Select Model:</label>
          <select
            id="model-select"
            value={selectedModel || ''}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="selector-dropdown"
          >
            {Object.entries(models).map(([modelId, modelInfo]) => (
              <option key={modelId} value={modelId}>
                {modelInfo.name}
              </option>
            ))}
          </select>
        </div>

        <div className="selector-group">
          <label htmlFor="camera-select">📷 Select Camera:</label>
          <select
            id="camera-select"
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(Number(e.target.value))}
            className="selector-dropdown"
          >
            {cameras.map((cam) => (
              <option key={cam.index} value={cam.index}>
                {cam.name}
              </option>
            ))}
          </select>
        </div>

        <div className="stats-info">
          <span className="stat-label">Status:</span>
          <span className="stat-value status-ok">● Online</span>
        </div>
      </div>

      <div className="single-model-container">
        {selectedModel && models[selectedModel] && (
          <ModelBox
            modelId={selectedModel}
            modelInfo={models[selectedModel]}
            apiUrl={API_URL}
            selectedCamera={selectedCamera}
          />
        )}
      </div>

      <footer className="app-footer">
        <p>
          Built with React | Backend: Flask + Ultralytics YOLO | Single-Model Mode (Low Power)
        </p>
      </footer>
    </div>
  )
}

export default App
