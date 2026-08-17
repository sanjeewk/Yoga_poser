import { useEffect, useRef, useState } from 'react'
import './styles.css'
import { useStore } from './store'
import { startSession, resetSession, getSessionStatus } from './api'
import { usePoseDetection } from './hooks/usePoseDetection'
import WebcamView from './components/WebcamView'
import PoseOverlay from './components/PoseOverlay'
import FeedbackBanner from './components/FeedbackBanner'
import SessionStats from './components/SessionStats'
import PosePicker from './components/PosePicker'

export default function App() {
  const videoRef = useRef(null)
  const [error, setError] = useState(null)
  const [enabled, setEnabled] = useState(false)
  const {
    sessionId, prediction, feedback,
    setSession, setRunning, reset, setSessionStatus,
  } = useStore()

  usePoseDetection(videoRef, enabled)

  useEffect(() => {
    if (!sessionId) return
    const id = setInterval(() => getSessionStatus(sessionId)
      .then(setSessionStatus)
      .catch(() => {}), 2000)
    return () => clearInterval(id)
  }, [sessionId, setSessionStatus])

  const handlePick = async (targets) => {
    const { session_id, target_poses } = await startSession(targets)
    setSession(session_id, target_poses)
    setEnabled(true)
    setRunning(true)
  }

  const handleStop = async () => {
    setEnabled(false)
    setRunning(false)
    if (sessionId) await resetSession(sessionId).catch(() => {})
    reset()
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Yoga Poser</h1>
        <div>{sessionId ? `Session ${sessionId.slice(0, 8)}` : 'No session'}</div>
      </div>

      <div className="grid">
        <div className="left-col">
          <div className="stage">
            <WebcamView videoRef={videoRef} onError={setError} />
            <PoseOverlay landmarks={prediction?.landmarks || null} />
          </div>
          {error && <div style={{ color: '#c00' }}>Camera error: {String(error)}</div>}
          <FeedbackBanner feedback={feedback} />
        </div>

        <div className="right-col">
          <SessionStats />
          <div>
            <h3>Pick a pose</h3>
            <PosePicker onPick={handlePick} />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={handleStop} disabled={!sessionId}>Reset session</button>
          </div>
        </div>
      </div>
    </div>
  )
}
