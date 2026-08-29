import { useStore } from '../store'

export default function SessionStats() {
  const prediction = useStore((s) => s.prediction)
  const holdSeconds = useStore((s) => s.holdSeconds)
  const repCount = useStore((s) => s.repCount)
  const repCountPerPose = useStore((s) => s.repCountPerPose)

  return (
    <div className="session-stats">
      <div className="stat">
        <span className="stat-label">Current pose</span>
        <strong className="stat-value">{prediction?.label || 'Looking…'}</strong>
      </div>
      <div className="stat">
        <span className="stat-label">Confidence</span>
        <strong className="stat-value">{prediction ? `${Math.round((prediction.confidence || 0) * 100)}%` : '—'}</strong>
      </div>
      <div className="stat">
        <span className="stat-label">Steady hold</span>
        <strong className="stat-value">{holdSeconds.toFixed(1)} s</strong>
      </div>
      <div className="stat">
        <span className="stat-label">Reps · total</span>
        <strong className="stat-value">{repCount} · {Object.values(repCountPerPose).reduce((sum, count) => sum + count, 0)}</strong>
      </div>
    </div>
  )
}
