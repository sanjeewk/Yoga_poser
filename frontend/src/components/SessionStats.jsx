import { useStore } from '../store'

export default function SessionStats() {
  const prediction = useStore((s) => s.prediction)
  const holdSeconds = useStore((s) => s.holdSeconds)
  const repCount = useStore((s) => s.repCount)
  const repCountPerPose = useStore((s) => s.repCountPerPose)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div>
        <strong>Current pose:</strong> {prediction?.label || '—'}{' '}
        {prediction && `(${Math.round((prediction.confidence || 0) * 100)}%)`}
      </div>
      <div><strong>Hold:</strong> {holdSeconds.toFixed(1)} s</div>
      <div><strong>Reps this pose:</strong> {repCount}</div>
      <div>
        <strong>Session totals:</strong>{' '}
        {Object.keys(repCountPerPose).length === 0
          ? 'none yet'
          : Object.entries(repCountPerPose).map(([k, v]) => `${k}: ${v}`).join(', ')}
      </div>
    </div>
  )
}
