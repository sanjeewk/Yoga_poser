export default function FeedbackBanner({ feedback }) {
  if (!feedback || feedback.length === 0) {
    return (
      <div className="feedback-list">
        <div className="feedback-item">
        <strong>Nice form</strong>
        <span>Hold the pose and breathe.</span>
        </div>
      </div>
    )
  }
  return (
    <div className="feedback-list">
      {feedback.map((h, i) => (
        <div key={i} className={`feedback-item ${h.severity}`}>
          <strong>{h.severity === 'major' ? 'Adjust' : 'Tip'}</strong>
          <span>{h.cue}</span>
        </div>
      ))}
    </div>
  )
}
