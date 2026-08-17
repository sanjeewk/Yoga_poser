export default function FeedbackBanner({ feedback }) {
  if (!feedback || feedback.length === 0) {
    return (
      <div style={styles.box({ severity: 'ok' })}>
        <strong>Nice form</strong>
        <span>Hold the pose and breathe.</span>
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {feedback.map((h, i) => (
        <div key={i} style={styles.box({ severity: h.severity })}>
          <strong>{h.severity === 'major' ? 'Adjust' : 'Tip'}</strong>
          <span>{h.cue}</span>
        </div>
      ))}
    </div>
  )
}

const styles = {
  box: ({ severity }) => ({
    padding: '10px 14px',
    borderRadius: 8,
    background: severity === 'major' ? '#ffe3e3' : severity === 'minor' ? '#fff6d6' : '#e3f7e8',
    border: `1px solid ${severity === 'major' ? '#d33' : severity === 'minor' ? '#dc0' : '#3c3'}`,
    display: 'flex',
    gap: 10,
    alignItems: 'center',
  }),
}
