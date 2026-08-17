const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
]

export default function PoseOverlay({ landmarks, width = 640, height = 480 }) {
  return (
    <canvas
      width={width}
      height={height}
      ref={(canvas) => {
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, width, height)
        if (!landmarks || landmarks.length < 33) return
        ctx.strokeStyle = 'rgba(80, 220, 120, 0.9)'
        ctx.lineWidth = 3
        POSE_CONNECTIONS.forEach(([a, b]) => {
          const p1 = landmarks[a], p2 = landmarks[b]
          if (!p1 || !p2) return
          ctx.beginPath()
          ctx.moveTo(p1[0] * width, p1[1] * height)
          ctx.lineTo(p2[0] * width, p2[1] * height)
          ctx.stroke()
        })
        ctx.fillStyle = 'rgba(255, 220, 80, 0.95)'
        landmarks.forEach((lm) => {
          ctx.beginPath()
          ctx.arc(lm[0] * width, lm[1] * height, 4, 0, 2 * Math.PI)
          ctx.fill()
        })
      }}
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none', width: '100%', maxWidth: 640 }}
    />
  )
}
