import { useEffect } from 'react'

export default function WebcamView({ videoRef, onError }) {
  useEffect(() => {
    let stream
    let active = true
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((s) => {
        if (!active) { s.getTracks().forEach((t) => t.stop()); return }
        stream = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
          videoRef.current.play().catch(onError)
        }
      })
      .catch(onError)
    return () => {
      active = false
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [videoRef, onError])

  return (
    <video
      ref={videoRef}
      playsInline
      muted
      style={{ width: '100%', maxWidth: 640, borderRadius: 8, background: '#000' }}
    />
  )
}
