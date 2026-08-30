import { useEffect, useRef } from 'react'

export default function WebcamView({ videoRef, onError }) {
  const onErrorRef = useRef(onError)
  onErrorRef.current = onError

  useEffect(() => {
    let stream
    let active = true
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((s) => {
        if (!active) { s.getTracks().forEach((t) => t.stop()); return }
        stream = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
          videoRef.current.play().catch((error) => onErrorRef.current?.(error))
        }
      })
      .catch((error) => onErrorRef.current?.(error))
    return () => {
      active = false
      if (stream) stream.getTracks().forEach((t) => t.stop())
    }
  }, [videoRef])

  return (
    <video
      ref={videoRef}
      playsInline
      muted
    />
  )
}
