import { useEffect, useRef } from 'react'
import { predict } from '../api'
import { useStore } from '../store'

const FRAME_INTERVAL_MS = 125

export function usePoseDetection(videoRef, enabled) {
  const runningRef = useRef(false)
  const canvasRef = useRef(null)
  const inFlightRef = useRef(false)
  const sessionId = useStore((s) => s.sessionId)
  const setPrediction = useStore((s) => s.setPrediction)

  useEffect(() => {
    if (!enabled || !sessionId) return
    runningRef.current = true
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas')
      canvasRef.current.width = 640
      canvasRef.current.height = 480
    }
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    const tick = async () => {
      if (!runningRef.current) return
      const video = videoRef.current
      if (video && video.readyState >= 2 && !inFlightRef.current) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        inFlightRef.current = true
        canvas.toBlob(async (blob) => {
          try {
            const result = await predict(blob, sessionId)
            if (runningRef.current) setPrediction(result)
          } catch (err) {
            console.error('predict failed', err)
          } finally {
            inFlightRef.current = false
          }
        }, 'image/jpeg', 0.7)
      }
      setTimeout(tick, FRAME_INTERVAL_MS)
    }
    tick()

    return () => { runningRef.current = false }
  }, [enabled, sessionId, videoRef, setPrediction])
}
