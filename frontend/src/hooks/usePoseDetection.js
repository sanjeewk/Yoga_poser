import { useEffect } from 'react'
import { predict } from '../api'
import { useStore } from '../store'

const FRAME_INTERVAL_MS = 125

export function usePoseDetection(videoRef, enabled, onError) {
  const sessionId = useStore((s) => s.sessionId)
  const setPrediction = useStore((s) => s.setPrediction)

  useEffect(() => {
    if (!enabled || !sessionId) return
    let active = true
    let timer = null
    let lastTimestamp = 0
    const worker = new Worker(
      new URL('../workers/poseLandmarker.worker.js', import.meta.url),
      { type: 'module' },
    )

    const scheduleNextFrame = () => {
      if (active) timer = setTimeout(captureFrame, FRAME_INTERVAL_MS)
    }

    const captureFrame = async () => {
      if (!active) return
      const video = videoRef.current
      if (!video || video.readyState < 2 || !video.videoWidth) {
        scheduleNextFrame()
        return
      }
      try {
        const bitmap = await createImageBitmap(video)
        if (!active) {
          bitmap.close()
          return
        }
        const now = performance.now()
        const timestamp = now > lastTimestamp ? now : lastTimestamp + 1
        lastTimestamp = timestamp
        worker.postMessage({ type: 'DETECT', bitmap, timestamp }, [bitmap])
      } catch (err) {
        console.error('pose frame capture failed', err)
        scheduleNextFrame()
      }
    }

    worker.onmessage = async ({ data }) => {
      if (!active) return
      if (data.type === 'READY') {
        captureFrame()
        return
      }
      if (data.type === 'RESULT') {
        try {
          const result = await predict(data.landmarks, sessionId)
          if (active) setPrediction(result)
        } catch (err) {
          console.error('predict failed', err)
        } finally {
          scheduleNextFrame()
        }
        return
      }
      if (data.type === 'ERROR') {
        console.error('pose detection failed', data.error)
        if (data.fatal) onError?.('Pose detection could not start in this browser.')
        if (!data.fatal) scheduleNextFrame()
      }
    }
    worker.postMessage({ type: 'INIT' })

    return () => {
      active = false
      clearTimeout(timer)
      worker.terminate()
    }
  }, [enabled, sessionId, videoRef, setPrediction, onError])
}
