import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'

const PACKAGE_VERSION = '1.0.1'
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${PACKAGE_VERSION}/wasm`
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'

let landmarker = null

async function initialize() {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT, true)
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'CPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  })
}

self.onmessage = async ({ data }) => {
  if (data.type === 'INIT') {
    try {
      await initialize()
      self.postMessage({ type: 'READY' })
    } catch (err) {
      self.postMessage({
        type: 'ERROR',
        error: err instanceof Error ? err.message : String(err),
        fatal: true,
      })
    }
    return
  }

  if (data.type !== 'DETECT') return
  if (!landmarker) {
    data.bitmap.close()
    self.postMessage({ type: 'ERROR', error: 'Pose Landmarker is not initialized' })
    return
  }

  try {
    const result = landmarker.detectForVideo(data.bitmap, data.timestamp)
    const landmarks = result.landmarks[0]?.map((point) => [
      point.x, point.y, point.z, point.visibility,
    ]) || null
    self.postMessage({ type: 'RESULT', landmarks })
  } catch (err) {
    self.postMessage({ type: 'ERROR', error: err instanceof Error ? err.message : String(err) })
  } finally {
    data.bitmap.close()
  }
}
