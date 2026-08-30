import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { predict } from '../api'
import { useStore } from '../store'
import { usePoseDetection } from './usePoseDetection'

vi.mock('../api', () => ({ predict: vi.fn() }))

describe('usePoseDetection', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    useStore.getState().reset()
  })

  it('sends browser-detected landmarks instead of camera images', async () => {
    let worker
    class FakeWorker {
      constructor() {
        worker = this
        this.postMessage = vi.fn()
        this.terminate = vi.fn()
      }

      emit(data) {
        return this.onmessage({ data })
      }
    }

    const bitmap = { close: vi.fn() }
    vi.stubGlobal('Worker', FakeWorker)
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap))
    predict.mockResolvedValue({ label: 'vrksasana', confidence: 0.92, landmarks: [] })
    useStore.setState({ sessionId: 'session-1' })
    const videoRef = {
      current: { readyState: 4, videoWidth: 640 },
    }

    const view = renderHook(() => usePoseDetection(videoRef, true))
    expect(worker.postMessage).toHaveBeenCalledWith({ type: 'INIT' })

    await act(async () => {
      await worker.emit({ type: 'READY' })
      await Promise.resolve()
    })
    expect(createImageBitmap).toHaveBeenCalledWith(videoRef.current)
    expect(worker.postMessage).toHaveBeenLastCalledWith(
      expect.objectContaining({ type: 'DETECT', bitmap }),
      [bitmap],
    )

    const landmarks = Array.from({ length: 33 }, () => [0.1, 0.2, -0.1, 0.99])
    await act(async () => {
      await worker.emit({ type: 'RESULT', landmarks })
    })
    expect(predict).toHaveBeenCalledWith(landmarks, 'session-1')
    expect(useStore.getState().prediction.label).toBe('vrksasana')

    view.unmount()
    expect(worker.terminate).toHaveBeenCalledOnce()
  })
})
