import { act, render } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import WebcamView from './WebcamView'

describe('WebcamView', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps the same camera stream when its parent rerenders', async () => {
    const stop = vi.fn()
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [{ stop }] })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia },
    })
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()

    const videoRef = createRef()
    const view = render(<WebcamView videoRef={videoRef} onError={vi.fn()} />)
    await act(async () => {})

    view.rerender(<WebcamView videoRef={videoRef} onError={vi.fn()} />)
    await act(async () => {})

    expect(getUserMedia).toHaveBeenCalledTimes(1)
    expect(stop).not.toHaveBeenCalled()

    view.unmount()
    expect(stop).toHaveBeenCalledTimes(1)
  })
})
