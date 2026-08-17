import { describe, it, expect } from 'vitest'
import { useStore } from '../store'

describe('store', () => {
  it('starts with no session', () => {
    expect(useStore.getState().sessionId).toBeNull()
  })

  it('setPrediction updates prediction and feedback', () => {
    useStore.getState().setPrediction({
      label: 'tadasana',
      confidence: 0.9,
      feedback: [{ joint: 'left_knee', cue: 'Straighten', severity: 'major' }],
      hold_seconds: 1.5,
      rep_count: 0,
    })
    const s = useStore.getState()
    expect(s.prediction.label).toBe('tadasana')
    expect(s.feedback).toHaveLength(1)
    expect(s.holdSeconds).toBe(1.5)
  })

  it('reset clears session state', () => {
    useStore.getState().setPrediction({ label: 'tadasana', confidence: 0.9, feedback: [], hold_seconds: 2, rep_count: 1 })
    useStore.getState().reset()
    const s = useStore.getState()
    expect(s.sessionId).toBeNull()
    expect(s.prediction).toBeNull()
  })
})
