import { create } from 'zustand'

export const useStore = create((set) => ({
  sessionId: null,
  targetPoses: [],
  prediction: null,
  feedback: [],
  holdSeconds: 0,
  repCount: 0,
  repCountPerPose: {},
  history: [],
  isRunning: false,

  setSession: (sessionId, targetPoses) => set({ sessionId, targetPoses }),
  setRunning: (isRunning) => set({ isRunning }),

  setPrediction: (p) => set({
    prediction: p,
    feedback: p.feedback || [],
    holdSeconds: p.hold_seconds || 0,
    repCount: p.rep_count || 0,
  }),

  setSessionStatus: (s) => set({
    repCountPerPose: s.rep_count_per_pose || {},
    history: s.history || [],
  }),

  reset: () => set({
    sessionId: null,
    targetPoses: [],
    prediction: null,
    feedback: [],
    holdSeconds: 0,
    repCount: 0,
    repCountPerPose: {},
    history: [],
    isRunning: false,
  }),
}))
