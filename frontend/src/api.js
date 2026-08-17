const BASE = '/api'

async function jsonOrThrow(res) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function getHealth() {
  return fetch(`${BASE}/health`).then(jsonOrThrow)
}

export async function getPoses() {
  return fetch(`${BASE}/poses`).then(jsonOrThrow)
}

export async function startSession(targets = null) {
  return fetch(`${BASE}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_poses: targets }),
  }).then(jsonOrThrow)
}

export async function getSessionStatus(sessionId) {
  return fetch(`${BASE}/session/${sessionId}`).then(jsonOrThrow)
}

export async function resetSession(sessionId) {
  return fetch(`${BASE}/session/${sessionId}/reset`, { method: 'POST' }).then(jsonOrThrow)
}

export async function predict(imageBlob, sessionId) {
  const form = new FormData()
  form.append('image', imageBlob)
  form.append('session_id', sessionId)
  return fetch(`${BASE}/predict`, { method: 'POST', body: form }).then(jsonOrThrow)
}
