import { useEffect, useState } from 'react'
import { getPoses } from '../api'

export default function PosePicker({ onPick }) {
  const [poses, setPoses] = useState([])
  useEffect(() => {
    getPoses().then((r) => setPoses(r.poses || [])).catch(() => {})
  }, [])
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      <button onClick={() => onPick(null)}>Free practice</button>
      {poses.map((p) => (
        <button key={p.key} onClick={() => onPick([p.key])}>{p.display_name}</button>
      ))}
    </div>
  )
}
