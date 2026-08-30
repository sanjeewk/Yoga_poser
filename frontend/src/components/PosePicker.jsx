import { useEffect, useState } from 'react'
import { getPoses } from '../api'

export default function PosePicker({ onPick, disabled = false }) {
  const [poses, setPoses] = useState([])
  useEffect(() => {
    getPoses().then((r) => setPoses(r.poses || [])).catch(() => {})
  }, [])
  return (
    <div className="pose-picker">
      <button disabled={disabled} onClick={() => onPick(null)}>Free practice</button>
      {poses.map((p) => (
        <button disabled={disabled} key={p.key} onClick={() => onPick([p.key])}>{p.display_name}</button>
      ))}
    </div>
  )
}
