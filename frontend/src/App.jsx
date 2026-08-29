import { useEffect, useRef, useState } from 'react'
import './styles.css'
import { useStore } from './store'
import { startSession, resetSession, getSessionStatus } from './api'
import { usePoseDetection } from './hooks/usePoseDetection'
import WebcamView from './components/WebcamView'
import PoseOverlay from './components/PoseOverlay'
import FeedbackBanner from './components/FeedbackBanner'
import SessionStats from './components/SessionStats'
import PosePicker from './components/PosePicker'

const features = [
  {
    number: '01',
    title: 'AI pose tracking',
    copy: 'Real-time skeletal tracking turns movement into immediate, practical alignment feedback while you practice.',
  },
  {
    number: '02',
    title: 'Singing bowl guidance',
    copy: 'Immersive bowl tones mark transitions and support breathing rhythms without pulling your attention to another screen.',
  },
  {
    number: '03',
    title: 'Progress that personalizes',
    copy: 'Build a clearer picture of your practice through hold tracking, completed poses, and suggested sequences.',
  },
]

function FeatureIcon({ number }) {
  if (number === '01') {
    return (
      <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <rect x="7" y="13" width="34" height="25" rx="6" />
        <path d="M16 13l3-5h10l3 5M24 20a6 6 0 1 1 0 12 6 6 0 0 1 0-12Z" />
      </svg>
    )
  }

  if (number === '02') {
    return (
      <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <path d="M8 29h32c0 7-7 12-16 12S8 36 8 29Z" />
        <path d="M12 29c1-6 5-10 12-10s11 4 12 10M33 8l6 18M30 9l6-2" />
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path d="M8 38h33M11 34l9-10 7 6 13-16" />
      <path d="M32 14h8v8" />
      <circle cx="11" cy="34" r="2" />
    </svg>
  )
}

export default function App() {
  const videoRef = useRef(null)
  const heroAudioRef = useRef(null)
  const [error, setError] = useState(null)
  const [enabled, setEnabled] = useState(false)
  const [starting, setStarting] = useState(false)
  const [heroSoundOn, setHeroSoundOn] = useState(false)
  const [soundPreference, setSoundPreference] = useState(() => {
    try {
      return localStorage.getItem('auralis-sound-preference')
    } catch {
      return null
    }
  })
  const {
    sessionId, prediction, feedback,
    setSession, setRunning, reset, setSessionStatus,
  } = useStore()

  usePoseDetection(videoRef, enabled, setError)

  useEffect(() => {
    if (!sessionId) return
    const id = setInterval(() => getSessionStatus(sessionId)
      .then(setSessionStatus)
      .catch(() => {}), 2000)
    return () => clearInterval(id)
  }, [sessionId, setSessionStatus])

  useEffect(() => {
    if (soundPreference !== 'accepted') return
    const audio = heroAudioRef.current
    if (!audio) return
    audio.volume = 0.22
    audio.play().then(() => setHeroSoundOn(true)).catch(() => setHeroSoundOn(false))
  }, [soundPreference])

  const handlePick = async (targets) => {
    setStarting(true)
    setError(null)
    try {
      const { session_id, target_poses } = await startSession(targets)
      setSession(session_id, target_poses)
      setEnabled(true)
      setRunning(true)
    } catch (err) {
      setError('The practice studio is unavailable. Make sure the classifier service is running.')
    } finally {
      setStarting(false)
    }
  }

  const handleStop = async () => {
    setEnabled(false)
    setRunning(false)
    if (sessionId) await resetSession(sessionId).catch(() => {})
    reset()
  }

  const toggleHeroSound = async () => {
    const audio = heroAudioRef.current
    if (!audio) return

    if (heroSoundOn) {
      audio.pause()
      setHeroSoundOn(false)
      return
    }

    audio.volume = 0.22
    try {
      await audio.play()
      setHeroSoundOn(true)
      setSoundPreference('accepted')
      localStorage.setItem('auralis-sound-preference', 'accepted')
    } catch {
      setHeroSoundOn(false)
    }
  }

  const acceptHeroSound = async () => {
    setSoundPreference('accepted')
    localStorage.setItem('auralis-sound-preference', 'accepted')
    const audio = heroAudioRef.current
    if (!audio) return
    audio.volume = 0.22
    try {
      await audio.play()
      setHeroSoundOn(true)
    } catch {
      setHeroSoundOn(false)
    }
  }

  const declineHeroSound = () => {
    setSoundPreference('silent')
    localStorage.setItem('auralis-sound-preference', 'silent')
  }

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Auralis home">
          <span className="brand-mark" aria-hidden="true"><img src="/media/auralis-mark.png" alt="" /></span>
          <span>Auralis</span>
        </a>
        <nav className="nav-links" aria-label="Main navigation">
          <a href="#demo">Demo</a>
          <a href="#features">Features</a>
          <a href="#how-it-works">How it works</a>
          <a href="#studio">Live studio</a>
        </nav>
        <a className="button button-small" href="#studio">Start practicing</a>
      </header>

      <main id="top">
        <section className="hero">
          <video className="hero-video" autoPlay loop muted playsInline preload="metadata" aria-hidden="true">
            <source src="/media/woman-singing-bowl.mp4" type="video/mp4" />
          </video>
          <audio ref={heroAudioRef} loop preload="metadata">
            <source src="/media/singing-bowl-strike.mp3" type="audio/mpeg" />
          </audio>
          <div className="hero-wash" />
          {!soundPreference && (
            <div className="sound-consent-backdrop">
              <div className="sound-consent" role="dialog" aria-modal="true" aria-labelledby="sound-consent-title">
                <img src="/media/auralis-mark.png" alt="" />
                <span className="kicker">An immersive welcome</span>
                <h2 id="sound-consent-title">Enter Auralis with sound?</h2>
                <p>A gentle singing-bowl tone accompanies the opening experience. You can turn it off at any time.</p>
                <div className="sound-consent-actions">
                  <button className="button" type="button" onClick={acceptHeroSound}>Enter with sound <span aria-hidden="true">◖))</span></button>
                  <button className="sound-consent-silent" type="button" onClick={declineHeroSound}>Continue silently</button>
                </div>
              </div>
            </div>
          )}
          <div className="hero-content section-wrap">
            <div className="hero-copy">
              <div className="eyebrow"><span /> AI-powered yoga & mindfulness</div>
              <h1>Move with awareness.<br /><em>Flow with confidence.</em></h1>
              <p className="hero-lede">
                Auralis brings real-time pose feedback and immersive singing-bowl guidance
                together—so home practice feels focused, supported, and deeply mindful.
              </p>
              <div className="hero-actions">
                <a className="button hero-primary" href="#studio">Try it now <span aria-hidden="true">→</span></a>
                <a className="text-link" href="#how-it-works"><span className="play">▶</span> See how it works</a>
              </div>
              <div className="hero-proof">
                <div className="avatar-stack" aria-hidden="true">
                  <span>SK</span><span>AM</span><span>RJ</span>
                </div>
                <div><strong>Technology that respects tradition</strong><small>Guidance without disrupting your practice.</small></div>
              </div>
            </div>

            <div className="hero-insight" aria-label="Guided session preview">
              <div className="insight-topline"><span><i /> Guided session</span><b>In sync</b></div>
              <div className="insight-main">
                <span className="insight-icon">⌁</span>
                <div><small>Session transition</small><strong>Singing bowl</strong></div>
              </div>
              <div className="insight-cue"><span>◌</span> Let the next breath begin softly</div>
            </div>
          </div>
          <button className="sound-toggle" type="button" onClick={toggleHeroSound} aria-pressed={heroSoundOn}>
            <span aria-hidden="true">{heroSoundOn ? '◖))' : '◖'}</span>
            {heroSoundOn ? 'Bowl sound on' : soundPreference === 'accepted' ? 'Play bowl sound' : 'Enter with bowl sound'}
          </button>
        </section>

        <section className="trust-strip" aria-label="Product highlights">
          <span>REAL-TIME FEEDBACK</span><i>✦</i><span>8 FOUNDATIONAL POSES</span><i>✦</i>
          <span>SINGING BOWL GUIDANCE</span><i>✦</i><span>MINDFUL BY DESIGN</span>
        </section>

        <section className="demo section-wrap" id="demo">
          <div className="demo-heading">
            <div>
              <span className="kicker">Classifier in motion</span>
              <h2>See movement become<br />meaningful feedback.</h2>
            </div>
            <div className="demo-copy">
              <p>
                This clip was processed frame by frame into the same 33-landmark feature
                vector used by the live studio, then classified by the same Random Forest
                model. The confidence shown is real output, gently smoothed for readability.
              </p>
              <a className="button demo-button" href="#studio">Try the live classifier <span aria-hidden="true">→</span></a>
            </div>
          </div>
          <div className="demo-media">
            <video autoPlay loop muted playsInline controls preload="metadata"
              poster="/media/classified-tree-pose-poster.jpg"
              aria-label="Tree pose video with pose landmarks and live classifier confidence">
              <source src="/media/classified-tree-pose.mp4" type="video/mp4" />
            </video>
            <div className="demo-meta">
              <span><i /> Pre-classified demo</span>
              <small>9 seconds · Tree pose · 33 landmarks per frame</small>
            </div>
          </div>
        </section>

        <section className="studio section-wrap" id="studio">
          <div className="studio-heading">
            <div><span className="kicker">Try the working prototype</span><h2>Your mat. Your camera.<br />Your practice.</h2></div>
            <p>Experience Auralis pose tracking in your browser. Camera frames stay on your device—only anonymous body landmarks reach the classifier.</p>
          </div>

          <div className="practice-app">
            <div className="practice-stage-column">
              <div className="stage">
                {enabled ? (
                  <>
                    <WebcamView videoRef={videoRef} onError={(err) => setError(`Camera error: ${String(err)}`)} />
                    <PoseOverlay landmarks={prediction?.landmarks || null} />
                    <span className="live-pill stage-live"><i /> Analyzing</span>
                  </>
                ) : (
                  <div className="camera-placeholder">
                    <div className="camera-icon">◉</div>
                    <strong>Your live practice appears here</strong>
                    <span>Choose a pose to switch on the camera and classifier.</span>
                  </div>
                )}
              </div>
              {error && <div className="error-message" role="alert">{error}</div>}
              {enabled && <FeedbackBanner feedback={feedback} />}
            </div>

            <aside className="practice-panel">
              <div className="panel-topline">
                <span>Practice session</span>
                <span className={`status-dot ${enabled ? 'is-live' : ''}`}>{enabled ? 'Live' : 'Ready'}</span>
              </div>
              {enabled ? <SessionStats /> : (
                <div className="empty-stats"><strong>Ready when you are.</strong><span>Pick one pose for focused practice or move freely.</span></div>
              )}
              <div className="pose-select">
                <h3>Choose a pose</h3>
                <PosePicker onPick={handlePick} disabled={starting || enabled} />
              </div>
              <button className="reset-button" onClick={handleStop} disabled={!sessionId}>End session</button>
            </aside>
          </div>
        </section>

        <section className="features section-wrap" id="features">
          <div className="section-heading">
            <div><span className="kicker">Movement, sound, and awareness</span><h2>A calmer way to<br />practice at home.</h2></div>
            <p>Auralis combines computer vision with thoughtful sound guidance, so you can focus less on whether you’re doing it right—and more on how it feels.</p>
          </div>
          <div className="feature-grid">
            {features.map((feature) => (
              <article className="feature-card" key={feature.number}>
                <span className="feature-number">{feature.number}</span>
                <div className={`feature-icon feature-icon-${feature.number}`} aria-hidden="true">
                  <FeatureIcon number={feature.number} />
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="how section-wrap" id="how-it-works">
          <div className="how-card">
            <span className="kicker">How it works</span>
            <h2>From camera to cue<br />in the blink of an eye.</h2>
            <div className="steps">
              <div><b>1</b><span><strong>Choose your practice</strong><small>Select a pose, sequence, or mindful free-flow session.</small></span></div>
              <div><b>2</b><span><strong>Step into frame</strong><small>Computer vision maps your movement for instant feedback.</small></span></div>
              <div><b>3</b><span><strong>Move with the moment</strong><small>Follow gentle cues and bowl-led transitions without breaking focus.</small></span></div>
            </div>
          </div>
          <blockquote>
            <span className="quote-mark">“</span>
            <p>Practice isn’t about perfect shapes. It’s about building awareness—one breath at a time.</p>
            <footer>THE AURALIS PHILOSOPHY</footer>
          </blockquote>
        </section>

        <section className="closing-cta section-wrap">
          <div>
            <span className="kicker">A little progress is still progress</span>
            <h2>Meet yourself<br /><em>on the mat.</em></h2>
          </div>
          <a className="button button-light" href="#studio">Begin your practice <span>→</span></a>
        </section>
      </main>

      <footer className="site-footer section-wrap">
        <a className="brand" href="#top"><span className="brand-mark"><img src="/media/auralis-mark.png" alt="" /></span><span>Auralis</span></a>
        <p>Computer vision for a more mindful practice.</p>
        <div className="footer-meta">
          <a href="https://www.pexels.com/video/a-woman-making-music-with-a-tibetan-singing-bowl-6892372/" target="_blank" rel="noreferrer">Hero video by Mikhail Nilov · Pexels</a>
          <a href="https://www.pexels.com/video/8712742/" target="_blank" rel="noreferrer">Classifier demo video by Kampus Production · Pexels</a>
          <a href="https://freesound.org/people/s-light/sounds/415140/" target="_blank" rel="noreferrer">Singing bowl sound by s-light · CC0</a>
          <span>© 2026 Auralis</span>
        </div>
      </footer>
    </div>
  )
}
