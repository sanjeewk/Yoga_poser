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
    status: 'Prototype available',
    title: 'Real-time AI feedback',
    copy: 'Browser-based skeletal tracking turns movement into immediate alignment feedback across eight common yoga poses.',
  },
  {
    number: '02',
    status: 'In development',
    title: 'Physical singing bowl',
    copy: 'An integrated metal singing-bowl player will mark session transitions without pulling attention back to a screen.',
  },
  {
    number: '03',
    status: 'Planned',
    title: 'Companion app & AI coach',
    copy: 'Progress tracking, suggested sequences, and an LLM-based vocal coach will make guidance more personal over time.',
  },
  {
    number: '04',
    status: 'Planned',
    title: 'Classes with real instructors',
    copy: 'Remote live classes will bring human teaching and shared motivation home when getting to the studio is difficult.',
  },
]

const problems = [
  {
    number: '01',
    title: 'No instant correction',
    copy: 'Without feedback in the moment, it is difficult to improve form or catch habits as they develop.',
  },
  {
    number: '02',
    title: 'Subtle misalignment',
    copy: 'Small form breakdowns can go unnoticed, making practice less effective and increasing avoidable strain.',
  },
  {
    number: '03',
    title: 'Practicing alone',
    copy: 'Solo sessions can miss the guidance, confidence, and shared motivation of a studio environment.',
  },
  {
    number: '04',
    title: 'Screens break focus',
    copy: 'Following tutorials on a phone or laptop can interrupt the mindfulness that home practice should create.',
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

  if (number === '03') {
    return (
      <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <rect x="13" y="6" width="22" height="36" rx="5" />
        <path d="M19 31c3-7 7-7 10 0M19 16h10M19 21h7" />
        <circle cx="24" cy="36" r="1" />
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect x="7" y="10" width="34" height="26" rx="5" />
      <circle cx="20" cy="20" r="4" />
      <path d="M13 31c1-5 4-7 7-7s6 2 7 7M31 17l5 3-5 3v-6ZM18 41h12" />
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
      return localStorage.getItem('auralys-sound-preference')
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
      localStorage.setItem('auralys-sound-preference', 'accepted')
    } catch {
      setHeroSoundOn(false)
    }
  }

  const acceptHeroSound = async () => {
    setSoundPreference('accepted')
    localStorage.setItem('auralys-sound-preference', 'accepted')
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
    localStorage.setItem('auralys-sound-preference', 'silent')
  }

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Auralys home">
          <span className="brand-mark" aria-hidden="true"><img src="/media/auralys-mark.png" alt="" /></span>
          <span>Auralys</span>
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
                <img src="/media/auralys-mark.png" alt="" />
                <span className="kicker">An immersive welcome</span>
                <h2 id="sound-consent-title">Enter Auralys with sound?</h2>
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
                Auralys brings real-time pose feedback and immersive singing-bowl guidance
                together—so home practice feels focused, supported, and deeply mindful.
              </p>
              <div className="hero-actions">
                <a className="button hero-primary" href="#studio">Try it now <span aria-hidden="true">→</span></a>
                <a className="text-link" href="#how-it-works"><span className="play">▶</span> See how it works</a>
              </div>
              <div className="hero-proof">
                <div className="avatar-stack" aria-hidden="true">
                  <span>SK</span><span>AS</span><span>TK</span>
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
          <span>90% PROTOTYPE VALIDATION ACCURACY</span><i>✦</i><span>8 COMMON POSES</span><i>✦</i>
          <span>CAMERA FRAMES STAY ON-DEVICE</span><i>✦</i><span>LIVE BROWSER DEMO</span>
        </section>

        <section className="problems section-wrap" id="why-auralys">
          <div className="section-heading">
            <div><span className="kicker">Why Auralys</span><h2>Studio guidance belongs at home.</h2></div>
            <p>Yoga at home is convenient, but the feedback, confidence, and focused atmosphere of a studio are difficult to recreate with a screen alone.</p>
          </div>
          <div className="problem-grid">
            {problems.map((problem) => (
              <article className="problem-card" key={problem.number}>
                <span>{problem.number}</span>
                <h3>{problem.title}</h3>
                <p>{problem.copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="demo section-wrap" id="demo">
          <div className="demo-heading">
            <div>
              <span className="kicker">Classifier in motion</span>
              <h2>Useful feedback for every move, every hold, and every pose.</h2>
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
            <div><span className="kicker">Try the working prototype</span><h2>Your mat.<br />Your camera.<br />Your practice.</h2></div>
            <p>Experience Auralys pose tracking in your browser. Camera frames stay on your device—only anonymous body landmarks reach the classifier.</p>
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

        <section className="product-status section-wrap" id="vision">
          <div className="status-heading">
            <div><span className="kicker">Auralys today and tomorrow</span><h2>A working foundation. A more complete practice ahead.</h2></div>
            <p>The live browser experience is the first proof point for a broader at-home yoga coach built around movement, sound, and human guidance.</p>
          </div>
          <div className="status-grid">
            <article>
              <span className="status-badge status-live">Available now</span>
              <h3>Browser pose prototype</h3>
              <p>Try real-time camera tracking, feedback, hold timing, and classification across eight common poses on this page.</p>
            </article>
            <article>
              <span className="status-badge status-building">In development</span>
              <h3>Physical bowl player</h3>
              <p>3D modelling and PCB design are underway for an integrated metal singing-bowl player that supports session transitions.</p>
            </article>
            <article>
              <span className="status-badge status-planned">Planned</span>
              <h3>Connected coaching</h3>
              <p>A companion app, AI vocal cues, personalized sequences, and remote classes with real instructors complete the product vision.</p>
            </article>
          </div>
        </section>

        <section className="features section-wrap" id="features">
          <div className="section-heading">
            <div><span className="kicker">One connected practice</span><h2>Technology, tradition, and human guidance.</h2></div>
            <p>Auralys is designed as four connected parts. The pose prototype works today; the physical and service layers are clearly marked as in development or planned.</p>
          </div>
          <div className="feature-grid">
            {features.map((feature) => (
              <article className="feature-card" key={feature.number}>
                <div className="feature-topline"><span className="feature-number">{feature.number}</span><span className="feature-status">{feature.status}</span></div>
                <div className={`feature-icon feature-icon-${feature.number}`} aria-hidden="true">
                  <FeatureIcon number={feature.number} />
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="audiences section-wrap" aria-labelledby="audiences-title">
          <div className="audiences-heading">
            <span className="kicker">Built for practice and partnership</span>
            <h2 id="audiences-title"><span>Auralys starts</span> <span>at home – and grows</span> <span>through community.</span></h2>
          </div>
          <div className="audience-grid">
            <article>
              <span className="audience-label">For practitioners</span>
              <h3>Bring more confidence to home practice.</h3>
              <p>Designed for health-conscious adults who want immediate guidance without giving up a calm, mindful environment.</p>
              <button className="button audience-button" type="button" disabled>Early access coming soon</button>
            </article>
            <article>
              <span className="audience-label">For studios & wellness partners</span>
              <h3>Extend great teaching beyond the studio.</h3>
              <p>A future platform for yoga studios, instructors, and wellness programmes seeking scalable at-home support.</p>
              <button className="button audience-button" type="button" disabled>Partner programme coming soon</button>
            </article>
          </div>
        </section>

        <section className="how section-wrap" id="how-it-works">
          <div className="how-card">
            <span className="kicker">How it works</span>
            <h2>From camera to cue in the blink of an eye.</h2>
            <div className="steps">
              <div><b>1</b><span><strong>Choose your practice</strong><small>Select a pose, sequence, or mindful free-flow session.</small></span></div>
              <div><b>2</b><span><strong>Step into frame</strong><small>Computer vision maps your movement for instant feedback.</small></span></div>
              <div><b>3</b><span><strong>Move with the moment</strong><small>Follow gentle cues and bowl-led transitions without breaking focus.</small></span></div>
            </div>
          </div>
          <blockquote>
            <span className="quote-mark">“</span>
            <p>Practice isn’t about perfect shapes. It’s about building awareness—one breath at a time.</p>
            <footer>THE AURALYS PHILOSOPHY</footer>
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
        <a className="brand" href="#top"><span className="brand-mark"><img src="/media/auralys-mark.png" alt="" /></span><span>Auralys</span></a>
        <p>Computer vision for a more mindful practice.</p>
        <div className="footer-meta">
          <a href="https://www.pexels.com/video/a-woman-making-music-with-a-tibetan-singing-bowl-6892372/" target="_blank" rel="noreferrer">Hero video by Mikhail Nilov · Pexels</a>
          <a href="https://www.pexels.com/video/8712742/" target="_blank" rel="noreferrer">Classifier demo video by Kampus Production · Pexels</a>
          <a href="https://freesound.org/people/s-light/sounds/415140/" target="_blank" rel="noreferrer">Singing bowl sound by s-light · CC0</a>
          <span>© 2026 Auralys</span>
        </div>
      </footer>
    </div>
  )
}
