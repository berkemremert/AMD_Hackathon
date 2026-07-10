import React from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {Bg, Card, Counter, Kicker, Rise, Title, Wordmark} from './components';
import {T} from './theme';

const Center: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill
    style={{justifyContent: 'center', alignItems: 'center', textAlign: 'center', padding: 120}}
  >
    {children}
  </AbsoluteFill>
);

/* ————— Scene 1 · Intro ————— */
export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pop = spring({frame: frame - 8, fps, config: {damping: 14, stiffness: 120, mass: 0.8}});
  return (
    <Bg>
      <Center>
        <Kicker delay={0}>AMD Developer Hackathon · Track 1</Kicker>
        <div style={{transform: `scale(${0.7 + 0.3 * pop})`, opacity: pop, marginTop: 30}}>
          <Wordmark size={170} />
        </div>
        <Rise delay={35}>
          <div style={{fontSize: 44, fontWeight: 300, color: T.muted, marginTop: 26}}>
            the prove-or-escalate agent
          </div>
        </Rise>
        <Rise delay={60}>
          <div style={{fontSize: 38, marginTop: 60, opacity: 0.9}}>
            It never pays for what it can prove.
          </div>
        </Rise>
      </Center>
    </Bg>
  );
};

/* ————— Scene 2 · Every token is a purchase ————— */
export const Idea: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const coins = [0, 1, 2, 3, 4, 5, 6, 7];
  const slam = spring({frame: frame - 120, fps, config: {damping: 12, stiffness: 160}});

  return (
    <Bg>
      <Center>
        <Rise>
          <div style={{fontSize: 76, fontWeight: 800}}>
            Every API token is a <span style={{color: T.hot}}>purchase</span>.
          </div>
        </Rise>

        {/* token coins streaming into the API meter */}
        <div style={{position: 'relative', width: 1200, height: 190, marginTop: 70}}>
          {coins.map((i) => {
            const local = frame - 20 - i * 9;
            const x = interpolate(local, [0, 45], [0, 880], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
              easing: Easing.in(Easing.quad),
            });
            const o = interpolate(local, [0, 6, 40, 46], [0, 1, 1, 0], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            });
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: 40 + x,
                  top: 62,
                  width: 56,
                  height: 56,
                  borderRadius: '50%',
                  background: `linear-gradient(135deg, ${T.hot}, ${T.warm})`,
                  opacity: o,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 800,
                  color: '#14100a',
                  fontSize: 30,
                }}
              >
                t
              </div>
            );
          })}
          <div
            style={{
              position: 'absolute',
              right: 0,
              top: 20,
              width: 240,
              height: 140,
              borderRadius: 24,
              border: `2px solid ${T.cardBorder}`,
              background: T.card,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <div style={{fontFamily: T.mono, fontSize: 22, color: T.muted}}>tokens billed</div>
            <div style={{fontSize: 56, fontWeight: 800, color: T.hot}}>
              <Counter from={0} to={598} start={22} duration={90} />
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: 60,
            transform: `scale(${0.6 + 0.4 * slam})`,
            opacity: slam,
          }}
        >
          <Card accent style={{padding: '30px 60px'}}>
            <div style={{fontSize: 52, fontWeight: 800}}>
              …unless you can <span style={{color: T.warm}}>prove it for free</span>
            </div>
          </Card>
        </div>
      </Center>
    </Bg>
  );
};

/* ————— Scene 3 · The ladder, animated ————— */
const TIERS = [
  {
    k: 'TIER 0 · 0 TOKENS',
    name: 'Plain Go code',
    rows: ['arithmetic AST', '6 puzzle solvers', 'mutation repair', 'proven library'],
    border: 'rgba(255,122,26,0.6)',
  },
  {
    k: 'TIER 1 · 0 TOKENS',
    name: 'Bundled Gemma 4 E2B',
    rows: ['code runs vs prompt tests', 'math recomputed in Go', 'format & length gates'],
    border: 'rgba(255,122,26,0.6)',
  },
  {
    k: 'TIER 2 · MINIMAL SPEND',
    name: 'Fireworks API',
    rows: ['terse prompts', 'reasoning_effort: none', 'prefix-cache affinity'],
    border: 'rgba(100,116,139,0.8)',
  },
];

// task pill journeys: [tier it settles in, settle frame]
const PILLS = [
  {label: 'task A', settles: 0, start: 90, color: T.good, verdict: '✓ proved · 0 tokens'},
  {label: 'task B', settles: 1, start: 190, color: T.good, verdict: '✓ verified · 0 tokens'},
  {label: 'task C', settles: 2, start: 300, color: T.warm, verdict: '23 tokens'},
];

export const Ladder: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const laneX = [210, 810, 1410]; // card centers-ish

  return (
    <Bg>
      <AbsoluteFill style={{padding: '90px 120px'}}>
        <Kicker>Architecture</Kicker>
        <Title size={72} delay={5}>
          Three tiers, one ladder
        </Title>

        <div style={{display: 'flex', gap: 40, marginTop: 70}}>
          {TIERS.map((t, i) => {
            const s = spring({frame: frame - 20 - i * 12, fps, config: {damping: 200}});
            return (
              <div key={t.k} style={{opacity: s, transform: `translateY(${(1 - s) * 80}px)`, flex: 1}}>
                <Card style={{borderColor: t.border, minHeight: 330}}>
                  <div style={{fontFamily: T.mono, fontSize: 22, letterSpacing: '0.2em', color: i < 2 ? T.hot : T.muted}}>
                    {t.k}
                  </div>
                  <div style={{fontSize: 40, fontWeight: 800, margin: '14px 0 20px'}}>{t.name}</div>
                  {t.rows.map((r) => (
                    <div key={r} style={{fontSize: 27, color: T.ink, opacity: 0.9, lineHeight: 1.7}}>
                      {r}
                    </div>
                  ))}
                </Card>
              </div>
            );
          })}
        </div>

        {/* traveling task pills + verdicts */}
        <div style={{position: 'relative', height: 170, marginTop: 55}}>
          {PILLS.map((p, idx) => {
            const local = frame - p.start;
            if (local < 0) return null;
            const x = interpolate(local, [0, 20 + p.settles * 22], [0, laneX[p.settles]], {
              extrapolateRight: 'clamp',
              easing: Easing.out(Easing.cubic),
            });
            const settled = local > 20 + p.settles * 22 + 5;
            const vo = interpolate(local, [25 + p.settles * 22, 37 + p.settles * 22], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            });
            return (
              <div key={p.label} style={{position: 'absolute', top: idx * 56, left: 0}}>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 18,
                    transform: `translateX(${x}px)`,
                  }}
                >
                  <div
                    style={{
                      padding: '8px 26px',
                      borderRadius: 999,
                      background: T.hot,
                      color: '#14100a',
                      fontWeight: 800,
                      fontSize: 26,
                    }}
                  >
                    {p.label}
                  </div>
                  {settled && (
                    <div style={{fontSize: 26, fontWeight: 700, color: p.color, opacity: vo}}>
                      {p.verdict}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </Bg>
  );
};

/* ————— Scene 4 · Measured results ————— */
export const Numbers: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cache = interpolate(frame, [150, 210], [0, 54 / 62], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  const stats = [
    {big: '~2/3', small: 'of tasks answered at\nzero scored tokens'},
    {big: '20 → 7', small: 'calls on the hard set\n(−65%)'},
    {big: '31 → 2', small: 'reasoning tokens with\neffort: none (live)'},
  ];
  return (
    <Bg>
      <Center>
        <Kicker>Measured results</Kicker>
        <div style={{fontSize: 200, fontWeight: 800, letterSpacing: '-0.03em', marginTop: 10}}>
          <span style={{color: T.hot}}>
            <Counter from={59} to={59} start={0} duration={1} />
          </span>
          <span style={{color: T.muted, margin: '0 30px'}}>→</span>
          <span style={{color: T.warm}}>
            <Counter from={59} to={23} start={25} duration={55} />
          </span>
        </div>
        <Rise delay={85}>
          <div style={{fontSize: 42, marginTop: 4}}>
            Fireworks calls on our 64-task eval — <b style={{color: T.hot}}>−61%</b>
          </div>
        </Rise>

        <div style={{display: 'flex', gap: 36, marginTop: 70}}>
          {stats.map((s, i) => {
            const sp = spring({frame: frame - 100 - i * 10, fps, config: {damping: 200}});
            return (
              <div key={s.big} style={{opacity: sp, transform: `translateY(${(1 - sp) * 60}px)`}}>
                <Card style={{width: 400, textAlign: 'center'}}>
                  <div style={{fontSize: 62, fontWeight: 800, color: T.mid}}>{s.big}</div>
                  <div style={{fontSize: 24, color: T.muted, whiteSpace: 'pre-line', marginTop: 8}}>
                    {s.small}
                  </div>
                </Card>
              </div>
            );
          })}
        </div>

        <Rise delay={150}>
          <div style={{width: 900, marginTop: 66}}>
            <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 24, color: T.muted}}>
              <span>prompt prefix served from cache (measured live)</span>
              <span style={{fontFamily: T.mono, color: T.warm}}>54 / 62</span>
            </div>
            <div
              style={{
                height: 18,
                borderRadius: 9,
                background: 'rgba(255,255,255,0.08)',
                marginTop: 12,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${cache * 100}%`,
                  height: '100%',
                  borderRadius: 9,
                  background: `linear-gradient(90deg, ${T.hot}, ${T.warm})`,
                }}
              />
            </div>
          </div>
        </Rise>
      </Center>
    </Bg>
  );
};

/* ————— Scene 5 · Bake-off ————— */
export const Bakeoff: React.FC = () => {
  const frame = useCurrentFrame();
  const rows = [
    {name: 'Gemma 4 E2B (ours)', wrong: 2, color: T.good},
    {name: 'Gemma 3 4B (bigger)', wrong: 12, color: T.bad},
  ];
  const flash = interpolate(frame % 30, [0, 15, 30], [1, 0.55, 1]);
  return (
    <Bg>
      <AbsoluteFill style={{padding: '110px 140px'}}>
        <Kicker>Engineering culture</Kicker>
        <Title size={68} delay={5}>
          We measure. Even when it kills our favorite idea.
        </Title>

        <Rise delay={30}>
          <div style={{fontSize: 32, color: T.muted, marginTop: 40, maxWidth: 1250}}>
            Research suggested a stronger local model. We ran a 54-task bake-off with a
            3-judge panel — and counted <b style={{color: T.ink}}>wrong free answers</b>, the
            metric that loses the accuracy gate.
          </div>
        </Rise>

        <div style={{marginTop: 70}}>
          {rows.map((r, i) => {
            const w = interpolate(frame, [60 + i * 20, 120 + i * 20], [0, (r.wrong / 12) * 1150], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
              easing: Easing.out(Easing.cubic),
            });
            return (
              <div key={r.name} style={{marginBottom: 56}}>
                <div style={{fontSize: 32, marginBottom: 14, fontWeight: 700}}>{r.name}</div>
                <div style={{display: 'flex', alignItems: 'center', gap: 26}}>
                  <div
                    style={{
                      width: Math.max(w, 8),
                      height: 46,
                      borderRadius: 12,
                      background: r.color,
                      opacity: r.wrong > 6 ? flash : 1,
                    }}
                  />
                  <div style={{fontSize: 46, fontWeight: 800, color: r.color}}>
                    {frame > 120 + i * 20 ? r.wrong : ''}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <Rise delay={175}>
          <Card accent style={{maxWidth: 1300}}>
            <div style={{fontSize: 34}}>
              <b>6× more wrong answers.</b> The bigger model lost — calibration to the
              verification gates beats raw strength. <span style={{color: T.warm}}>E2B stays.</span>
            </div>
          </Card>
        </Rise>
      </AbsoluteFill>
    </Bg>
  );
};

/* ————— Scene 6 · Robustness ————— */
const SHIELDS = [
  ['📄', 'skeleton results.json at startup'],
  ['⚛️', 'atomic writes — never half-written'],
  ['🛑', 'SIGTERM flush — partial answers survive'],
  ['🧯', 'per-task panic isolation'],
  ['⏱️', 'pacer degrades gracefully under pressure'],
  ['🧪', 'fuzz-tested: 200k-char prompts, no panics'],
];

export const Robustness: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return (
    <Bg>
      <AbsoluteFill style={{padding: '110px 140px'}}>
        <Kicker>Robustness</Kicker>
        <Title size={68} delay={5}>
          It cannot crash, stall, or emit bad JSON
        </Title>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 30,
            marginTop: 70,
          }}
        >
          {SHIELDS.map(([icon, txt], i) => {
            const s = spring({frame: frame - 25 - i * 8, fps, config: {damping: 15, stiffness: 130}});
            return (
              <div key={txt} style={{opacity: s, transform: `scale(${0.85 + 0.15 * s})`}}>
                <Card style={{display: 'flex', gap: 24, alignItems: 'center', padding: '26px 36px'}}>
                  <div style={{fontSize: 44}}>{icon}</div>
                  <div style={{fontSize: 30}}>{txt}</div>
                </Card>
              </div>
            );
          })}
        </div>
        <Rise delay={95}>
          <div style={{marginTop: 56, fontSize: 30, textAlign: 'center', color: T.muted}}>
            verified under the real limits: <span style={{fontFamily: T.mono, color: T.warm}}>--memory 4g --cpus 2</span>
            {'  '}→ model up in 21 s · 8/8 answered · exit 0
          </div>
        </Rise>
      </AbsoluteFill>
    </Bg>
  );
};

/* ————— Scene 7 · Outro ————— */
export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pop = spring({frame: frame - 6, fps, config: {damping: 16, stiffness: 110}});
  const fadeOut = interpolate(frame, [150, 178], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <Bg>
      <AbsoluteFill style={{opacity: fadeOut}}>
        <Center>
          <div style={{transform: `scale(${0.8 + 0.2 * pop})`, opacity: pop}}>
            <Wordmark size={140} />
          </div>
          <Rise delay={25}>
            <div style={{fontSize: 46, marginTop: 34}}>
              Prove it for free — <span style={{color: T.warm}}>or buy the minimum.</span>
            </div>
          </Rise>
          <Rise delay={50}>
            <div style={{fontSize: 28, color: T.muted, marginTop: 60, lineHeight: 1.8}}>
              Go orchestrator · llama.cpp + Gemma 4 E2B in-container · Fireworks AI escalation
              <br />
              every claim measured & logged in eval/PERF.md
            </div>
          </Rise>
          <Rise delay={70}>
            <div style={{fontFamily: T.mono, fontSize: 30, color: T.hot, marginTop: 56}}>
              github.com/omerdduran/token-router
            </div>
          </Rise>
        </Center>
      </AbsoluteFill>
    </Bg>
  );
};
