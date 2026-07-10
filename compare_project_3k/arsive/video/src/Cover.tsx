import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Bg, Wordmark} from './components';
import {T} from './theme';

const Chip: React.FC<{k: string; name: string; dim?: boolean}> = ({k, name, dim}) => (
  <div
    style={{
      background: dim ? 'rgba(148,163,184,0.07)' : 'rgba(255,122,26,0.10)',
      border: `2px solid ${dim ? 'rgba(100,116,139,0.7)' : 'rgba(255,122,26,0.55)'}`,
      borderRadius: 18,
      padding: '18px 34px',
      textAlign: 'center',
    }}
  >
    <div
      style={{
        fontFamily: T.mono,
        fontSize: 19,
        letterSpacing: '0.18em',
        color: dim ? T.muted : T.hot,
      }}
    >
      {k}
    </div>
    <div style={{fontSize: 27, fontWeight: 700, marginTop: 6}}>{name}</div>
  </div>
);

const Arrow: React.FC = () => (
  <div style={{fontSize: 34, color: T.muted, fontWeight: 700}}>→</div>
);

export const Cover: React.FC = () => (
  <Bg>
    <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center', paddingBottom: 40}}>
      <div
        style={{
          fontFamily: T.mono,
          fontSize: 25,
          letterSpacing: '0.35em',
          textTransform: 'uppercase',
          color: T.hot,
          marginBottom: 34,
        }}
      >
        AMD Developer Hackathon · Track 1
      </div>

      <Wordmark size={165} />

      <div style={{fontSize: 42, fontWeight: 300, color: T.muted, marginTop: 22}}>
        the prove-or-escalate agent
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 26,
          marginTop: 66,
        }}
      >
        <Chip k="TIER 0 · FREE" name="plain Go code" />
        <Arrow />
        <Chip k="TIER 1 · FREE" name="local Gemma, verified" />
        <Arrow />
        <Chip k="TIER 2 · MINIMAL" name="Fireworks API" dim />
      </div>

      <div
        style={{
          marginTop: 60,
          padding: '16px 44px',
          borderRadius: 999,
          background: `linear-gradient(90deg, ${T.hot}, ${T.warm})`,
          color: '#14100a',
          fontSize: 30,
          fontWeight: 800,
        }}
      >
        two-thirds of tasks answered at zero scored tokens
      </div>
    </AbsoluteFill>
  </Bg>
);
