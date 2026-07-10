import React from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {T} from './theme';

/** Full-bleed dark gradient with two slowly drifting glow orbs. */
export const Bg: React.FC<{children?: React.ReactNode}> = ({children}) => {
  const frame = useCurrentFrame();
  const drift = Math.sin(frame / 90) * 40;
  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(160deg, ${T.bg0}, ${T.bg1})`,
        fontFamily: T.font,
        color: T.ink,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          width: 900,
          height: 900,
          borderRadius: '50%',
          background: T.hot,
          filter: 'blur(180px)',
          opacity: 0.16,
          top: -300 + drift,
          right: -250,
        }}
      />
      <div
        style={{
          position: 'absolute',
          width: 700,
          height: 700,
          borderRadius: '50%',
          background: '#2563eb',
          filter: 'blur(170px)',
          opacity: 0.12,
          bottom: -260 - drift,
          left: -200,
        }}
      />
      {children}
    </AbsoluteFill>
  );
};

export const Kicker: React.FC<{children: React.ReactNode; delay?: number}> = ({
  children,
  delay = 0,
}) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame - delay, [0, 12], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div
      style={{
        fontFamily: T.mono,
        fontSize: 26,
        letterSpacing: '0.35em',
        textTransform: 'uppercase',
        color: T.hot,
        opacity: o,
      }}
    >
      {children}
    </div>
  );
};

/** Springs in from below with fade. */
export const Rise: React.FC<{
  children: React.ReactNode;
  delay?: number;
  distance?: number;
  style?: React.CSSProperties;
}> = ({children, delay = 0, distance = 60, style}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - delay, fps, config: {damping: 200, stiffness: 90}});
  return (
    <div
      style={{
        opacity: s,
        transform: `translateY(${(1 - s) * distance}px)`,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

export const Card: React.FC<{
  children: React.ReactNode;
  accent?: boolean;
  style?: React.CSSProperties;
}> = ({children, accent, style}) => (
  <div
    style={{
      background: accent
        ? 'linear-gradient(180deg, rgba(255,122,26,0.14), rgba(255,122,26,0.05))'
        : T.card,
      border: `2px solid ${accent ? 'rgba(255,122,26,0.55)' : T.cardBorder}`,
      borderRadius: 24,
      padding: '36px 44px',
      boxShadow: '0 18px 60px rgba(0,0,0,0.45)',
      ...style,
    }}
  >
    {children}
  </div>
);

/** Animated integer counter. */
export const Counter: React.FC<{
  from: number;
  to: number;
  start: number;
  duration: number;
  suffix?: string;
  style?: React.CSSProperties;
}> = ({from, to, start, duration, suffix = '', style}) => {
  const frame = useCurrentFrame();
  const v = interpolate(frame, [start, start + duration], [from, to], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.out(Easing.cubic),
  });
  return (
    <span style={{fontVariantNumeric: 'tabular-nums', ...style}}>
      {Math.round(v)}
      {suffix}
    </span>
  );
};

export const Title: React.FC<{children: React.ReactNode; size?: number; delay?: number}> = ({
  children,
  size = 84,
  delay = 0,
}) => (
  <Rise delay={delay}>
    <div style={{fontSize: size, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.08}}>
      {children}
    </div>
    <div
      style={{
        width: 110,
        height: 8,
        borderRadius: 4,
        marginTop: 18,
        background: `linear-gradient(90deg, ${T.hot}, ${T.warm})`,
      }}
    />
  </Rise>
);

export const Wordmark: React.FC<{size?: number}> = ({size = 150}) => (
  <div style={{fontSize: size, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1}}>
    <span style={{color: T.hot}}>Token</span>
    <span style={{color: T.warm}}>Router</span>
  </div>
);
