import React from 'react';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import {Bakeoff, Idea, Intro, Ladder, Numbers, Outro, Robustness} from './scenes';

const t = () => (
  <TransitionSeries.Transition
    presentation={fade()}
    timing={linearTiming({durationInFrames: 14})}
  />
);

export const Main: React.FC = () => (
  <TransitionSeries>
    <TransitionSeries.Sequence durationInFrames={150}>
      <Intro />
    </TransitionSeries.Sequence>
    {t()}
    <TransitionSeries.Sequence durationInFrames={220}>
      <Idea />
    </TransitionSeries.Sequence>
    {t()}
    <TransitionSeries.Sequence durationInFrames={430}>
      <Ladder />
    </TransitionSeries.Sequence>
    {t()}
    <TransitionSeries.Sequence durationInFrames={300}>
      <Numbers />
    </TransitionSeries.Sequence>
    {t()}
    <TransitionSeries.Sequence durationInFrames={290}>
      <Bakeoff />
    </TransitionSeries.Sequence>
    {t()}
    <TransitionSeries.Sequence durationInFrames={200}>
      <Robustness />
    </TransitionSeries.Sequence>
    {t()}
    <TransitionSeries.Sequence durationInFrames={180}>
      <Outro />
    </TransitionSeries.Sequence>
  </TransitionSeries>
);
