package router

import (
	"sync/atomic"
	"time"
)

// VerifyMode is the verification depth the time budget currently affords.
type VerifyMode int

const (
	// ModeFull: full-CoT verification samples — minimum escalations, hence
	// minimum scored tokens. The default whenever time allows.
	ModeFull VerifyMode = iota
	// ModeBrief: brief-reasoning samples (~25% volume) when the projection
	// says full verification won't fit.
	ModeBrief
	// ModeOff: emergency — skip verification and escalation entirely,
	// answer everything with the best local attempt.
	ModeOff
)

func (m VerifyMode) String() string {
	switch m {
	case ModeBrief:
		return "brief"
	case ModeOff:
		return "off"
	}
	return "full"
}

// Pacer projects the finish time from observed throughput and picks the
// deepest verification mode that still fits the deadline.
type Pacer struct {
	start    time.Time
	deadline time.Time
	total    int64
	done     atomic.Int64
}

func NewPacer(deadline time.Time, total int) *Pacer {
	return &Pacer{start: time.Now(), deadline: deadline, total: int64(total)}
}

func (p *Pacer) TaskDone() {
	if p != nil {
		p.done.Add(1)
	}
}

func (p *Pacer) Mode() VerifyMode {
	if p == nil {
		return ModeFull
	}
	done := p.done.Load()
	if done < 3 {
		return ModeFull // not enough throughput data yet
	}
	elapsed := time.Since(p.start).Seconds()
	rate := float64(done) / elapsed // tasks per second, parallelism included
	remaining := float64(p.total - done)
	projected := remaining / rate
	left := time.Until(p.deadline).Seconds()
	switch {
	case projected < left*0.70:
		return ModeFull
	case projected < left*0.95:
		return ModeBrief
	default:
		return ModeOff
	}
}
