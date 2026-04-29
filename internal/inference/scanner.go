package inference

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/systalyze/utilyze/internal/gpu"
	"github.com/systalyze/utilyze/internal/inference/procfs"
)

// Scanner reports the current per-GPU inference attribution.
type Scanner interface {
	Scan(ctx context.Context, gpus []int) (map[int]Attribution, error)
}

// Backend resolves a cohort of GPU processes into an endpoint/model attribution.
type Backend interface {
	Name() string
	Discover(ctx context.Context, cohort ProcessCohort) (Endpoint, string, bool, error)
}

type Process struct {
	PID     int
	Stat    procfs.Stat
	Comm    string
	Cmdline []string
	Ports   []int
}

type ProcessCohort struct {
	GPU          int
	SessionID    int
	GPUProcesses []gpu.ProcessInfo
	Processes    []Process
}

type Endpoint struct {
	URL  string
	Port int
}

type Attribution struct {
	GPU       int
	SessionID int
	Backend   string
	ModelID   string
	Endpoint  Endpoint
	LastProbe time.Time
}

// Engine performs per-GPU attribution with caching.
type Engine struct {
	processes gpu.ProcessProvider
	backends  []Backend
	cacheTTL  time.Duration

	mu    sync.Mutex
	cache map[int]*cachedAttribution
}

type sessionKey struct {
	PID       int
	SessionID int
	StartTime uint64
}

type gpuSession struct {
	GPU                 int
	SessionID           int
	GPUProcesses        []gpu.ProcessInfo
	TotalGPUMemoryBytes uint64
	Key                 sessionKey
}

type cachedAttribution struct {
	att       Attribution
	updatedAt time.Time
	key       sessionKey
}

func New(processes gpu.ProcessProvider, backends []Backend, cacheTTL time.Duration) *Engine {
	if processes == nil {
		return nil
	}

	return &Engine{
		processes: processes,
		backends:  backends,
		cacheTTL:  cacheTTL,
		cache:     make(map[int]*cachedAttribution),
	}
}

func (e *Engine) Scan(ctx context.Context, gpus []int) (map[int]Attribution, error) {
	atts := make(map[int]Attribution)
	now := time.Now()

	var scanErrs error
	for _, gpu := range gpus {
		sessions, err := e.currentGPUSessions(gpu)
		if err != nil && !os.IsNotExist(err) {
			scanErrs = errors.Join(scanErrs, err)
		}

		if len(sessions) == 0 {
			e.mu.Lock()
			delete(e.cache, gpu)
			e.mu.Unlock()
			continue
		}

		e.mu.Lock()
		cached := e.cache[gpu]
		e.mu.Unlock()

		if cached != nil && e.cacheStillFresh(cached, now, sessions) {
			atts[gpu] = cached.att
			continue
		}

		att, key, valid, err := e.discoverGPU(ctx, sessions)
		e.mu.Lock()
		if err != nil || !valid {
			scanErrs = errors.Join(scanErrs, err)
			delete(e.cache, gpu)
		} else {
			e.cache[gpu] = &cachedAttribution{
				att:       att,
				updatedAt: now,
				key:       key,
			}
			atts[gpu] = att
		}
		e.mu.Unlock()
	}

	return atts, scanErrs
}

func (e *Engine) currentGPUSessions(gpu int) ([]gpuSession, error) {
	if e == nil || e.processes == nil {
		return nil, errors.New("scanner or gpu process provider is nil")
	}

	procs, err := e.processes.GetComputeProcesses(gpu)
	if err != nil || len(procs) == 0 {
		return nil, err
	}

	var procErrs error
	bySession := make(map[int]*gpuSession)
	for _, proc := range procs {
		stat, err := procfs.StatForPID(proc.PID)
		if err != nil && !os.IsNotExist(err) {
			procErrs = errors.Join(procErrs, err)
			continue
		}

		session := bySession[stat.SessionID]
		if session == nil {
			session = &gpuSession{
				GPU:       gpu,
				SessionID: stat.SessionID,
				Key: sessionKey{
					PID:       proc.PID,
					SessionID: stat.SessionID,
					StartTime: stat.StartTime,
				},
			}
			bySession[stat.SessionID] = session
		}
		session.GPUProcesses = append(session.GPUProcesses, proc)
		session.TotalGPUMemoryBytes += proc.UsedGpuMemoryBytes
	}

	if len(bySession) == 0 {
		return nil, procErrs
	}

	sessions := make([]gpuSession, 0, len(bySession))
	for sessionID, session := range bySession {
		if leaderStat, err := procfs.StatForPID(sessionID); err == nil {
			session.Key = sessionKey{
				PID:       leaderStat.PID,
				SessionID: leaderStat.SessionID,
				StartTime: leaderStat.StartTime,
			}
		} else if !os.IsNotExist(err) {
			procErrs = errors.Join(procErrs, err)
		}
		sessions = append(sessions, *session)
	}

	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].SessionID < sessions[j].SessionID
	})
	return sessions, procErrs
}

func (e *Engine) cacheStillFresh(cached *cachedAttribution, now time.Time, sessions []gpuSession) bool {
	if e.cacheTTL <= 0 || now.Sub(cached.updatedAt) > e.cacheTTL {
		return false
	}

	for _, session := range sessions {
		if session.SessionID != cached.key.SessionID {
			continue
		}
		return session.Key.PID == cached.key.PID &&
			session.Key.SessionID == cached.key.SessionID &&
			session.Key.StartTime == cached.key.StartTime
	}

	return false
}

func (e *Engine) discoverGPU(ctx context.Context, sessions []gpuSession) (Attribution, sessionKey, bool, error) {
	var cohortErrs error
	for _, session := range sessions {
		cohort, err := e.buildProcessCohort(session)
		if err != nil && !os.IsNotExist(err) {
			cohortErrs = errors.Join(cohortErrs, err)
			continue
		}

		for _, backend := range e.backends {
			endpoint, modelID, ok, err := backend.Discover(ctx, cohort)
			if err != nil {
				slog.Debug("inference: backend error", "backend", backend.Name(), "gpu", cohort.GPU, "sid", cohort.SessionID, "err", err)
				continue
			}
			if !ok {
				continue
			}
			return Attribution{
				GPU:       cohort.GPU,
				SessionID: cohort.SessionID,
				Backend:   backend.Name(),
				ModelID:   modelID,
				Endpoint:  endpoint,
				LastProbe: time.Now(),
			}, session.Key, true, cohortErrs
		}
	}

	return Attribution{}, sessionKey{}, false, cohortErrs
}

func (e *Engine) buildProcessCohort(session gpuSession) (ProcessCohort, error) {
	peers, err := procfs.SessionPeers(session.SessionID)
	if err != nil {
		return ProcessCohort{}, err
	}

	gpuProcs := slices.Clone(session.GPUProcesses)
	cohort := ProcessCohort{
		GPU:          session.GPU,
		SessionID:    session.SessionID,
		GPUProcesses: gpuProcs,
		Processes:    make([]Process, 0, len(peers)),
	}

	for _, pid := range peers {
		stat, err := procfs.StatForPID(pid)
		if err != nil {
			continue
		}
		comm, err := procfs.Comm(pid)
		if err != nil {
			comm = stat.Comm
		}
		cmdline, err := procfs.Cmdline(pid)
		if err != nil {
			cmdline = nil
		}
		ports, err := procfs.ListeningPorts(pid)
		if err != nil {
			ports = nil
		}
		cohort.Processes = append(cohort.Processes, Process{
			PID:     pid,
			Stat:    stat,
			Comm:    comm,
			Cmdline: cmdline,
			Ports:   ports,
		})
	}

	sort.Slice(cohort.Processes, func(i, j int) bool { return cohort.Processes[i].PID < cohort.Processes[j].PID })
	return cohort, nil
}
