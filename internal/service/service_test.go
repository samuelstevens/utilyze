package service

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/systalyze/utilyze/internal/gpu"
)

type fakeCollector struct{}

func (c fakeCollector) Devices() []gpu.Device {
	return []gpu.Device{{ID: 0, Monitored: true}, {ID: 1}}
}

func (c fakeCollector) Poll(ctx context.Context, now time.Time) ([]gpu.GPUSnapshot, error) {
	return []gpu.GPUSnapshot{{
		DeviceID: 0,
		SOL: gpu.SOLSnapshot{
			ComputePct: 42,
			MemoryPct:  7,
			Valid:      true,
		},
	}}, nil
}

func (c fakeCollector) DeviceInfo(deviceID int) (gpu.DeviceInfo, error) {
	return gpu.DeviceInfo{ID: deviceID}, nil
}

func (c fakeCollector) GetComputeProcesses(deviceID int) ([]gpu.ProcessInfo, error) {
	return nil, nil
}

func (c fakeCollector) Close() error {
	return nil
}

func TestRunCollectorObservesSnapshots(t *testing.T) {
	svc := NewService()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	observed := make(chan gpu.MetricsSnapshot, 1)
	go svc.RunCollector(ctx, fakeCollector{}, time.Millisecond, func(snapshot gpu.MetricsSnapshot) {
		observed <- snapshot
		cancel()
	})

	select {
	case snapshot := <-observed:
		if len(snapshot.GPUs) != 1 {
			t.Fatalf("len(snapshot.GPUs) = %d, want 1", len(snapshot.GPUs))
		}
		if snapshot.GPUs[0].SOL.ComputePct != 42 {
			t.Fatalf("compute SOL = %f, want 42", snapshot.GPUs[0].SOL.ComputePct)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("timed out waiting for snapshot")
	}

	if svc.lastInit == nil || !slices.Equal(svc.lastInit.DeviceIDs, []int{0}) {
		t.Fatalf("last init device IDs = %v, want [0]", svc.lastInit)
	}
}
