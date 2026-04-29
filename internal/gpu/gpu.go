package gpu

import (
	"context"
	"time"
)

const (
	VendorNVIDIA = "nvidia"
	VendorAMD    = "amd"
)

type Vendor interface {
	Name() string
	Ready() error
	NewCollector(deviceIDs []int, interval time.Duration) (Collector, error)
}

type Collector interface {
	ProcessProvider

	Devices() []Device
	Poll(ctx context.Context, now time.Time) ([]GPUSnapshot, error)
	DeviceInfo(deviceID int) (DeviceInfo, error)
	Close() error
}

type ProcessProvider interface {
	GetComputeProcesses(deviceID int) ([]ProcessInfo, error)
}

type Device struct {
	ID        int
	Monitored bool
}

type DeviceInfo struct {
	ID   int
	Name string
	UUID string
}

type ProcessInfo struct {
	PID                int
	UsedGpuMemoryBytes uint64
	GpuInstanceID      uint32
	ComputeInstanceID  uint32
}

type SOLSnapshot struct {
	ComputePct float64
	MemoryPct  float64
	Valid      bool
}

type BandwidthSnapshot struct {
	PCIeTxBps   float64
	PCIeRxBps   float64
	FabricTxBps float64
	FabricRxBps float64
	Valid       bool
}

type CoreActivitySnapshot struct {
	ActivePct float64
	Valid     bool
}

type BasicUtilizationSnapshot struct {
	UtilPct float64
	Valid   bool
}

type GPUSnapshot struct {
	DeviceID         int
	SOL              SOLSnapshot
	Bandwidth        BandwidthSnapshot
	CoreActivity     CoreActivitySnapshot
	BasicUtilization BasicUtilizationSnapshot
}

type MetricsSnapshot struct {
	Timestamp time.Time
	GPUs      []GPUSnapshot
}

func MonitoredDeviceIDs(devices []Device) []int {
	ids := make([]int, 0, len(devices))
	for _, device := range devices {
		if device.Monitored {
			ids = append(ids, device.ID)
		}
	}
	return ids
}
