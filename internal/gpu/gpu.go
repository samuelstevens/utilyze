package gpu

import (
	"context"
	"time"
)

const (
	// VendorNVIDIA is the stable vendor name for NVIDIA-backed collectors.
	VendorNVIDIA = "nvidia"

	// VendorAMD is the stable vendor name for AMD/ROCm-backed collectors.
	VendorAMD = "amd"
)

// Vendor is the entry point for a GPU vendor implementation.
//
// A vendor owns preflight checks and collector construction. Callers should use
// this interface instead of reaching directly for vendor libraries such as NVML,
// CUPTI, AMD-SMI, or rocprofiler.
type Vendor interface {
	// Name returns a stable vendor identifier, such as VendorNVIDIA or VendorAMD.
	Name() string

	// Ready verifies that this process can collect full Utilyze metrics for the vendor.
	// Implementations may load helper libraries, check profiler permissions, or prompt
	// the user before collection starts.
	Ready() error

	// NewCollector creates a collector for deviceIDs. An empty deviceIDs slice means all
	// devices. A positive interval starts any continuous profiling resources needed by
	// Poll; interval <= 0 creates a non-profiling collector for discovery-only paths.
	NewCollector(deviceIDs []int, interval time.Duration) (Collector, error)
}

// Collector owns the vendor resources needed to inspect and poll a set of GPUs.
//
// Collectors are long-lived while the service or TUI is running. They hide
// vendor-specific handles and return vendor-neutral snapshots to the rest of the app.
type Collector interface {
	ProcessProvider

	// Devices returns all physical devices visible to the vendor. Device.Monitored marks
	// the subset selected for polling.
	Devices() []Device

	// Poll returns the latest samples for monitored devices. It may return fewer
	// snapshots than monitored devices when a device has no valid data for this tick.
	Poll(ctx context.Context, now time.Time) ([]GPUSnapshot, error)

	// DeviceInfo returns stable metadata for a physical device ID.
	DeviceInfo(deviceID int) (DeviceInfo, error)

	// Close releases vendor resources owned by the collector.
	Close() error
}

// ProcessProvider reports GPU compute processes for inference attribution.
type ProcessProvider interface {
	// GetComputeProcesses returns processes currently using compute on deviceID.
	GetComputeProcesses(deviceID int) ([]ProcessInfo, error)
}

// Device describes a physical GPU and whether the collector polls it.
type Device struct {
	ID        int
	Monitored bool
}

// DeviceInfo is stable identifying metadata for a GPU.
type DeviceInfo struct {
	ID   int
	Name string
	UUID string
}

// ProcessInfo describes a process reported by a vendor GPU management library.
type ProcessInfo struct {
	PID                int
	UsedGpuMemoryBytes uint64
	GpuInstanceID      uint32
	ComputeInstanceID  uint32
}

// SOLSnapshot reports speed-of-light utilization percentages.
// ComputePct and MemoryPct are only meaningful when Valid is true.
type SOLSnapshot struct {
	ComputePct float64
	MemoryPct  float64
	Valid      bool
}

// BandwidthSnapshot reports host and peer/fabric bandwidth in bytes per second.
// Fabric fields cover vendor-specific GPU interconnects such as NVLink or Infinity Fabric.
type BandwidthSnapshot struct {
	PCIeTxBps   float64
	PCIeRxBps   float64
	FabricTxBps float64
	FabricRxBps float64
	Valid       bool
}

// CoreActivitySnapshot reports whether GPU compute cores were active.
type CoreActivitySnapshot struct {
	ActivePct float64
	Valid     bool
}

// BasicUtilizationSnapshot reports vendor management-library utilization.
// This is a coarse activity signal, not a substitute for SOL metrics.
type BasicUtilizationSnapshot struct {
	UtilPct float64
	Valid   bool
}

// GPUSnapshot is one vendor-neutral sample for a physical GPU.
type GPUSnapshot struct {
	DeviceID         int
	SOL              SOLSnapshot
	Bandwidth        BandwidthSnapshot
	CoreActivity     CoreActivitySnapshot
	BasicUtilization BasicUtilizationSnapshot
}

// MetricsSnapshot groups GPU samples taken at the same time.
type MetricsSnapshot struct {
	Timestamp time.Time
	GPUs      []GPUSnapshot
}

// MonitoredDeviceIDs extracts the device IDs selected for polling.
func MonitoredDeviceIDs(devices []Device) []int {
	ids := make([]int, 0, len(devices))
	for _, device := range devices {
		if device.Monitored {
			ids = append(ids, device.ID)
		}
	}
	return ids
}
