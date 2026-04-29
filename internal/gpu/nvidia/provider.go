package nvidia

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/systalyze/utilyze/internal/ffi/cupti"
	"github.com/systalyze/utilyze/internal/ffi/nvml"
	"github.com/systalyze/utilyze/internal/ffi/sampler"
	"github.com/systalyze/utilyze/internal/gpu"

	"golang.org/x/term"
)

type Vendor struct {
	once    sync.Once
	nv      *nvml.Client
	initErr error
}

var _ gpu.Vendor = (*Vendor)(nil)

// NewVendor returns a lazily initialized NVIDIA vendor.
// NVML is loaded on first use, so construction does not fail.
func NewVendor() *Vendor {
	return &Vendor{}
}

func (v *Vendor) Name() string {
	return gpu.VendorNVIDIA
}

func (v *Vendor) Ready() error {
	if err := cupti.EnsureLoaded(); err != nil {
		return err
	}
	if hasCaps, _ := sampler.HasProfilingCapabilities(); hasCaps || os.Getenv("UTLZ_DISABLE_PROFILING_WARNING") == "1" {
		return nil
	}

	fmt.Fprintln(os.Stderr, "Warning: GPU profiling requires CAP_SYS_ADMIN. You will likely need to run with sudo:")
	fmt.Fprintln(os.Stderr, "  sudo utlz")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "If you've disabled the NVIDIA profiling restriction on the host you can ignore this warning. To do so, run:")
	fmt.Fprintln(os.Stderr, "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf")
	fmt.Fprintln(os.Stderr, "Then either reboot, or reload the driver (stops all GPU processes):")
	fmt.Fprintln(os.Stderr, "  sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia")
	fmt.Fprintln(os.Stderr, "")
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		fmt.Fprintln(os.Stderr, "To proceed anyway in non-interactive environments, set UTLZ_DISABLE_PROFILING_WARNING=1")
		return errors.New("gpu profiling requires CAP_SYS_ADMIN")
	}

	fmt.Fprintln(os.Stderr, "Press Enter to continue anyway, or Ctrl-C to quit.")
	fmt.Fprintln(os.Stderr, "To skip this prompt in the future, set UTLZ_DISABLE_PROFILING_WARNING=1")
	if _, err := bufio.NewReader(os.Stdin).ReadString('\n'); err != nil {
		return fmt.Errorf("failed to read confirmation: %w", err)
	}
	return nil
}

func (v *Vendor) NewCollector(deviceIDs []int, interval time.Duration) (gpu.Collector, error) {
	return withLockedOSThread(func() (gpu.Collector, error) {
		nv, err := v.client()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize NVML: %w", err)
		}

		count, err := nv.GetDeviceCount()
		if err != nil {
			return nil, fmt.Errorf("failed to get device count: %w", err)
		}

		monitorIDs := slices.Clone(deviceIDs)
		if len(monitorIDs) == 0 {
			monitorIDs = make([]int, count)
			for i := 0; i < count; i++ {
				monitorIDs[i] = i
			}
		}

		var s *sampler.Sampler
		if interval > 0 {
			s, err = sampler.Init(monitorIDs, sampler.DefaultMetrics, interval)
			if err != nil {
				return nil, err
			}
			monitorIDs = s.InitializedDeviceIDs()
		}

		monitored := make(map[int]struct{}, len(monitorIDs))
		for _, id := range monitorIDs {
			monitored[id] = struct{}{}
		}

		devices := make([]gpu.Device, count)
		for i := 0; i < count; i++ {
			_, ok := monitored[i]
			devices[i] = gpu.Device{ID: i, Monitored: ok}
		}

		return &Collector{
			nv:        nv,
			sampler:   s,
			devices:   devices,
			deviceIDs: monitorIDs,
		}, nil
	})
}

func (v *Vendor) init() error {
	v.once.Do(func() {
		v.nv, v.initErr = nvml.Init()
	})
	return v.initErr
}

func (v *Vendor) client() (*nvml.Client, error) {
	if err := v.init(); err != nil {
		return nil, err
	}
	return v.nv, nil
}

type Collector struct {
	nv        *nvml.Client
	sampler   *sampler.Sampler
	devices   []gpu.Device
	deviceIDs []int
}

var _ gpu.Collector = (*Collector)(nil)

func (c *Collector) Devices() []gpu.Device {
	return slices.Clone(c.devices)
}

func (c *Collector) Poll(ctx context.Context, now time.Time) ([]gpu.GPUSnapshot, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	gpus := make([]gpu.GPUSnapshot, 0, len(c.deviceIDs))
	for _, deviceID := range c.deviceIDs {
		snapshot, ok := c.pollDevice(deviceID, now)
		if ok {
			gpus = append(gpus, snapshot)
		}
	}
	return gpus, nil
}

func (c *Collector) pollDevice(deviceID int, now time.Time) (gpu.GPUSnapshot, bool) {
	snapshot := gpu.GPUSnapshot{DeviceID: deviceID}
	hasData := false

	if c.sampler != nil {
		perf, err := c.sampler.Poll(deviceID)
		if err == nil {
			if perf.ComputeSOLPct != nil && perf.MemorySOLPct != nil {
				snapshot.SOL.ComputePct = *perf.ComputeSOLPct
				snapshot.SOL.MemoryPct = *perf.MemorySOLPct
				snapshot.SOL.Valid = true
				hasData = true
			}
			if perf.SMActivePct != nil {
				snapshot.CoreActivity.ActivePct = *perf.SMActivePct
				snapshot.CoreActivity.Valid = true
				hasData = true
			}
		}
	}

	utilization, err := c.nv.PollUtilization(deviceID, now)
	if err == nil && utilization.GPUUtilPct != nil {
		snapshot.BasicUtilization.UtilPct = *utilization.GPUUtilPct
		snapshot.BasicUtilization.Valid = true
		hasData = true
	}

	bandwidth, err := c.nv.PollBandwidth(deviceID, now)
	if err == nil {
		hasBandwidth := false
		if bandwidth.PCIeTxBps != nil && bandwidth.PCIeRxBps != nil {
			snapshot.Bandwidth.PCIeTxBps = *bandwidth.PCIeTxBps
			snapshot.Bandwidth.PCIeRxBps = *bandwidth.PCIeRxBps
			hasBandwidth = true
		}
		if bandwidth.NVLinkTxBps != nil && bandwidth.NVLinkRxBps != nil {
			snapshot.Bandwidth.FabricTxBps = *bandwidth.NVLinkTxBps
			snapshot.Bandwidth.FabricRxBps = *bandwidth.NVLinkRxBps
			hasBandwidth = true
		}
		if hasBandwidth {
			snapshot.Bandwidth.Valid = true
			hasData = true
		}
	}

	return snapshot, hasData
}

func (c *Collector) DeviceInfo(deviceID int) (gpu.DeviceInfo, error) {
	uuid, err := c.nv.GetDeviceUUID(deviceID)
	if err != nil {
		return gpu.DeviceInfo{}, err
	}
	name, err := c.nv.GetDeviceName(deviceID)
	if err != nil {
		return gpu.DeviceInfo{}, err
	}
	return gpu.DeviceInfo{ID: deviceID, Name: name, UUID: uuid}, nil
}

func (c *Collector) GetComputeProcesses(deviceID int) ([]gpu.ProcessInfo, error) {
	processes, err := c.nv.GetComputeProcesses(deviceID)
	if err != nil {
		return nil, err
	}
	out := make([]gpu.ProcessInfo, len(processes))
	for i, proc := range processes {
		out[i] = gpu.ProcessInfo{
			PID:                proc.PID,
			UsedGpuMemoryBytes: proc.UsedGpuMemoryBytes,
			GpuInstanceID:      proc.GpuInstanceID,
			ComputeInstanceID:  proc.ComputeInstanceID,
		}
	}
	return out, nil
}

func (c *Collector) Close() error {
	if c.sampler == nil {
		return nil
	}
	return c.sampler.Close()
}

func withLockedOSThread[T any](fn func() (T, error)) (T, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return fn()
}
