# SMGPU Integration Guide

> How to integrate the SMGPU hardware accelerator with the SMGP Python
> software stack, including HAL setup, HWExecutor backend, memory mapping,
> PCIe driver build, and end-to-end workflows.

---

## 1. Overview of Hardware Acceleration

SMGPU provides a drop-in hardware backend for the SMGP Python library. The
software stack normally executes spectral transforms, HD operations,
topological analysis, and graph rewriting in pure Python (with NumPy/SciPy
acceleration). With the hardware backend, these operations are offloaded to the
FPGA via a PCIe connection, delivering 10–100x speedup on supported kernels.

**Software-only path:**
```
Python (smgp.core) -> NumPy/SciPy -> CPU
```

**Hardware-accelerated path:**
```
Python (smgp.core) -> HWExecutor (HAL) -> PCIe Driver -> FPGA (SMGPU)
```

The transition between paths is transparent: you simply configure the executor
backend at session initialisation.

---

## 2. Setting Up the Python HAL

### 2.1 Prerequisites

- Python 3.10+
- SMGP installed (`pip install smgp`)
- FPGA bitstream loaded and PCIe device accessible at `/dev/smgpu0`
- Linux kernel 5.15+ (for UIO or dedicated PCIe driver)

### 2.2 Install the HAL

```bash
# The HAL is part of the SMGP distribution
pip install smgp

# Verify hardware is accessible
python -c "from smgp.hw.hal import SMGPU_HAL; h = SMGPU_HAL(); print(h.version)"
```

### 2.3 Initialising the HAL

```python
from smgp.hw.hal import SMGPU_HAL

# Connect to the FPGA device
hal = SMGPU_HAL(
    device="/dev/smgpu0",       # PCIe device path
    timeout_ms=5000,            # Default operation timeout
    poll_interval_us=100,       # Status polling interval
    debug=False,                # Enable verbose logging
)

# Print device info
print(f"SMGPU version:  {hal.version}")
print(f"Core clock:     {hal.clock_freq_mhz} MHz")
print(f"HD dimension:   {hal.hd_dim}")
print(f"Frac bits:      {hal.frac_bits}")
print(f"Active engines: {bin(hal.engine_mask)}")
```

### 2.4 HAL API Summary

| Method | Description |
|--------|-------------|
| `execute(opcode, sub_opcode, flags, operand)` | Issue a single instruction and return a result handle |
| `execute_chain(instructions)` | Issue a chained sequence of instructions |
| `read_buffer(address, length)` | Read `length` words from device memory |
| `write_buffer(address, data)` | Write data array to device memory |
| `wait_done(timeout_ms)` | Block until the last instruction completes |
| `poll_status()` | Read engine status bitmask |
| `ack_irq(irq_num)` | Acknowledge an interrupt |
| `reset(engine_mask)` | Soft-reset one or more engines |
| `close()` | Release the device handle |

---

## 3. Using HWExecutor as a Drop-In Backend

The `HWExecutor` class implements the same interface as the software executor
but routes operations to the FPGA. You can swap executors without changing any
application code.

### 3.1 Basic Usage

```python
from smgp.core.graph import SpectralMemoryGraph
from smgp.hw.executor import HWExecutor

# Create a graph with the hardware executor backend
graph = SpectralMemoryGraph(
    hd_dim=10000,
    seed=42,
    executor=HWExecutor(device="/dev/smgpu0"),
)

# All standard operations now run on the FPGA
graph.add_node("Socrates", label="person")
graph.add_node("Plato", label="person")
graph.add_edge("Socrates", "Plato", "taught")

# Spectral analysis is hardware-accelerated
from smgp.core.spectral import SpectralMethods
spectral = SpectralMethods(graph, num_eigenvalues=8)
L = spectral.compute_laplacian(normalized=True)   # runs on FPGA
eigenvalues, eigenvectors = spectral.compute_eigen()  # runs on FPGA
```

### 3.2 Selective Acceleration

You can choose to accelerate only specific operations while keeping others
in software:

```python
from smgp.hw.executor import HWExecutor

# Accelerate only spectral and HD operations
executor = HWExecutor(
    device="/dev/smgpu0",
    accel_ops={"spectral", "hd"},  # only these opcode classes
)

graph = SpectralMemoryGraph(executor=executor)
# Topology and rewrite remain in Python
```

### 3.3 Fallback Strategy

If the FPGA is unavailable, the HWExecutor automatically falls back to the
software executor:

```python
executor = HWExecutor(device="/dev/smgpu0", fallback=True)
# If /dev/smgpu0 does not exist, silently uses software backend
```

---

## 4. MemoryMapper Configuration

The `MemoryMapper` manages the host-to-device memory transfer, translating
NumPy arrays into fixed-point device buffers.

### 4.1 Address Allocation

```python
from smgp.hw.memory_mapper import MemoryMapper

mapper = MemoryMapper(hal)

# Allocate device memory regions
graph_buf = mapper.allocate("graph", size=4096)    # returns base address 0x0000
spectral_buf = mapper.allocate("spectral", size=4096)  # returns base address 0x4000
hd_buf = mapper.allocate("hd", size=313)          # returns base address 0x2000

# Transfer data to device
import numpy as np
adj_matrix = np.random.randn(4096, 4096).astype(np.float32)
mapper.to_device(graph_buf, adj_matrix)  # converts to Q8.24 and uploads

# Transfer results back
result = mapper.from_device(spectral_buf, size=4096, dtype=np.float32)
```

### 4.2 Fixed-Point Conversion

The mapper automatically converts between Python floats and device fixed-point:

| Python Type | Device Format | Conversion |
|-------------|---------------|------------|
| `float32` | Q8.24 | `val * 2^24`, clamped to [-128, 128) |
| `float64` | Q16.16 | `val * 2^16`, clamped to [-32768, 32768) |
| `bool` / `uint8` | bipolar bit | `0 -> +1`, `1 -> -1` |

You can override the default format:

```python
mapper.set_precision(frac_bits=16)  # switch to Q16.16
```

### 4.3 Memory Regions

| Region | Base | Size | Usage |
|--------|------|------|-------|
| Graph Memory | `0x0000` | 8 KB | CSR adjacency, node properties |
| HD Crossbar | `0x2000` | 8 KB | HD vector storage |
| Spectral Buffer | `0x4000` | 8 KB | Eigenvector workspace |
| Topology Buffer | `0x6000` | 8 KB | Distance matrix, barcodes |
| Rewrite Buffer | `0x8000` | 8 KB | Patterns, match results |

---

## 5. Building the PCIe Driver

### 5.1 Prerequisites

- Linux kernel headers (`linux-headers-$(uname -r)`)
- Build tools (`gcc`, `make`, `dkms` if using DKMS)
- Xilinx XDMA or QDMA driver source (provided by Xilinx)

### 5.2 Build Steps

```bash
cd hardware/sw/driver

# Build the kernel module
make

# Load the module (requires root)
sudo insmod smgpu_pci.ko

# Verify the device is registered
lsmod | grep smgpu
ls -la /dev/smgpu0
```

### 5.3 Device Tree (FPGA Side)

The FPGA must expose the SMGPU control registers via a PCIe BAR:

```dts
/ {
    smgpu {
        compatible = "smgp,smgpu-1.0";
        reg = <0x0 0x00000000 0x0 0x10000>;  // 64 KB BAR
        interrupts = <0 64 4>;               // MSI interrupt
        clocks = <&clk_250mhz>;
    };
};
```

### 5.4 IOCTL Interface

The driver exposes these IOCTLs to user space:

| IOCTL | Command | Data |
|-------|---------|------|
| `SMGPU_IOC_EXEC` | Execute instruction | `struct smgpu_ioc_exec { u32 instr; }` |
| `SMGPU_IOC_READ` | Read device memory | `struct smgpu_ioc_rw { u16 addr; u32 *data; }` |
| `SMGPU_IOC_WRITE` | Write device memory | `struct smgpu_ioc_rw { u16 addr; u32 data; }` |
| `SMGPU_IOC_POLL` | Poll status | `struct smgpu_ioc_status { u32 mask; u32 *status; }` |
| `SMGPU_IOC_RESET` | Reset engines | `struct smgpu_ioc_reset { u32 engine_mask; }` |

### 5.5 DKMS Installation (optional)

```bash
# For automatic rebuild across kernel updates
sudo cp -r hardware/sw/driver /usr/src/smgpu-pci-1.0
sudo dkms add -m smgpu-pci -v 1.0
sudo dkms build -m smgpu-pci -v 1.0
sudo dkms install -m smgpu-pci -v 1.0
```

---

## 6. End-to-End Workflow

### 6.1 Complete Example: Spectral Analysis on Hardware

```python
import numpy as np
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.spectral import SpectralMethods
from smgp.hw.executor import HWExecutor
from smgp.hw.memory_mapper import MemoryMapper

# 1. Connect to FPGA
executor = HWExecutor(device="/dev/smgpu0", fallback=True)

# 2. Create a hardware-backed graph
graph = SpectralMemoryGraph(hd_dim=10000, seed=42, executor=executor)

# 3. Build a graph
for i in range(100):
    graph.add_node(f"node_{i}", label="entity")
for i in range(99):
    graph.add_edge(f"node_{i}", f"node_{i+1}", "connects")
graph.add_edge("node_0", "node_50", "connects")

# 4. Run spectral analysis (hardware-accelerated)
spectral = SpectralMethods(graph, num_eigenvalues=8)
L = spectral.compute_laplacian(normalized=True)
eigenvalues, eigenvectors = spectral.compute_eigen()

print(f"Top 8 eigenvalues: {eigenvalues}")

# 5. Run a graph Fourier transform
signal = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(100)])
fourier = spectral.graph_fourier_transform(signal)
print(f"Fourier coefficients: {fourier}")

# 6. Clean up
executor.close()
```

### 6.2 Performance Comparison Script

```python
import time
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.spectral import SpectralMethods
from smgp.hw.executor import HWExecutor

# Software baseline
graph_sw = SpectralMemoryGraph(hd_dim=10000, seed=42)
# ... add nodes/edges ...
spectral_sw = SpectralMethods(graph_sw, num_eigenvalues=64)
t0 = time.perf_counter()
L_sw = spectral_sw.compute_laplacian(normalized=True)
eigenvalues_sw, _ = spectral_sw.compute_eigen()
t_sw = time.perf_counter() - t0

# Hardware accelerated
graph_hw = SpectralMemoryGraph(hd_dim=10000, seed=42, executor=HWExecutor("/dev/smgpu0"))
# ... add same nodes/edges ...
spectral_hw = SpectralMethods(graph_hw, num_eigenvalues=64)
t0 = time.perf_counter()
L_hw = spectral_hw.compute_laplacian(normalized=True)
eigenvalues_hw, _ = spectral_hw.compute_eigen()
t_hw = time.perf_counter() - t0

print(f"Software: {t_sw:.3f}s, Hardware: {t_hw:.3f}s, Speedup: {t_sw/t_hw:.1f}x")
```

### 6.3 Debugging Tips

1. **Enable HAL debug logging:**
   ```python
   hal = SMGPU_HAL(device="/dev/smgpu0", debug=True)
   ```
   This prints every instruction issued, every memory transfer, and every
   status poll to stderr.

2. **Check engine status:**
   ```python
   status = hal.poll_status()
   print(f"Spectral: {status.spectral}, HD: {status.hd}, Topo: {status.topo}")
   ```

3. **Read performance counters:**
   ```python
   perf = hal.read_perf_counters()
   print(f"Cycles: {perf.cycles}, HBM reads: {perf.hbm_reads}, NoC flits: {perf.noc_flits}")
   ```

4. **Verify with golden model:**
   ```bash
   python hardware/sim/models/gold_spectral_model.py
   python hardware/sim/models/gold_hd_model.py
   ```
   These scripts compare hardware output against the NumPy reference and
   report maximum absolute error.

---

## References

- Xilinx (2023). *UG1244: XDMA IP Controller User Guide.*
- Xilinx (2022). *UG1073: Versal ACAP Memory Resources.*
- Cummings, C.E. (2008). "Clock Domain Crossing Design & Verification Using SystemVerilog." *SNUG*.
- Corbet, J., Rubini, A., Kroah-Hartman, G. (2005). *Linux Device Drivers.* 3rd Ed., O'Reilly.
