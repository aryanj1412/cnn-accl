# CNN-ACCL: Convolutional Neural Network Accelerator

A hardware-software co-design project for CNN acceleration using Verilog hardware implementations and Python utilities. This repository contains FPGA design files, RTL modules for CNN computation, and supporting tools for model acceleration.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Hardware Architecture](#hardware-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Verilog Simulation Setup (Optional)](#verilog-simulation-setup-optional)
- [Troubleshooting](#troubleshooting)
- [Environment Variables (Optional)](#environment-variables-optional)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

CNN-ACCL is a CNN accelerator design project that implements convolution, pooling, activation functions, and other neural network operations in hardware using Verilog. The design targets FPGA deployment with Xilinx Vivado as the primary design tool.

### Key Components

- **Verilog RTL** — Hardware implementation of CNN layers and processing elements
- **FPGA Design** — Block diagram and project scripts for Xilinx Vivado
- **Python Support** — Runtime and training utilities for host-side interaction
- **Processing Elements** — Systolic array-based computation architecture

---

## Repository Structure

```
cnn-accl/
├── src/                    # Verilog RTL source files
│   ├── top.v               # Top-level module
│   ├── system_top.v        # System integration module
│   ├── control.v           # Control logic
│   ├── pe.v                # Processing element
│   ├── pe_array.v          # Processing element array
│   ├── input_regfile.v     # Input register file
│   ├── input_buf.v         # Input buffer
│   ├── input_stream.v      # Input streaming interface
│   ├── output_buf.v        # Output buffer
│   ├── out_stream.v        # Output streaming interface
│   ├── layer_config.v      # Layer configuration
│   ├── adder_acc.v         # Accumulator/adder unit
│   ├── maxpool.v           # Max pooling layer
│   ├── relu_unit.v         # ReLU activation
│   ├── quantize.v          # Quantization module
│   ├── bram_sp.v           # FPGA block RAM (single port)
│   └── row_out.v           # Row-wise output control
│
├── bd/                     # FPGA Block Diagram
│   ├── block_design.tcl    # Block design TCL script
│   ├── design_1.bd         # Vivado block design file
│   └── design_1.pdf        # Block design schematic
│
├── scripts/
│   └── project.tcl         # Vivado project creation script
│
├── python/                 # Python utilities
│   ├── runtime/            # Runtime environment
│   └── training/           # Training utilities
│
├── bitstream/              # Generated bitstreams (initially empty)
├── LICENSE
├── .gitignore
└── README.md
```

---

## Technology Stack

| Component | Technology | Percentage |
|---|---|---|
| Documentation & Notebooks | Jupyter Notebook | 94.1% |
| Hardware Design | Verilog | 3.4% |
| Software Tools | Python | 1.7% |
| Build Scripts & Config | Other | 0.8% |

### Tools Required

- **Xilinx Vivado 2020.1 or later** — For FPGA synthesis and implementation
- **Python 3.8+** — For runtime and utility scripts
- **Verilog Simulator** — ModelSim, VCS, or Verilator (optional, for RTL simulation)

---

## Hardware Architecture

The accelerator is built around a systolic array-based architecture with the following key modules:

**Computation**
- `pe.v` — Individual processing element for multiply-accumulate operations
- `pe_array.v` — 2D array of processing elements for parallel computation

**Memory & Data Movement**
- `input_regfile.v` — Register file for input data storage and management
- `input_buf.v` — Input buffering stage
- `output_buf.v` — Output buffering stage
- `bram_sp.v` — FPGA block RAM instances for weight/feature storage

**Data Interfaces**
- `input_stream.v` — Streaming input interface
- `out_stream.v` — Streaming output interface
- `row_out.v` — Row-based output generation

**Layer Operations**
- `maxpool.v` — Maximum pooling operation
- `relu_unit.v` — ReLU activation function
- `quantize.v` — Quantization for reduced precision
- `adder_acc.v` — Accumulation and addition operations

**Control & Configuration**
- `control.v` — Main control state machine and orchestration
- `layer_config.v` — Layer parameter configuration
- `system_top.v` — System-level integration
- `top.v` — Top-level module with all interfaces

### Data Flow

```
Input Data
    ↓
Input Buffer        (input_buf.v)
    ↓
Input Register File (input_regfile.v)
    ↓
PE Array            (pe.v × N)
    ↓
Adder/Accumulator   (adder_acc.v)
    ↓
MaxPool / ReLU      (maxpool.v, relu_unit.v)
    ↓
Quantize            (quantize.v)
    ↓
Output Buffer       (output_buf.v)
    ↓
Output Stream       (out_stream.v)
```

---

## Prerequisites

### Required Software

1. **Xilinx Vivado 2020.1 or later**
   - Download from [Xilinx Downloads](https://www.xilinx.com/support/download.html)
   - Installation should include the Vivado Design Suite and support for your target FPGA board

2. **Python 3.8 or higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python --version
     ```

3. **Git**
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation:
     ```bash
     git --version
     ```

4. **Verilog Simulator** *(optional, for RTL simulation)*
   - ModelSim / QuestaSim (Mentor Graphics)
   - VCS (Synopsys)
   - Verilator (open-source)

### System Requirements

- **OS** — Linux (Ubuntu 18.04+), macOS, or Windows 10+ with WSL2
- **RAM** — Minimum 8 GB (16 GB+ recommended)
- **Disk Space** — Minimum 20 GB for Vivado and project files
- **Processor** — Multi-core processor recommended

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryanj1412/cnn-accl.git
cd cnn-accl
```

### 2. Verify Repository Structure

```bash
ls -la
```

You should see:

```
src/           # Verilog source files
bd/            # Block diagram files
scripts/       # Vivado project scripts
python/        # Python utilities
bitstream/     # Generated bitstreams (initially empty)
LICENSE
README.md
```

### 3. FPGA Project Setup with Vivado

#### Option A: Using TCL Script (Automated)

```bash
cd scripts
vivado -source project.tcl &
```

This will:
- Create a new Vivado project
- Add all Verilog source files from `src/`
- Import block design from `bd/`
- Configure project settings

#### Option B: Manual Setup in Vivado

**Step 1: Launch Vivado**

```bash
vivado &
```

**Step 2: Create New Project**

1. Click **File → Create Project**
2. Project name: `cnn-accl`
3. Project location: Repository root directory
4. Project type: **RTL Project**
5. Click **Next**

**Step 3: Add Source Files**

1. Click **Add Files** → Select all files in `src/`
2. Enable **Copy sources into project**
3. Click **Next**

**Step 4: Add Constraints (if available)**

- Skip or add constraint files if they exist, then click **Next**

**Step 5: Select Board**

1. Choose your target FPGA board
2. Click **Next** and **Finish**

**Step 6: Import Block Design**

In the Vivado TCL console, run:

```tcl
source bd/block_design.tcl
```

Or manually recreate from `bd/design_1.bd`.

### 4. Python Environment Setup

#### Option A: Using Virtual Environment (Recommended)

```bash
python3 -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### Option B: Using Conda

```bash
conda create -n cnn-accl python=3.8
conda activate cnn-accl
```

#### Install Dependencies

```bash
# If requirements.txt exists:
pip install -r requirements.txt

# Otherwise:
pip install numpy scipy matplotlib scikit-learn
```

### 5. Verify Installation

```bash
# Check Vivado
vivado -version

# Check Python
python --version
pip list

# Count Verilog source files (should output 18)
find src/ -name "*.v" | wc -l
```

### 6. Full Verification Script

Save the following as `verify_installation.sh` and run it:

```bash
#!/bin/bash

echo "=== CNN-ACCL Installation Verification ==="

echo "1. Checking Git..."
git --version || echo "❌ Git not found"

echo "2. Checking Python..."
python --version || echo "❌ Python not found"

echo "3. Checking Vivado..."
vivado -version 2>/dev/null || echo "❌ Vivado not found"

echo "4. Checking Verilog files..."
VERILOG_COUNT=$(find src/ -name "*.v" 2>/dev/null | wc -l)
echo "Found $VERILOG_COUNT Verilog files"

echo "5. Checking repository structure..."
[ -d "src" ]     && echo "✓ src/ exists"     || echo "❌ src/ missing"
[ -d "bd" ]      && echo "✓ bd/ exists"      || echo "❌ bd/ missing"
[ -d "scripts" ] && echo "✓ scripts/ exists" || echo "❌ scripts/ missing"
[ -d "python" ]  && echo "✓ python/ exists"  || echo "❌ python/ missing"
[ -f "LICENSE" ] && echo "✓ LICENSE exists"  || echo "❌ LICENSE missing"

echo "=== Verification Complete ==="
```

```bash
chmod +x verify_installation.sh
./verify_installation.sh
```

---

## Building the Project

### Synthesis

```bash
# In Vivado TCL Console:
launch_runs synth_1
wait_on_run synth_1
```

Or via GUI: **Flow Navigator → Synthesis → Run Synthesis**

### Implementation

```bash
# In Vivado TCL Console:
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
```

Or via GUI: **Flow Navigator → Implementation → Run Implementation**

### Bitstream Generation

The generated bitstream will be located at:

```
cnn-accl.runs/impl_1/design_1.bit
```

Copy it to the `bitstream/` directory for deployment:

```bash
cp cnn-accl.runs/impl_1/design_1.bit bitstream/design_1.bit
```

---

## Verilog Simulation Setup (Optional)

### Using ModelSim

```bash
mkdir sim_work
cd sim_work

vlib work
vlog ../src/*.v

vsim work.top -do "run -all"
```

### Using Verilator (Open Source)

**Install:**

```bash
# Ubuntu/Debian
sudo apt-get install verilator

# macOS
brew install verilator
```

**Simulate:**

```bash
cd sim
verilator --cc --exe --trace ../src/top.v
cd obj_dir
make -f Vtop.mk
./Vtop
```

---

## Troubleshooting

### Vivado Not Found

```bash
export PATH="/path/to/vivado/bin:$PATH"
vivado -version
```

### Python Modules Missing

```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Git Clone Fails

```bash
git clone --depth 1 https://github.com/aryanj1412/cnn-accl.git
```

### Vivado Project Creation Fails

```bash
# Check TCL syntax
tclsh scripts/project.tcl
```

Or recreate the project manually via **File → Create Project** in the Vivado GUI.

### Insufficient Disk Space

```bash
df -h
# Vivado typically requires 20 GB+
```

---

## Environment Variables (Optional)

### Linux / macOS

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export XILINX_VIVADO="/path/to/vivado"
export XILINX_HLS="/path/to/vivado_hls"
export PATH="$XILINX_VIVADO/bin:$PATH"
```

Reload:

```bash
source ~/.bashrc
```

### Windows

1. Right-click **This PC → Properties**
2. Click **Advanced system settings → Environment Variables**
3. Add:
   - Variable name: `XILINX_VIVADO`
   - Variable value: `C:\Xilinx\Vivado\2020.1` (or your installed version)

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of your changes.

---

## License

This project is licensed under the [MIT License](LICENSE).
