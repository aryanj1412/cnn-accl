# CNN-ACCL: Convolutional Neural Network Accelerator

A hardware-software co-design project for CNN acceleration using Verilog hardware implementations and Python utilities. This repository contains FPGA design files, RTL modules for CNN computation, and supporting tools for model acceleration.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Hardware Architecture](#hardware-architecture)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Modules](#hardware-modules)
- [File Descriptions](#file-descriptions)
- [Building the Project](#building-the-project)
- [Contributing](#contributing)
- [License](#license)

## Overview

CNN-ACCL is a CNN accelerator design project that implements convolution, pooling, activation functions, and other neural network operations in hardware using Verilog. The design targets FPGA deployment with Xilinx Vivado as the primary design tool.

### Key Components

- **Verilog RTL**: Hardware implementation of CNN layers and processing elements
- **FPGA Design**: Block diagram and project scripts for Xilinx Vivado
- **Python Support**: Runtime and training utilities for host-side interaction
- **Processing Elements**: Systolic array-based computation architecture

## Repository Structure

cnn-accl/ ├── src/ # Verilog RTL source files │ ├── top.v # Top-level module │ ├── system_top.v # System integration module │ ├── control.v # Control logic (20KB) │ ├── pe.v # Processing element │ ├── pe_array.v # Processing element array │ ├── input_regfile.v # Input register file │ ├── input_buf.v # Input buffer │ ├── input_stream.v # Input streaming interface │ ├── output_buf.v # Output buffer │ ├── out_stream.v # Output streaming interface │ ├── layer_config.v # Layer configuration │ ├── adder_acc.v # Accumulator/adder unit │ ├── maxpool.v # Max pooling layer │ ├── relu_unit.v # ReLU activation │ ├── quantize.v # Quantization module │ ├── bram_sp.v # FPGA block RAM (single port) │ ├── row_out.v # Row-wise output control │ └── README.md │ ├── bd/ # FPGA Block Diagram │ ├── block_design.tcl # Block design TCL script │ ├── design_1.bd # Vivado block design file │ └── design_1.pdf # Block design schematic │ ├── scripts/ # Project scripts │ └── project.tcl # Vivado project creation script │ ├── python/ # Python utilities │ ├── runtime/ # Runtime environment │ └── training/ # Training utilities │ ├── bitstream/ # Generated bitstreams │ ├── LICENSE # MIT License ├── .gitignore # Git ignore rules └── README.md # This file


## Technology Stack

| Component | Technology | Percentage |
|-----------|-----------|-----------|
| Documentation & Notebooks | Jupyter Notebook | 94.1% |
| Hardware Design | Verilog | 3.4% |
| Software Tools | Python | 1.7% |
| Build Scripts & Config | Other | 0.8% |

### Tools Required

- **Xilinx Vivado**: For FPGA synthesis and implementation
- **Verilog Simulator**: ModelSim, VCS, or similar for RTL simulation
- **Python 3.8+**: For runtime and utility scripts

## Hardware Architecture

### System Components

The accelerator is built around a systolic array-based architecture with the following key modules:

**Computation**
- `pe.v`: Individual processing element for multiply-accumulate operations
- `pe_array.v`: 2D array of processing elements for parallel computation

**Memory & Data Movement**
- `input_regfile.v`: Register file for input data storage and management
- `input_buf.v`: Input buffering stage
- `output_buf.v`: Output buffering stage
- `bram_sp.v`: FPGA block RAM instances for weight/feature storage

**Data Interfaces**
- `input_stream.v`: Streaming input interface
- `out_stream.v`: Streaming output interface
- `row_out.v`: Row-based output generation

**Layer Operations**
- `maxpool.v`: Maximum pooling operation
- `relu_unit.v`: ReLU activation function
- `quantize.v`: Quantization for reduced precision
- `adder_acc.v`: Accumulation and addition operations

**Control & Configuration**
- `control.v`: Main control state machine and orchestration
- `layer_config.v`: Layer parameter configuration
- `system_top.v`: System-level integration
- `top.v`: Top-level module with all interfaces

### Data Flow

Input Data ↓ Input Buffer (input_buf.v) ↓ Input Register File (input_regfile.v) ↓ PE Array (pe.v × N) ↓ Adder/Accumulator (adder_acc.v) ↓ MaxPool/ReLU (maxpool.v, relu_unit.v) ↓ Quantize (quantize.v) ↓ Output Buffer (output_buf.v) ↓ Output Stream (out_stream.v)


## Getting Started

### Prerequisites

1. **Xilinx Vivado 2020.1 or later** (for FPGA design)
2. **Verilog simulator** (optional, for RTL verification)
3. **Python 3.8+** (for runtime tools)
4. **Git** for version control

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/aryanj1412/cnn-accl.git
cd cnn-accl

# Navigate to the project directory
cd scripts

# Installation

## Prerequisites

### Required Software

1. **Xilinx Vivado 2020.1 or Later**
   - Download from [Xilinx Downloads](https://www.xilinx.com/support/download.html)
   - Installation should include:
     - Vivado Design Suite
     - Support for target FPGA board

2. **Python 3.8 or Higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python --version
     ```

3. **Git** (for cloning repository)
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation:
     ```bash
     git --version
     ```

4. **Verilog Simulator** (optional, for RTL simulation)
   - ModelSim/QuestaSim (Mentor Graphics)
   - VCS (Synopsys)
   - Verilator (open-source alternative)

### System Requirements

- **OS**: Linux (Ubuntu 18.04+), macOS, or Windows 10+ with WSL2
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Disk Space**: Minimum 20GB for Vivado + project files
- **Processor**: Multi-core processor recommended

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryanj1412/cnn-accl.git
cd cnn-accl

# Installation & Setup Guide

## Table of Contents

- [2. Verify Repository Structure](#2-verify-repository-structure)
- [3. FPGA Project Setup with Vivado](#3-fpga-project-setup-with-vivado)
- [4. Python Environment Setup](#4-python-environment-setup)
- [5. Verify Installation](#5-verify-installation)
- [6. Project File Organization](#6-project-file-organization)
- [Building the Project](#building-the-project)
- [Verilog Simulation Setup (Optional)](#verilog-simulation-setup-optional)
- [Troubleshooting Installation](#troubleshooting-installation)
- [Environment Variables (Optional)](#environment-variables-optional)
- [Verify Complete Installation](#verify-complete-installation)
- [Next Steps](#next-steps)

---

## 2. Verify Repository Structure

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
LICENSE        # MIT License
README.md      # Project documentation
```

---

## 3. FPGA Project Setup with Vivado

### Option A: Using TCL Script (Automated)

```bash
# Navigate to scripts directory
cd scripts

# Launch Vivado and execute project creation script
vivado -source project.tcl &
```

This will:

- Create a new Vivado project
- Add all Verilog source files from `src/` directory
- Import block design from `bd/` directory
- Configure project settings

### Option B: Manual Setup in Vivado

**Step 1: Launch Vivado**

```bash
vivado &
```

**Step 2: Create New Project**

1. Click **File → Create Project**
2. Project name: `cnn-accl`
3. Project location: Repository root directory
4. Project type: **RTL Project**
5. Keep default settings and click **Next**

**Step 3: Add Source Files**

1. Click **Add Files** → Select all files in `src/` directory
2. Copy sources into project
3. Click **Next**

**Step 4: Add Constraints (if available)**

- Skip or add if constraint files exist
- Click **Next**

**Step 5: Select Board**

1. Choose your target FPGA board
2. Click **Next** and **Finish**

**Step 6: Import Block Design**

In the Vivado TCL console, run:

```tcl
source bd/block_design.tcl
```

Or manually recreate from `bd/design_1.bd`.

---

## 4. Python Environment Setup

### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (prompt should show (venv))
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n cnn-accl python=3.8

# Activate environment
conda activate cnn-accl
```

### Install Python Dependencies

```bash
# Navigate to repository root
cd cnn-accl

# If requirements.txt exists:
pip install -r requirements.txt

# Otherwise, install commonly needed packages:
pip install numpy scipy matplotlib scikit-learn
```

---

## 5. Verify Installation

### Check Vivado Installation

```bash
# In Vivado TCL console or terminal
vivado -version
```

### Check Python Installation

```bash
python --version
pip list
```

### Check Verilog Files

```bash
# Count Verilog source files
find src/ -name "*.v" | wc -l
```

> Should output: `18` (number of Verilog modules)

---

## 6. Project File Organization

After setup, your directory should contain:

```
cnn-accl/
├── src/                    # RTL source files
│   ├── top.v
│   ├── system_top.v
│   ├── control.v
│   ├── pe.v
│   ├── pe_array.v
│   ├── input_regfile.v
│   ├── input_buf.v
│   ├── input_stream.v
│   ├── output_buf.v
│   ├── out_stream.v
│   ├── layer_config.v
│   ├── adder_acc.v
│   ├── maxpool.v
│   ├── relu_unit.v
│   ├── quantize.v
│   ├── row_out.v
│   └── bram_sp.v
│
├── bd/
│   ├── block_design.tcl
│   ├── design_1.bd
│   └── design_1.pdf
│
├── scripts/
│   └── project.tcl
│
├── python/
│   ├── runtime/
│   └── training/
│
├── bitstream/              # Will contain generated bitstreams
│
├── cnn-accl.xpr            # Generated Vivado project (after setup)
│
└── LICENSE, README.md, .gitignore
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

```bash
# In Vivado TCL Console:
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
```

Generated bitstream location:

```
cnn-accl.runs/impl_1/design_1.bit
```

Copy to `bitstream/` directory for deployment:

```bash
cp cnn-accl.runs/impl_1/design_1.bit bitstream/design_1.bit
```

---

## Verilog Simulation Setup (Optional)

### Using ModelSim

```bash
# Create work directory
mkdir sim_work
cd sim_work

# Compile Verilog files
vlib work
vlog ../src/*.v

# Run simulation
vsim work.top -do "run -all"
```

### Using Verilator (Open Source)

**Installation:**

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

## Troubleshooting Installation

### Issue: Vivado Not Found

```bash
# Add Vivado to PATH
export PATH="/path/to/vivado/bin:$PATH"

# Verify
vivado -version
```

### Issue: Python Modules Missing

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Git Clone Fails

```bash
# Check internet connection
ping github.com

# Try cloning with specific protocol
git clone --depth 1 https://github.com/aryanj1412/cnn-accl.git
```

### Issue: Vivado Project Creation Fails

```bash
# Check TCL syntax
tclsh scripts/project.tcl

# Or create project manually:
# File → Create Project → Follow GUI steps
```

### Issue: Insufficient Disk Space

```bash
# Check available space
df -h

# Vivado typically requires 20GB+
# Free up space before installation
```

---

## Environment Variables (Optional)

Set permanent environment variables for easier access.

### Linux / macOS

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export XILINX_VIVADO="/path/to/vivado"
export XILINX_HLS="/path/to/vivado_hls"
export PATH="$XILINX_VIVADO/bin:$PATH"
```

Then reload:

```bash
source ~/.bashrc
```

### Windows

Set system environment variables:

1. Right-click **This PC → Properties**
2. Click **Advanced system settings**
3. Click **Environment Variables**
4. Add:
   - Variable name: `XILINX_VIVADO`
   - Variable value: `C:\Xilinx\Vivado\2020.1` (or your version)

---

## Verify Complete Installation

Save the following as `verify_installation.sh` and run it to confirm all components are installed:

```bash
#!/bin/bash

echo "=== CNN-ACCL Installation Verification ==="
echo ""

echo "1. Checking Git..."
git --version || echo "❌ Git not found"

echo ""
echo "2. Checking Python..."
python --version || echo "❌ Python not found"

echo ""
echo "3. Checking Vivado..."
vivado -version 2>/dev/null || echo "❌ Vivado not found"

echo ""
echo "4. Checking Verilog files..."
VERILOG_COUNT=$(find src/ -name "*.v" 2>/dev/null | wc -l)
echo "Found $VERILOG_COUNT Verilog files"

echo ""
echo "5. Checking repository structure..."
[ -d "src" ]      && echo "✓ src/ directory exists"     || echo "❌ src/ missing"
[ -d "bd" ]       && echo "✓ bd/ directory exists"      || echo "❌ bd/ missing"
[ -d "scripts" ]  && echo "✓ scripts/ directory exists" || echo "❌ scripts/ missing"
[ -d "python" ]   && echo "✓ python/ directory exists"  || echo "❌ python/ missing"
[ -f "LICENSE" ]  && echo "✓ LICENSE file exists"       || echo "❌ LICENSE missing"

echo ""
echo "=== Verification Complete ==="
```

Run it with:

```bash
chmod +x verify_installation.sh
./verify_installation.sh
```

---

## Next Steps

After successful installation:

1. **Build the project** — See the [Building the Project](#building-the-project) section above.
2. **Review architecture** — Check `bd/design_1.pdf` for the system diagram.
3. **Simulate RTL** — Run Verilog simulations (optional).
4. **Generate bitstream** — Synthesize and implement for FPGA.
5. **Deploy on FPGA** — Load the bitstream to the target board.

---

## Getting Help

If you encounter issues:

- **Check official documentation:**
  - [Xilinx Vivado Documentation](https://docs.xilinx.com)
  - [Verilog Language Reference](https://ieeexplore.ieee.org/document/1620780)

- **Review repository files:**
  - `bd/design_1.pdf` — System architecture diagram
  - Individual `.v` files — Module-level documentation

- **Open a GitHub issue** with:
  - Installation steps followed
  - Error messages received
  - System configuration
  - Vivado version

---

> **Installation Status:** After completing these steps, your CNN-ACCL project is ready for synthesis and implementation.
