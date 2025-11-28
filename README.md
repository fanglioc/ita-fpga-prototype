# Immutable Tensor Architecture (ITA) - Public Release

This repository contains the complete FPGA prototype implementation and analytical models for the paper:

**"The Immutable Tensor Architecture: A Pure Dataflow Approach for Secure, Energy-Efficient AI Inference"**

## Overview

The Immutable Tensor Architecture (ITA) treats neural network weights as physical circuit topology rather than data, eliminating the memory hierarchy entirely. This prototype demonstrates the concept on a Xilinx Zynq-7020 FPGA.

## Hardware Requirements

- **FPGA Board:** Digilent Zybo Z7-20 (xc7z020clg400-1)
  - Available for ~$200 USD
  - Xilinx Zynq-7020 SoC
  - 53,200 LUTs, 13,300 CARRY4 cells
  
- **Development Tools:**
  - Vivado Design Suite 2020.2 or later (WebPACK edition is free)
  - Python 3.7+ with numpy, matplotlib

## Repository Structure

```
public_release/
├── rtl/                    # SystemVerilog RTL files
│   ├── ita_top.sv         # Top-level module
│   ├── expand_engine.sv   # 64→128 layer (baseline)
│   ├── contract_engine.sv # 128→64 layer (hardwired)
│   └── uart_tx.sv         # UART transmitter for output
├── scripts/               # Build and verification scripts
│   ├── build_vivado.tcl   # Baseline build script
│   ├── build_hardwired.tcl # Hardwired build script
│   ├── gen_weights.py     # Weight generation
│   └── verify_output.py   # Output verification
├── sim/                   # Analytical simulation models
│   ├── simulation.py      # Gate count analysis
│   ├── energy.py          # Energy efficiency model
│   └── economic.py        # Security cost analysis
├── constraints/           # FPGA constraints
│   └── zybo_z7_20.xdc    # Pin assignments and timing
└── docs/                  # Documentation
    ├── instructions.md    # Build instructions
    └── evaluation_results.md # Detailed results
```

## Quick Start

### 1. Build Baseline Version

```bash
cd public_release
vivado -mode batch -source scripts/build_vivado.tcl
```

This builds the conventional BRAM-based implementation (fits on device).

### 2. Build Hardwired Version

```bash
vivado -mode batch -source scripts/build_hardwired.tcl
```

This builds the constant-coefficient version (exceeds device capacity, synthesis-only).

### 3. Run Analytical Simulations

```bash
# Gate count analysis
python3 sim/simulation.py

# Energy efficiency model
python3 sim/energy.py

# Economic security analysis
python3 sim/economic.py
```

## Key Results

### FPGA Validation
- **Baseline:** 11,309 LUTs (21% utilization) - Successfully implemented
- **Hardwired:** 170,502 LUTs (321% utilization) - Synthesis-only proof-of-concept
- **LUT Reduction:** 1.81× per MAC unit (measured)

### Analytical Projections (ASIC)
- **Gate Count Reduction:** 4.85× (theoretical) / 1.62× (system-level conservative)
- **Energy Efficiency:** 50× (device) / 10-15× (system-level including host CPU)
- **Die Area (28nm):** 520 mm² (TinyLlama-1.1B), 3,680 mm² (Llama-2-7B)
- **Manufacturing Cost:** $52-165 at 100K+ volume (higher at low volume due to NRE)

## Reproducibility

All results in the paper can be reproduced with:
1. A $200 FPGA development board
2. Free Vivado WebPACK software
3. Python with standard scientific libraries

**Total cost to reproduce: ~$200**

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ita2025,
  title={The Immutable Tensor Architecture: A Pure Dataflow Approach for Secure, Energy-Efficient AI Inference},
  author={[Author Name]},
  booktitle={[Conference Name]},
  year={2025}
}
```

## License

This work is released under the MIT License. See LICENSE file for details.

## Acknowledgments

This work was conducted without external funding at a teaching-focused institution. The FPGA prototype was implemented on a development board purchased by the author. We are grateful to the open-source community for tools that enable hardware architecture research without institutional resources.

## Contact

For questions or collaboration opportunities, please contact: [your email]

## Future Work

We welcome contributions in the following areas:
- ASIC tape-out on 28nm/40nm nodes
- On-device attention acceleration
- Hybrid architectures (partial hardwiring)
- Alternative quantization schemes
- Security countermeasures for side-channel attacks

---

**Note:** This is a research prototype demonstrating architectural concepts. The hardwired version exceeds the FPGA capacity by design, validating that custom ASICs are required for practical deployment.
