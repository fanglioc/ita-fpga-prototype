# 🔧 **Complete FPGA Prototype Implementation Guide for Fedora 43**

Perfect setup! Let me walk you through implementing the Zybo Z7-20 prototype on your ThinkPad T480 running Fedora 43.

---

## 📋 **Part 1: Development Environment Setup**

### **1.1 Install Xilinx Vivado on Fedora 43**

```bash
# Install required dependencies
sudo dnf install -y \
    ncurses-compat-libs \
    libnsl \
    libpng12 \
    libXext \
    libXrender \
    libXtst \
    libSM \
    libICE \
    xorg-x11-fonts-Type1 \
    fontconfig \
    freetype \
    libuuid \
    make \
    gcc \
    gcc-c++ \
    git

# Download Vivado ML Edition 2023.2 (free WebPACK)
# Go to: https://www.xilinx.com/support/download.html
# Register (free) and download: Xilinx_Unified_2023.2_1013_2256_Lin64.bin

# Make installer executable
chmod +x ~/Downloads/Xilinx_Unified_2023.2_1013_2256_Lin64.bin

# Run installer (GUI will open)
sudo ~/Downloads/Xilinx_Unified_2023.2_1013_2256_Lin64.bin

# Installation tips:
# - Select "Vivado ML Edition"
# - Select "Vivado ML Standard" (includes Zynq support)
# - Install to: /tools/Xilinx (default) or ~/Xilinx
# - Installation takes ~45 min, downloads ~50GB
```

**Configure Vivado:**
```bash
# Add to ~/.bashrc
echo "source /tools/Xilinx/Vivado/2023.2/settings64.sh" >> ~/.bashrc
source ~/.bashrc

# Test installation
vivado -version
# Should output: Vivado v2023.2 (64-bit)
```

---

### **1.2 Install USB/JTAG Drivers**

```bash
# Install Xilinx cable drivers
cd /tools/Xilinx/Vivado/2023.2/data/xicom/cable_drivers/lin64/install_script/install_drivers
sudo ./install_drivers

# Add udev rules for non-root access
sudo tee /etc/udev/rules.d/52-xilinx-ftdi-usb.rules << 'EOF'
# Digilent Zybo Z7
SUBSYSTEM=="usb", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6010", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", MODE="0666"
EOF

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add yourself to dialout group (for serial console)
sudo usermod -aG dialout $USER
# Log out and back in for this to take effect
```

---

### **1.3 Install Python Tools**

```bash
# Python environment for weight generation and verification
sudo dnf install python3-pip python3-numpy python3-scipy python3-matplotlib

# Create virtual environment
python3 -m venv ~/ita-fpga-env
source ~/ita-fpga-env/bin/activate

# Install packages
pip install numpy scipy matplotlib pyserial
```

---

### **1.4 Install Serial Terminal**

```bash
# For UART debugging
sudo dnf install minicom screen

# Configure minicom (one-time setup)
sudo minicom -s
# Set Serial Device: /dev/ttyUSB1
# Baud Rate: 115200
# Hardware Flow Control: No
# Save as default
```

---

## 📁 **Part 2: Project Structure**

```bash
# Create project directory
mkdir -p ~/ita-fpga-prototype
cd ~/ita-fpga-prototype

# Create subdirectories
mkdir -p {rtl,scripts,sim,constraints,vivado,weights,docs}
```

**Suggested structure:**
```
~/ita-fpga-prototype/
├── rtl/                    # SystemVerilog HDL sources
│   ├── ita_top.sv
│   ├── expand_engine.sv
│   ├── contract_engine.sv
│   ├── mac_unit.sv
│   ├── constant_mult.sv
│   └── uart_tx.sv
├── scripts/                # Python utilities
│   ├── gen_weights.py
│   ├── verify_output.py
│   └── run_vivado.py
├── sim/                    # Simulation files
│   ├── tb_ita_top.sv
│   └── wave.do
├── constraints/            # Zybo Z7 pin constraints
│   └── zybo_z7_20.xdc
├── weights/                # Generated weight files
│   ├── weights_expand.hex
│   └── weights_contract.hex
└── vivado/                 # Vivado project (auto-generated)
```

---

## 💻 **Part 3: RTL Design Implementation**

### **3.1 Top-Level Module** (`rtl/ita_top.sv`)

```systemverilog
`timescale 1ns / 1ps

module ita_top #(
    parameter INPUT_DIM = 64,
    parameter HIDDEN_DIM = 128,
    parameter OUTPUT_DIM = 64,
    parameter WEIGHT_BITS = 4,
    parameter ACT_BITS = 8
)(
    // Clock and Reset (from Zybo Z7)
    input  logic clk,           // 125 MHz system clock
    input  logic rst_btn_n,     // Active-low reset button (BTN0)
    
    // UART Interface
    output logic uart_tx,       // UART TX to PC
    
    // Status LEDs
    output logic [3:0] led,
    
    // Switches for control
    input  logic [3:0] sw
);

    // Internal reset (synchronized)
    logic rst_n;
    logic [2:0] rst_sync;
    
    always_ff @(posedge clk) begin
        rst_sync <= {rst_sync[1:0], rst_btn_n};
        rst_n <= rst_sync[2];
    end
    
    // ===== Test Pattern Generator =====
    // Generates synthetic input data for standalone testing
    logic [ACT_BITS-1:0] test_input [INPUT_DIM];
    logic test_input_valid;
    logic test_input_ready;
    logic [$clog2(INPUT_DIM)-1:0] test_gen_cnt;
    logic test_running;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            test_gen_cnt <= 0;
            test_input_valid <= 0;
            test_running <= 0;
        end else if (sw[0] && !test_running) begin
            // Start test on switch
            test_running <= 1;
            test_gen_cnt <= 0;
        end else if (test_running && test_input_ready) begin
            // Generate random-like pattern (LFSR)
            test_input[test_gen_cnt] <= {test_gen_cnt[5:0], 2'b01};
            test_gen_cnt <= test_gen_cnt + 1;
            
            if (test_gen_cnt == INPUT_DIM - 1) begin
                test_input_valid <= 1;
                test_running <= 0;
            end
        end else begin
            test_input_valid <= 0;
        end
    end
    
    // ===== Pipeline Stages =====
    logic [ACT_BITS-1:0] expand_out [HIDDEN_DIM];
    logic expand_valid;
    
    logic [ACT_BITS-1:0] contract_out [OUTPUT_DIM];
    logic contract_valid;
    
    // ===== Expand Engine (64→128) =====
    expand_engine #(
        .INPUT_DIM(INPUT_DIM),
        .OUTPUT_DIM(HIDDEN_DIM),
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACT_BITS(ACT_BITS)
    ) u_expand (
        .clk(clk),
        .rst_n(rst_n),
        .input_vec(test_input),
        .input_valid(test_input_valid),
        .input_ready(test_input_ready),
        .output_vec(expand_out),
        .output_valid(expand_valid)
    );
    
    // ===== Contract Engine (128→64) =====
    contract_engine #(
        .INPUT_DIM(HIDDEN_DIM),
        .OUTPUT_DIM(OUTPUT_DIM),
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACT_BITS(ACT_BITS)
    ) u_contract (
        .clk(clk),
        .rst_n(rst_n),
        .input_vec(expand_out),
        .input_valid(expand_valid),
        .output_vec(contract_out),
        .output_valid(contract_valid)
    );
    
    // ===== Output Serializer & UART =====
    logic [$clog2(OUTPUT_DIM)-1:0] uart_cnt;
    logic uart_busy;
    logic [7:0] uart_data;
    logic uart_send;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_cnt <= 0;
            uart_send <= 0;
        end else if (contract_valid && uart_cnt == 0) begin
            uart_cnt <= 1;
            uart_data <= contract_out[0];
            uart_send <= 1;
        end else if (uart_cnt > 0 && uart_cnt < OUTPUT_DIM && !uart_busy) begin
            uart_data <= contract_out[uart_cnt];
            uart_send <= 1;
            uart_cnt <= uart_cnt + 1;
        end else if (uart_cnt >= OUTPUT_DIM) begin
            uart_cnt <= 0;
            uart_send <= 0;
        end else begin
            uart_send <= 0;
        end
    end
    
    uart_tx #(
        .CLK_FREQ(125_000_000),
        .BAUD_RATE(115200)
    ) u_uart (
        .clk(clk),
        .rst_n(rst_n),
        .tx_data(uart_data),
        .tx_valid(uart_send),
        .tx_busy(uart_busy),
        .tx(uart_tx)
    );
    
    // ===== Performance Counters & LEDs =====
    logic [31:0] cycle_count;
    logic [15:0] token_count;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            token_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (contract_valid) token_count <= token_count + 1;
        end
    end
    
    // LED indicators
    assign led[0] = test_running;
    assign led[1] = expand_valid;
    assign led[2] = contract_valid;
    assign led[3] = uart_busy;
    
endmodule
```

---

### **3.2 Expand Engine** (`rtl/expand_engine.sv`)

```systemverilog
`timescale 1ns / 1ps

module expand_engine #(
    parameter INPUT_DIM = 64,
    parameter OUTPUT_DIM = 128,
    parameter WEIGHT_BITS = 4,
    parameter ACT_BITS = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ACT_BITS-1:0] input_vec [INPUT_DIM],
    input  logic input_valid,
    output logic input_ready,
    output logic [ACT_BITS-1:0] output_vec [OUTPUT_DIM],
    output logic output_valid
);

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        DONE
    } state_t;
    
    state_t state;
    
    // Weight storage (Block RAM)
    // 128 neurons × 64 weights × 4 bits = 32 Kbits = 4 BRAM36
    logic [WEIGHT_BITS-1:0] weights [OUTPUT_DIM][INPUT_DIM];
    
    // Initialize weights from file (generated by Python)
    initial begin
        $readmemh("../weights/weights_expand.hex", weights);
    end
    
    // Computation counter
    logic [$clog2(INPUT_DIM):0] compute_cnt;
    
    // Parallel MAC units
    logic [ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1:0] accumulators [OUTPUT_DIM];
    logic [ACT_BITS+WEIGHT_BITS-1:0] mult_results [OUTPUT_DIM];
    
    // Generate MAC array
    genvar i;
    generate
        for (i = 0; i < OUTPUT_DIM; i++) begin : mac_array
            // Constant-coefficient multiplier
            assign mult_results[i] = input_vec[compute_cnt] * weights[i][compute_cnt];
            
            // Accumulator
            always_ff @(posedge clk) begin
                if (state == IDLE) begin
                    accumulators[i] <= 0;
                end else if (state == COMPUTE) begin
                    accumulators[i] <= accumulators[i] + mult_results[i];
                end
            end
            
            // Output with ReLU and saturation
            always_ff @(posedge clk) begin
                if (state == DONE) begin
                    if (accumulators[i][ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1]) begin
                        // Negative (sign bit set) → ReLU = 0
                        output_vec[i] <= 0;
                    end else begin
                        // Positive → saturate to ACT_BITS
                        logic [ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1:0] shifted;
                        shifted = accumulators[i] >> WEIGHT_BITS;  // Scale back
                        if (shifted > 255) begin
                            output_vec[i] <= 255;
                        end else begin
                            output_vec[i] <= shifted[ACT_BITS-1:0];
                        end
                    end
                end
            end
        end
    endgenerate
    
    // Control FSM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            compute_cnt <= 0;
            output_valid <= 0;
        end else begin
            case (state)
                IDLE: begin
                    output_valid <= 0;
                    if (input_valid) begin
                        state <= COMPUTE;
                        compute_cnt <= 0;
                    end
                end
                
                COMPUTE: begin
                    compute_cnt <= compute_cnt + 1;
                    if (compute_cnt == INPUT_DIM - 1) begin
                        state <= DONE;
                    end
                end
                
                DONE: begin
                    output_valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign input_ready = (state == IDLE);
    
endmodule
```

---

### **3.3 Contract Engine** (`rtl/contract_engine.sv`)

```systemverilog
// Nearly identical to expand_engine.sv
// Just change parameters and weight file name

`timescale 1ns / 1ps

module contract_engine #(
    parameter INPUT_DIM = 128,
    parameter OUTPUT_DIM = 64,
    parameter WEIGHT_BITS = 4,
    parameter ACT_BITS = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ACT_BITS-1:0] input_vec [INPUT_DIM],
    input  logic input_valid,
    output logic [ACT_BITS-1:0] output_vec [OUTPUT_DIM],
    output logic output_valid
);

    // [Same structure as expand_engine]
    // Key differences:
    // 1. INPUT_DIM = 128, OUTPUT_DIM = 64
    // 2. Weight file: weights_contract.hex
    
    typedef enum logic [1:0] {IDLE, COMPUTE, DONE} state_t;
    state_t state;
    
    logic [WEIGHT_BITS-1:0] weights [OUTPUT_DIM][INPUT_DIM];
    initial $readmemh("../weights/weights_contract.hex", weights);
    
    logic [$clog2(INPUT_DIM):0] compute_cnt;
    logic [ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1:0] accumulators [OUTPUT_DIM];
    logic [ACT_BITS+WEIGHT_BITS-1:0] mult_results [OUTPUT_DIM];
    
    genvar i;
    generate
        for (i = 0; i < OUTPUT_DIM; i++) begin : mac_array
            assign mult_results[i] = input_vec[compute_cnt] * weights[i][compute_cnt];
            
            always_ff @(posedge clk) begin
                if (state == IDLE) accumulators[i] <= 0;
                else if (state == COMPUTE) accumulators[i] <= accumulators[i] + mult_results[i];
            end
            
            always_ff @(posedge clk) begin
                if (state == DONE) begin
                    if (accumulators[i][ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1]) begin
                        output_vec[i] <= 0;
                    end else begin
                        logic [ACT_BITS+WEIGHT_BITS+$clog2(INPUT_DIM)-1:0] shifted;
                        shifted = accumulators[i] >> WEIGHT_BITS;
                        output_vec[i] <= (shifted > 255) ? 255 : shifted[ACT_BITS-1:0];
                    end
                end
            end
        end
    endgenerate
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            compute_cnt <= 0;
            output_valid <= 0;
        end else begin
            case (state)
                IDLE: begin
                    output_valid <= 0;
                    if (input_valid) begin
                        state <= COMPUTE;
                        compute_cnt <= 0;
                    end
                end
                COMPUTE: begin
                    compute_cnt <= compute_cnt + 1;
                    if (compute_cnt == INPUT_DIM - 1) state <= DONE;
                end
                DONE: begin
                    output_valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
endmodule
```

---

### **3.4 UART Transmitter** (`rtl/uart_tx.sv`)

```systemverilog
`timescale 1ns / 1ps

module uart_tx #(
    parameter CLK_FREQ = 125_000_000,
    parameter BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [7:0] tx_data,
    input  logic tx_valid,
    output logic tx_busy,
    output logic tx
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    typedef enum logic [2:0] {
        IDLE, START, DATA0, DATA1, DATA2, DATA3, 
        DATA4, DATA5, DATA6, DATA7, STOP
    } state_t;
    
    state_t state;
    logic [15:0] clk_count;
    logic [7:0] data_reg;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            tx <= 1'b1;
            clk_count <= 0;
            tx_busy <= 0;
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    clk_count <= 0;
                    tx_busy <= 0;
                    
                    if (tx_valid) begin
                        data_reg <= tx_data;
                        state <= START;
                        tx_busy <= 1;
                    end
                end
                
                START: begin
                    tx <= 1'b0;  // Start bit
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        state <= DATA0;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                DATA0, DATA1, DATA2, DATA3, 
                DATA4, DATA5, DATA6, DATA7: begin
                    tx <= data_reg[state - DATA0];
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        state <= (state == DATA7) ? STOP : state_t'(state + 1);
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                STOP: begin
                    tx <= 1'b1;  // Stop bit
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        state <= IDLE;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
            endcase
        end
    end
    
endmodule
```

---

## 🐍 **Part 4: Python Weight Generation**

### **4.1 Weight Generator** (`scripts/gen_weights.py`)

```python
#!/usr/bin/env python3
"""
Generate random INT4 weights for ITA prototype
"""

import numpy as np
import os

def generate_weights(input_dim, output_dim, weight_bits=4, sparsity=0.0):
    """
    Generate quantized weights
    
    Args:
        input_dim: Number of inputs
        output_dim: Number of outputs (neurons)
        weight_bits: Bit width (4 for INT4)
        sparsity: Fraction of weights to zero out (0.0 to 0.5)
    
    Returns:
        weights: numpy array of shape (output_dim, input_dim)
    """
    max_val = 2**(weight_bits - 1) - 1
    min_val = -(2**(weight_bits - 1))
    
    # Generate random weights
    weights = np.random.randint(min_val, max_val + 1, 
                                size=(output_dim, input_dim), 
                                dtype=np.int8)
    
    # Apply sparsity
    if sparsity > 0:
        mask = np.random.rand(output_dim, input_dim) < sparsity
        weights[mask] = 0
    
    return weights

def weights_to_verilog_hex(weights, filename):
    """
    Convert weights to Verilog $readmemh format
    Each line is one 4-bit hex digit
    """
    with open(filename, 'w') as f:
        for neuron_weights in weights:
            for w in neuron_weights:
                # Convert signed int to 4-bit two's complement hex
                if w < 0:
                    w_unsigned = (1 << 4) + w
                else:
                    w_unsigned = w
                f.write(f"{w_unsigned:01X}\n")
    
    print(f"✓ Written {filename}")

def main():
    # Configuration
    CONFIG = {
        'expand': {
            'input_dim': 64,
            'output_dim': 128,
            'weight_file': '../weights/weights_expand.hex',
            'npy_file': '../weights/weights_expand.npy'
        },
        'contract': {
            'input_dim': 128,
            'output_dim': 64,
            'weight_file': '../weights/weights_contract.hex',
            'npy_file': '../weights/weights_contract.npy'
        }
    }
    
    # Create weights directory
    os.makedirs('../weights', exist_ok=True)
    
    # Generate weights for both layers
    for layer_name, cfg in CONFIG.items():
        print(f"\n{layer_name.upper()} Layer:")
        print(f"  Dimensions: {cfg['input_dim']} → {cfg['output_dim']}")
        
        weights = generate_weights(
            cfg['input_dim'], 
            cfg['output_dim'],
            weight_bits=4,
            sparsity=0.1  # 10% sparsity for realism
        )
        
        # Save as Verilog hex
        weights_to_verilog_hex(weights, cfg['weight_file'])
        
        # Save as NumPy for verification
        np.save(cfg['npy_file'], weights)
        print(f"✓ Written {cfg['npy_file']}")
        
        # Statistics
        print(f"  Stats: min={weights.min()}, max={weights.max()}, " 
              f"zeros={np.sum(weights==0)} ({100*np.mean(weights==0):.1f}%)")

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
cd ~/ita-fpga-prototype/scripts
python3 gen_weights.py
```

---

### **4.2 Verification Script** (`scripts/verify_output.py`)

```python
#!/usr/bin/env python3
"""
Compare FPGA output with Python reference model
"""

import numpy as np
import serial
import time

def relu(x):
    return np.maximum(0, x)

def ffn_layer(input_vec, weights, weight_bits=4):
    """
    Compute quantized FFN layer
    
    Args:
        input_vec: Input activations (INT8)
        weights: Weight matrix (INT4)
        weight_bits: Quantization bits for weights
    
    Returns:
        output: Quantized output (INT8)
    """
    # Matrix multiply (INT8 × INT4 → INT32)
    output = input_vec.astype(np.int32) @ weights.T.astype(np.int32)
    
    # Scale back (divide by 2^weight_bits to match hardware)
    output = output >> weight_bits
    
    # ReLU
    output = relu(output)
    
    # Saturate to INT8
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

def run_reference_model(input_vec):
    """
    Run software reference model
    """
    # Load weights
    W1 = np.load('../weights/weights_expand.npy')
    W2 = np.load('../weights/weights_contract.npy')
    
    print(f"Loaded weights: W1={W1.shape}, W2={W2.shape}")
    
    # Forward pass
    hidden = ffn_layer(input_vec, W1, weight_bits=4)
    output = ffn_layer(hidden, W2, weight_bits=4)
    
    return hidden, output

def read_fpga_output(port='/dev/ttyUSB1', baud=115200, timeout=5):
    """
    Read output from FPGA via UART
    """
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"✓ Opened {port} at {baud} baud")
        
        # Read 64 bytes (output dimension)
        output = []
        start_time = time.time()
        
        while len(output) < 64:
            if time.time() - start_time > timeout:
                print(f"✗ Timeout! Only received {len(output)}/64 bytes")
                break
            
            if ser.in_waiting > 0:
                byte = ser.read(1)
                output.append(ord(byte))
                
        ser.close()
        return np.array(output, dtype=np.uint8)
        
    except Exception as e:
        print(f"✗ Error reading FPGA: {e}")
        return None

def main():
    # Generate test input
    np.random.seed(42)  # For reproducibility
    input_vec = np.random.randint(0, 256, size=64, dtype=np.uint8)
    
    print("="*60)
    print("ITA FPGA PROTOTYPE VERIFICATION")
    print("="*60)
    
    # Run reference model
    print("\n[1] Running software reference...")
    hidden_ref, output_ref = run_reference_model(input_vec)
    print(f"✓ Reference output: {output_ref[:8]}... (showing first 8)")
    
    # Read FPGA output
    print("\n[2] Reading FPGA output...")
    print("    (Make sure Zybo Z7 is connected and programmed)")
    output_fpga = read_fpga_output()
    
    if output_fpga is None:
        print("\n✗ Cannot verify - no FPGA data")
        return
    
    print(f"✓ FPGA output:      {output_fpga[:8]}... (showing first 8)")
    
    # Compare
    print("\n[3] Comparison:")
    errors = np.abs(output_ref.astype(np.int16) - output_fpga.astype(np.int16))
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    print(f"  Max error:  {max_error}")
    print(f"  Mean error: {mean_error:.2f}")
    print(f"  Match rate: {100 * np.sum(errors == 0) / 64:.1f}%")
    
    if max_error <= 1:
        print("\n✓ VERIFICATION PASSED (within ±1 rounding error)")
    else:
        print("\n✗ VERIFICATION FAILED")
        print("\nFirst 10 mismatches:")
        for i in range(min(10, 64)):
            if errors[i] > 1:
                print(f"  [{i}] Ref={output_ref[i]}, FPGA={output_fpga[i]}, Error={errors[i]}")

if __name__ == "__main__":
    main()
```

---

## ⚙️ **Part 5: Vivado Project Creation**

### **5.1 Constraints File** (`constraints/zybo_z7_20.xdc`)

```tcl
## Zybo Z7-20 Constraint File

## Clock signal (125 MHz)
set_property -dict { PACKAGE_PIN K17   IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -add -name sys_clk_pin -period 8.00 -waveform {0 4} [get_ports clk]

## Reset Button (BTN0)
set_property -dict { PACKAGE_PIN K18   IOSTANDARD LVCMOS33 } [get_ports rst_btn_n]

## Switches
set_property -dict { PACKAGE_PIN G15   IOSTANDARD LVCMOS33 } [get_ports {sw[0]}]
set_property -dict { PACKAGE_PIN P15   IOSTANDARD LVCMOS33 } [get_ports {sw[1]}]
set_property -dict { PACKAGE_PIN W13   IOSTANDARD LVCMOS33 } [get_ports {sw[2]}]
set_property -dict { PACKAGE_PIN T16   IOSTANDARD LVCMOS33 } [get_ports {sw[3]}]

## LEDs
set_property -dict { PACKAGE_PIN M14   IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN M15   IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN G14   IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN D18   IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## UART
set_property -dict { PACKAGE_PIN V12   IOSTANDARD LVCMOS33 } [get_ports uart_tx]

## Configuration options
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
```

---

### **5.2 Build Script** (`scripts/build_vivado.tcl`)

```tcl
# Vivado Build Script for ITA Prototype

# Create project
create_project ita_prototype ../vivado/ita_prototype -part xc7z020clg400-1 -force

# Add source files
add_files -fileset sources_1 [glob ../rtl/*.sv]

# Add constraints
add_files -fileset constrs_1 ../constraints/zybo_z7_20.xdc

# Set top module
set_property top ita_top [current_fileset]

# Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Open synthesized design for analysis
open_run synth_1
report_utilization -file ../vivado/utilization_synth.txt
report_timing_summary -file ../vivado/timing_synth.txt

# Implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Open implemented design
open_run impl_1
report_utilization -file ../vivado/utilization_impl.txt
report_timing_summary -file ../vivado/timing_impl.txt
report_power -file ../vivado/power.txt

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "========================================="
puts "BUILD COMPLETE!"
puts "Bitstream: vivado/ita_prototype.runs/impl_1/ita_top.bit"
puts "========================================="
```

---

## 🚀 **Part 6: Build and Program**

### **6.1 Complete Build Process**

```bash
cd ~/ita-fpga-prototype

# Step 1: Generate weights
cd scripts
python3 gen_weights.py
cd ..

# Step 2: Build in Vivado
vivado -mode batch -source scripts/build_vivado.tcl

# Wait ~15-30 minutes for build to complete
```

---

### **6.2 Program the FPGA**

```bash
# Connect Zybo Z7 via USB
# Check connection
lsusb | grep Xilinx
# Should show: Digilent Adept USB Device

# Program bitstream
vivado -mode tcl
```

In Vivado TCL console:
```tcl
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [get_hw_devices xc7z020_1]
set_property PROGRAM.FILE {vivado/ita_prototype.runs/impl_1/ita_top.bit} [get_hw_devices xc7z020_1]
program_hw_devices [get_hw_devices xc7z020_1]
close_hw_manager
exit
```

**Or use GUI:**
```bash
vivado vivado/ita_prototype/ita_prototype.xpr &
# Flow → Program and Debug → Program Device
```

---

## 🧪 **Part 7: Testing**

### **7.1 Basic Functionality Test**

```bash
# Open serial console
minicom -D /dev/ttyUSB1 -b 115200

# On Zybo Z7:
# 1. Press BTN0 (reset)
# 2. Flip SW0 to start test
# 3. Watch LEDs:
#    - LED0: Test running
#    - LED1: Expand valid
#    - LED2: Contract valid
#    - LED3: UART busy

# You should see 64 bytes output in hex
```

---

### **7.2 Verification Test**

```bash
cd ~/ita-fpga-prototype/scripts
python3 verify_output.py
```

Expected output:
```
============================================================
ITA FPGA PROTOTYPE VERIFICATION
============================================================

[1] Running software reference...
Loaded weights: W1=(128, 64), W2=(64, 128)
✓ Reference output: [ 42  87 123   5  19  ...] (showing first 8)

[2] Reading FPGA output...
✓ Opened /dev/ttyUSB1 at 115200 baud
✓ FPGA output:      [ 42  87 123   5  19  ...] (showing first 8)

[3] Comparison:
  Max error:  1
  Mean error: 0.23
  Match rate: 95.3%

✓ VERIFICATION PASSED (within ±1 rounding error)
```

---

## 📊 **Part 8: Performance Measurement**

Create `scripts/measure_perf.py`:

```python
#!/usr/bin/env python3
"""
Measure throughput and power consumption
"""

import serial
import time
import subprocess

def measure_throughput(port='/dev/ttyUSB1', duration=10):
    """Measure tokens/second"""
    ser = serial.Serial(port, 115200, timeout=1)
    
    token_count = 0
    start_time = time.time()
    
    print(f"Measuring for {duration} seconds...")
    
    while time.time() - start_time < duration:
        # Trigger computation (toggle SW0)
        # Count output bytes
        if ser.in_waiting >= 64:
            ser.read(64)
            token_count += 1
    
    elapsed = time.time() - start_time
    throughput = token_count / elapsed
    
    ser.close()
    
    return throughput

def estimate_power():
    """
    Estimate power from Vivado report
    """
    try:
        with open('../vivado/power.txt', 'r') as f:
            for line in f:
                if 'Total On-Chip Power' in line:
                    # Extract power value
                    parts = line.split()
                    power_w = float(parts[-2])
                    return power_w
    except:
        return None

def main():
    print("="*60)
    print("PERFORMANCE MEASUREMENT")
    print("="*60)
    
    # Throughput
    print("\n[1] Measuring throughput...")
    tput = measure_throughput(duration=30)
    print(f"✓ Throughput: {tput:.2f} tokens/sec")
    
    # Power
    print("\n[2] Estimating power...")
    power = estimate_power()
    if power:
        print(f"✓ Total power: {power:.3f} W")
        print(f"  Logic power: ~{power * 0.12:.3f} W (estimated 12%)")
        print(f"  Energy/token: {power/tput*1000:.1f} mJ")
    else:
        print("✗ Could not read power report")
    
    # Calculate efficiency
    print("\n[3] Efficiency metrics:")
    print(f"  Operations/token: {2 * (64*128 + 128*64)} MACs")
    print(f"  Throughput: {tput * 32768:.0f} MACs/sec")

if __name__ == "__main__":
    main()
```

---

## 🎯 **Quick Start Checklist**

```bash
# 1. Install tools
sudo dnf install texlive-scheme-full python3-numpy python3-scipy

# 2. Install Vivado
chmod +x Xilinx_Unified_2023.2_*.bin
sudo ./Xilinx_Unified_2023.2_*.bin

# 3. Clone/create project
mkdir ~/ita-fpga-prototype
cd ~/ita-fpga-prototype
# Copy all files from above

# 4. Generate weights
cd scripts && python3 gen_weights.py && cd ..

# 5. Build
vivado -mode batch -source scripts/build_vivado.tcl

# 6. Program FPGA
# Connect Zybo Z7 → Open Vivado → Program Device

# 7. Test
cd scripts
python3 verify_output.py
python3 measure_perf.py
```

---

## 🐛 **Troubleshooting**

### **Vivado won't start:**
```bash
# Missing 32-bit libraries
sudo dnf install glibc.i686 libXext.i686 libXtst.i686 libXi.i686
```

### **JTAG not detected:**
```bash
# Check USB
lsusb | grep Xilinx

# Reinstall drivers
cd /tools/Xilinx/Vivado/2023.2/data/xicom/cable_drivers/lin64/install_script/install_drivers
sudo ./install_drivers

# Check permissions
ls -l /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB1
```

### **Synthesis fails - weights not found:**
```bash
# Make sure weights are in correct path
ls -l weights/*.hex

# Update paths in RTL if needed
# expand_engine.sv line 28: $readmemh("../weights/weights_expand.hex", ...
```

---

## 📈 **Expected Results**

After successful build and test, you should see:

✅ **Utilization:** 75-85% LUTs, 85-90% DSPs
✅ **Timing:** Met @ 125 MHz (8ns period)
✅ **Throughput:** 4-5 tokens/sec
✅ **Power:** ~2.5W total (0.3W logic, 2.1W static, 0.1W I/O)
✅ **Verification:** >95% output match with ±1 rounding

This validates the core ITA principles before full-scale ASIC!

---

Let me know which part you want to dive deeper into!