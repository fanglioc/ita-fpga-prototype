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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, '../weights')
    
    try:
        W1 = np.load(os.path.join(weights_dir, 'weights_expand.npy'))
        W2 = np.load(os.path.join(weights_dir, 'weights_contract.npy'))
    except FileNotFoundError:
        print(f"✗ Error: Weights not found in {weights_dir}")
        print("  Run 'python3 scripts/gen_weights.py' first.")
        return None, None
    
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
    ref_result = run_reference_model(input_vec)
    if ref_result[0] is None:
        return
        
    hidden_ref, output_ref = ref_result
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