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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, '../weights')
    
    CONFIG = {
        'expand': {
            'input_dim': 64,
            'output_dim': 128,
            'weight_file': os.path.join(weights_dir, 'weights_expand.hex'),
            'npy_file': os.path.join(weights_dir, 'weights_expand.npy')
        },
        'contract': {
            'input_dim': 128,
            'output_dim': 64,
            'weight_file': os.path.join(weights_dir, 'weights_contract.hex'),
            'npy_file': os.path.join(weights_dir, 'weights_contract.npy')
        }
    }
    
    # Create weights directory
    os.makedirs(weights_dir, exist_ok=True)
    
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