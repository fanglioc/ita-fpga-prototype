import numpy as np
import matplotlib.pyplot as plt

def estimate_gate_count_generic_multiplier(bit_width=8):
    """
    Estimates logic gates for a generic array multiplier.
    Rough approximation: ~ 6N^2 - 12N gates for N-bit multiplier.
    For 8-bit: 6(64) - 96 = ~288 gates per multiplication.
    """
    return 6 * (bit_width**2) - 12 * bit_width

def estimate_gate_count_hardwired(weight_value, bit_width=8):
    """
    Estimates logic gates for a HARDWIRED constant multiplication.
    Instead of a multiplier, we use Shift-and-Add.
    Gate count ≈ (Number of non-zero bits - 1) * (Adder Cost).

    If weight is 0: 0 gates (hardwired to ground).
    If weight is power of 2 (e.g., 00100000): 0 gates (just a wire shift).
    If weight is 3 (00000011): 1 adder (input + input<<1).
    """
    # Convert weight to binary string representation
    # We use absolute value for gate estimation (sign handled by wire routing)
    int_val = int(abs(weight_value))

    # Count set bits (Population Count)
    set_bits = bin(int_val).count('1')

    if set_bits <= 1:
        return 0 # Just wires!

    # For K set bits, we need K-1 adders.
    # An 8-bit ripple carry adder is roughly ~40 gates (5 gates per bit).
    adder_cost = 5 * bit_width
    return (set_bits - 1) * adder_cost

# --- SIMULATION ---
# 1. Generate a "Fake" Layer of Llama-2 (4096 neurons)
# We use a normal distribution of weights, then quantize to INT8
np.random.seed(42)
weights = np.random.normal(0, 25, 4096).astype(int)
weights = np.clip(weights, -127, 127) # Clip to INT8 range

# 2. Calculate Costs
generic_cost_total = 0
hardwired_cost_total = 0

for w in weights:
    generic_cost_total += estimate_gate_count_generic_multiplier(8)
    hardwired_cost_total += estimate_gate_count_hardwired(w, 8)

# 3. The Result
savings = generic_cost_total / hardwired_cost_total
conservative_factor = 3.0 # Reviewer suggested 2-3x overhead
conservative_savings = savings / conservative_factor

print(f"--- SIMULATION RESULTS (Llama-2 Layer Mockup) ---")
print(f"Total Generic Gates (GPU/TPU):   {generic_cost_total:,}")
print(f"Total Hardwired Gates (Your IP): {hardwired_cost_total:,}")
print(f"Optimistic Efficiency: {savings:.2f}x reduction")
print(f"Conservative Efficiency (3x overhead): {conservative_savings:.2f}x reduction")

# 4. Visualization for the Paper
labels = ['Generic GPU', 'ITA (Optimistic)', 'ITA (Conservative)']
values = [generic_cost_total, hardwired_cost_total, hardwired_cost_total * conservative_factor]

plt.bar(labels, values, color=['gray', 'firebrick', 'darkred'])
plt.ylabel('Logic Gate Count (Lower is Better)')
plt.title('Silicon Area: Theory vs. Reality')
plt.savefig('wp1_gate_count.png')
# plt.show()