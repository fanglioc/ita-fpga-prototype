import matplotlib.pyplot as plt
import numpy as np

# --- PHYSICS CONSTANTS (The "Ground Truth") ---
# Source: standard ISSCC / IEEE Solid-State papers
# Units: Picojoules (pJ)

# 1. MEMORY ACCESS (The Killer)
E_HBM_FETCH = 20.0   # pJ per bit (External DRAM fetch)
E_SRAM_READ = 5.0    # pJ per bit (On-chip cache read)

# 2. COMPUTE (The Math)
# FP16 MAC (7nm) vs INT4 Add (28nm)
E_MAC_FP16  = 1.1    # pJ per op (Generic Multiplier)
E_ADD_INT4  = 0.05   # pJ per op (Simple Adder)

# 3. WIRE TRAVERSAL (The ITA Cost)
# ITA moves data ~5mm across the chip per layer.
# 28nm Wire Capacitance: ~0.2 pJ per mm per bit
E_WIRE_MM   = 0.2
DIST_MM     = 5.0    # Avg distance per layer

# --- SIMULATION ---
# Scenario: 1 Parameter (Weight) interacting with 1 Activation
# We compare the energy cost to process ONE parameter.

# A. NVIDIA A100 (Von Neumann)
# 1. Fetch weight from HBM (Off-chip)
# 2. Store in SRAM (L2 Cache)
# 3. Move to Register File (L1)
# 4. Compute (FP16 MAC)
cost_a100_fetch   = 16 * E_HBM_FETCH  # 16 bits * 20 pJ
cost_a100_sram    = 16 * E_SRAM_READ  # 16 bits * 5 pJ
cost_a100_compute = E_MAC_FP16
total_a100 = cost_a100_fetch + cost_a100_sram + cost_a100_compute

# B. ITA (Immutable Tensor)
# 1. No Fetch (Weight is locally embedded)
# 2. Activation flows through wire (Wire Cost)
# 3. Compute (Hardwired Shift-Add)
cost_ita_fetch   = 0.0 # ZERO
cost_ita_wire    = 4 * E_WIRE_MM * DIST_MM # 4-bit activation flows 5mm
cost_ita_compute = E_ADD_INT4
total_ita = cost_ita_fetch + cost_ita_wire + cost_ita_compute

# --- VISUALIZATION ---
labels = ['NVIDIA A100 (GPU)', 'ITA (Your Chip)']
fetch_costs = [cost_a100_fetch, cost_ita_fetch]
internal_costs = [cost_a100_sram, cost_ita_wire]
compute_costs = [cost_a100_compute, cost_ita_compute]

width = 0.5
fig, ax = plt.subplots(figsize=(8, 6))

# Stacked Bar Chart
ax.bar(labels, fetch_costs, width, label='Memory Fetch (DRAM)', color='#e74c3c')
ax.bar(labels, internal_costs, width, bottom=fetch_costs, label='Internal Movement (SRAM/Wire)', color='#f39c12')
ax.bar(labels, compute_costs, width, bottom=np.array(fetch_costs)+np.array(internal_costs), label='Compute (MAC/Add)', color='#2ecc71')

# Annotations
ax.set_ylabel('Energy per Parameter (picoJoules)')
ax.set_title('The Energy Cliff: Von Neumann vs. Static Flow')
ax.legend()

# Text Labels
speedup = total_a100 / total_ita
plt.text(1, total_ita + 20, f"{speedup:.1f}x More Efficient", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('wp2_energy_chart.png')