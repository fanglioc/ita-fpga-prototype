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

# 4. SYSTEM OVERHEAD (The Reality Check)
# Added based on Reviewer Feedback
P_SERDES_ACTIVE = 500.0  # mW (PCIe/USB PHY)
P_HOST_CPU_ATTN = 10000.0 # mW (Host CPU doing Attention)
P_INTERPOSER    = 0.15   # 15% overhead for 2.5D

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

# B. ITA (Immutable Tensor) - Device Level
# 1. No Fetch (Weight is locally embedded)
# 2. Activation flows through wire (Wire Cost)
# 3. Compute (Hardwired Shift-Add)
cost_ita_fetch   = 0.0 # ZERO
cost_ita_wire    = 4 * E_WIRE_MM * DIST_MM # 4-bit activation flows 5mm
cost_ita_compute = E_ADD_INT4
total_ita_device = cost_ita_fetch + cost_ita_wire + cost_ita_compute

# C. ITA (Immutable Tensor) - System Level
# Amortizing system power over 20 tokens/s * 7B params
# Total Ops/sec = 20 * 7e9 = 140e9 ops/s
# System Power = 10.5 W (10W CPU + 0.5W SerDes)
# System Energy per Op = 10.5 J / 140e9 = 75 pJ
cost_ita_system = (P_SERDES_ACTIVE + P_HOST_CPU_ATTN) / 140000.0 # mW / (Mops/s) -> pJ
total_ita_system = total_ita_device + cost_ita_system

# --- VISUALIZATION ---
labels = ['NVIDIA A100', 'ITA (Device)', 'ITA (System)']
fetch_costs = [cost_a100_fetch, cost_ita_fetch, cost_ita_fetch]
internal_costs = [cost_a100_sram, cost_ita_wire, cost_ita_wire]
compute_costs = [cost_a100_compute, cost_ita_compute, cost_ita_compute]
system_costs = [0, 0, cost_ita_system]

width = 0.5
fig, ax = plt.subplots(figsize=(10, 6))

# Stacked Bar Chart
ax.bar(labels, fetch_costs, width, label='Memory Fetch (DRAM)', color='#e74c3c')
ax.bar(labels, internal_costs, width, bottom=fetch_costs, label='Internal Movement', color='#f39c12')
ax.bar(labels, compute_costs, width, bottom=np.array(fetch_costs)+np.array(internal_costs), label='Compute', color='#2ecc71')
ax.bar(labels, system_costs, width, bottom=np.array(fetch_costs)+np.array(internal_costs)+np.array(compute_costs), label='System Overhead (CPU/SerDes)', color='#9b59b6')

# Annotations
ax.set_ylabel('Energy per Parameter (picoJoules)')
ax.set_title('Energy Analysis: Device vs System Reality')
ax.legend()

# Text Labels
speedup_dev = total_a100 / total_ita_device
speedup_sys = total_a100 / total_ita_system
plt.text(1, total_ita_device + 5, f"Device: {speedup_dev:.1f}x", ha='center')
plt.text(2, total_ita_system + 5, f"System: {speedup_sys:.1f}x", ha='center')

plt.tight_layout()
plt.savefig('wp2_energy_chart.png')