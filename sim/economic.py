import matplotlib.pyplot as plt
import numpy as np

# --- THREAT MODEL PARAMETERS ---

# Scenario: Attacker wants to steal a 7B Parameter Model
# Costs are estimated based on hardware security industry standards (e.g., Chipworks/TechInsights pricing)

# 1. GPU Server / Consumer PC
# Attack: Software Memory Dump
cost_gpu_equipment = 2000    # Cost of the PC
cost_gpu_labor     = 100     # 2 hours of a hacker's time
total_gpu = cost_gpu_equipment + cost_gpu_labor

# 2. ITA (Neural Cartridge) - 28nm Node
# Attack: Delayering & SEM Imaging
# 28nm has ~8-10 metal layers. Imaging 400mm^2 at nm resolution is expensive.
cost_ita_lab       = 500000  # SEM/FIB Machine rental time
cost_ita_labor     = 1500000 # 6 months of specialized EE Reverse Engineering team
total_ita = cost_ita_lab + cost_ita_labor

# --- VISUALIZATION ---
labels = ['Standard GPU\n(Software Attack)', 'ITA Cartridge\n(Physical Attack)']
costs = [total_gpu, total_ita]

fig, ax = plt.subplots(figsize=(8, 6))

# We use Log Scale because the difference is massive (Orders of Magnitude)
bars = ax.bar(labels, costs, color=['#e74c3c', '#2c3e50'])

# Annotations
ax.set_yscale('log')
ax.set_ylabel('Cost to Steal Model (USD) - Log Scale')
ax.set_title('The Economic Barrier: Cost of Model Extraction')

# Add text on bars
plt.text(0, total_gpu * 1.5, f"${total_gpu:,}", ha='center', fontweight='bold')
plt.text(1, total_ita * 1.2, f"~${total_ita/1000000:.1f} Million", ha='center', fontweight='bold', color='black')

# The "Breakthrough" Annotation
plt.annotate('Economic Viability Threshold\n(Piracy is cheaper than Training)',
             xy=(0.5, 10000), xytext=(1.5, 1000),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('wp3_security_chart.png')