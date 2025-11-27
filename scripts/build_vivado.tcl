# Vivado Build Script for ITA Prototype

# Create project
create_project ita_prototype vivado/ita_prototype -part xc7z020clg400-1 -force

# Add source files
add_files -fileset sources_1 [glob rtl/*.sv]

# Add constraints
add_files -fileset constrs_1 constraints/zybo_z7_20.xdc

# Set top module
set_property top ita_top [current_fileset]

# Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Open synthesized design for analysis
open_run synth_1
report_utilization -file vivado/utilization_synth.txt
report_timing_summary -file vivado/timing_synth.txt

# Implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Open implemented design
open_run impl_1
report_utilization -file vivado/utilization_impl.txt
report_timing_summary -file vivado/timing_impl.txt
report_power -file vivado/power.txt

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "========================================="
puts "BUILD COMPLETE!"
puts "Bitstream: vivado/ita_prototype.runs/impl_1/ita_top.bit"
puts "========================================="