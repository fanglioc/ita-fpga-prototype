# Vivado Build Script for ITA Prototype (Hardwired Version)

# Create project
create_project ita_hardwired ./vivado/ita_hardwired -part xc7z020clg400-1 -force

# Add source files
# We add all SV files, but set the top module explicitly
add_files -fileset sources_1 [glob ./rtl/*.sv]

# Add constraints
add_files -fileset constrs_1 ./constraints/zybo_z7_20.xdc

# Set top module to the hardwired version
set_property top ita_top_hardwired [current_fileset]

# Synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Open synthesized design for analysis
open_run synth_1
report_utilization -file ./vivado/utilization_synth_hardwired.txt
report_timing_summary -file ./vivado/timing_synth_hardwired.txt

# Implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Open implemented design
open_run impl_1
report_utilization -file ./vivado/utilization_impl_hardwired.txt
report_timing_summary -file ./vivado/timing_impl_hardwired.txt
report_power -file ./vivado/power_hardwired.txt

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "========================================="
puts "HARDWIRED BUILD COMPLETE!"
puts "Bitstream: vivado/ita_hardwired.runs/impl_1/ita_top_hardwired.bit"
puts "Reports:   vivado/utilization_impl_hardwired.txt"
puts "           vivado/power_hardwired.txt"
puts "========================================="
