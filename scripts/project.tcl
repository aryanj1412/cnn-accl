# ============================================
# Vivado Project Reconstruction Script
# Project: cnn-1
# Vivado Version: 2023.2
# Description:
# Recreates the project using relative paths
# suitable for source control (GitHub).
# ============================================

# Set origin directory
set origin_dir "."

# Project name
set _xil_proj_name_ "cnn-1"

# Create project
create_project ${_xil_proj_name_} ./${_xil_proj_name_} -part xc7z020clg400-1

# Get project directory
set proj_dir [get_property directory [current_project]]

# ============================================
# Project properties
# ============================================

set obj [current_project]
set_property default_lib xil_defaultlib $obj
set_property enable_vhdl_2008 1 $obj
set_property feature_set FeatureSet_Classic $obj
set_property ip_cache_permissions {read write} $obj
set_property ip_output_repo "$proj_dir/${_xil_proj_name_}.cache/ip" $obj
set_property mem.enable_memory_map_generation 1 $obj
set_property part xc7z020clg400-1 $obj
set_property revised_directory_structure 1 $obj
set_property sim.central_dir "$proj_dir/${_xil_proj_name_}.ip_user_files" $obj
set_property sim.ip.auto_export_scripts 1 $obj
set_property xpm_libraries {XPM_CDC XPM_FIFO XPM_MEMORY} $obj

# ============================================
# Create sources_1 fileset
# ============================================

if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}

set obj [get_filesets sources_1]

# ============================================
# Add HDL sources (from ./src)
# ============================================

set files [glob -nocomplain "${origin_dir}/src/*.v"]

set imported_files ""
foreach f $files {
    lappend imported_files [import_files -fileset sources_1 $f]
}

# ============================================
# Recreate Block Design
# ============================================

if {[file exists "${origin_dir}/bd/block_design.tcl"]} {
    puts "INFO: Rebuilding block design"
    source "${origin_dir}/bd/block_design.tcl"
}

# Generate wrapper if BD exists
set bd_files [get_files -quiet *.bd]
if {[llength $bd_files] > 0} {
    set wrapper_path [make_wrapper -fileset sources_1 -files $bd_files -top]
    add_files -norecurse -fileset sources_1 $wrapper_path
}

# Set top module
set_property top system_top [get_filesets sources_1]

# ============================================
# Constraints (optional)
# ============================================

if {[string equal [get_filesets -quiet constrs_1] ""]} {
    create_fileset -constrset constrs_1
}

set constr_files [glob -nocomplain "${origin_dir}/src/*.xdc"]

if {[llength $constr_files] > 0} {
    add_files -fileset constrs_1 $constr_files
}

# ============================================
# Simulation fileset
# ============================================

if {[string equal [get_filesets -quiet sim_1] ""]} {
    create_fileset -simset sim_1
}

set_property top system_top [get_filesets sim_1]

# ============================================
# Create synthesis run
# ============================================

if {[string equal [get_runs -quiet synth_1] ""]} {
    create_run -name synth_1 -part xc7z020clg400-1 \
        -flow {Vivado Synthesis 2023} \
        -strategy "Vivado Synthesis Defaults" \
        -constrset constrs_1
}

current_run -synthesis [get_runs synth_1]

# ============================================
# Create implementation run
# ============================================

if {[string equal [get_runs -quiet impl_1] ""]} {
    create_run -name impl_1 -part xc7z020clg400-1 \
        -flow {Vivado Implementation 2023} \
        -strategy "Vivado Implementation Defaults" \
        -constrset constrs_1 \
        -parent_run synth_1
}

current_run -implementation [get_runs impl_1]

# ============================================
# Completion message
# ============================================

puts "INFO: Project created successfully: ${_xil_proj_name_}"
