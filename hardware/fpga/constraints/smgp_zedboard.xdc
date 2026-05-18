# =============================================================================
# SMGP FPGA Constraints — Xilinx Zynq-7000 ZedBoard
# =============================================================================
# Maps I/O pins for the ZedBoard FPGA (XC7Z020-CLG484-1).
# Includes PCIe, clock, reset, and debug LED pin assignments.
# =============================================================================

# --- System Clock (125 MHz from PS7) ---
set_property -dict { PACKAGE_PIN H9    CLOCK_DEDICATED_ROUTE FALSE } [get_pins clk_sys]
create_clock -add -name sys_clk -period 8.000 -waveform {0 4.000} [get_pins clk_sys]

# --- Reset (active low, from PS7) ---
set_property PACKAGE_PIN G15 [get_ports rst_n_sys]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n_sys]

# --- Debug LEDs (LD0-LD3 on ZedBoard) ---
set_property PACKAGE_PIN M14 [get_ports {status_led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {status_led[0]}]
set_property PACKAGE_PIN M15 [get_ports {status_led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {status_led[1]}]
set_property PACKAGE_PIN G14 [get_ports {status_led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {status_led[2]}]
set_property PACKAGE_PIN D18 [get_ports {status_led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {status_led[3]}]

# --- UART (for debug output) ---
set_property PACKAGE_PIN Y18 [get_ports uart_txd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_txd]
set_property PACKAGE_PIN Y19 [get_ports uart_rxd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rxd]

# --- GPIO Buttons (for manual control) ---
set_property PACKAGE_PIN N15 [get_ports btn_c]
set_property IOSTANDARD LVCMOS33 [get_ports btn_c]
set_property PACKAGE_PIN D19 [get_ports btn_d]
set_property IOSTANDARD LVCMOS33 [get_ports btn_d]

# --- Timing Constraints ---
set_clock_groups -asynchronous \
    -group [get_clocks -include_generated_clocks sys_clk]

# --- False Paths (no timing on debug signals) ---
set_false_path -to [get_ports status_led*]
set_false_path -to [get_ports uart_txd]
set_false_path -from [get_ports btn_*]
set_false_path -from [get_ports uart_rxd]

# --- Placement Constraints ---
# Keep SMGP core in the programmable logic region
set_property LOC SLICE_X50Y50 [get_cells -hier -filter {NAME =~ "*u_core*"}]
set_property CONTAIN_ROUTING TRUE [get_cells -hier -filter {NAME =~ "*u_core*"}]
