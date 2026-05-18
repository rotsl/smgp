# ===========================================================================
# SMGP FPGA Constraints — Xilinx Versal VCK190 Evaluation Kit
# ===========================================================================
# Board: Xilinx VCK190 Evaluation Board
# FPGA:  xcvc1902-vsva2197-2MP-eSV-es1 (Versal Prime)
# ============================================================================

# ---------------------------------------------------------------------------
# System Clock — 300 MHz differential clock from on-board oscillator
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN G23  [get_ports sys_clk_p]
set_property PACKAGE_PIN G24  [get_ports sys_clk_n]
set_property IOSTANDARD LVDS  [get_ports sys_clk_p]
set_property IOSTANDARD LVDS  [get_ports sys_clk_n]
create_clock -add -name sys_clk_300mhz -period 3.33 -waveform {0 1.665} [get_ports sys_clk_p]

# ---------------------------------------------------------------------------
# PL Clocks — from the NOC (Network on Chip) / CPM (Control Processor Module)
# ---------------------------------------------------------------------------
# The Versal architecture provides PL clocks through the NOC and CPM
# subsystems. These are generated internally and assigned to clock buffers.

# NOC to PL clock — typically 200-500 MHz
# Created via the CIPS (Control, Interfaces & Processing System) IP
# create_generated_clock -name noc_to_pl_clk_0 -source [get_pins CIPS/PLCLK0] \
#     [get_pins noc_clk_buf/O]

# ---------------------------------------------------------------------------
# Reset — from system controller
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN L16  [get_ports sys_reset_n]
set_property IOSTANDARD LVCMOS18 [get_ports sys_reset_n]

# ---------------------------------------------------------------------------
# NoC (Network on Chip) Connections — for HBM access
# ---------------------------------------------------------------------------
# The Versal NOC provides high-bandwidth access to HBM stacks.
# Each NoC unit connects to one or more HBM pseudo-channels.

# NoC 0 — connects to HBM stack 0 (top)
# NoC 1 — connects to HBM stack 1 (bottom)
# NoC configuration is done through the CIPS IP and Xilinx Platform Management.
#
# The following documents the intended mapping:
#   NoC0 NMU -> HBM PC[0:3]   (Graph topology data, adjacency matrix)
#   NoC1 NMU -> HBM PC[4:7]   (Spectral data, eigenpairs)
#   NoC2 NMU -> HBM PC[8:11]  (HD manifold coordinates)
#   NoC3 NMU -> HBM PC[12:15] (Scratchpad / working memory)

# ---------------------------------------------------------------------------
# HBM — 4 HBM stacks (32 GB total)
# ---------------------------------------------------------------------------
# Versal Prime HBM is accessed exclusively through the NOC.
# No external pin-level constraints are needed — the NOC-to-HBM
# connections are inside the hardened silicon.

# HBM controller AXI clocks are derived from the NOC
create_clock -add -name hbm_user_clk_0 -period 2.50 [get_pins -hierarchical -filter {NAME =~ *hbm_ctrl_0/inst/axi_aclk*}]

# ---------------------------------------------------------------------------
# PCIe Gen4 x16 — from CPM (Integrated PCIe controller)
# ---------------------------------------------------------------------------
# PCIe reference clock (100 MHz) — from board oscillator
set_property PACKAGE_PIN C3  [get_ports cpm_pcie_refclk_p]
set_property PACKAGE_PIN C4  [get_ports cpm_pcie_refclk_n]
set_property IOSTANDARD LVDS [get_ports cpm_pcie_refclk_p]

# PCIe reset
set_property PACKAGE_PIN E4  [get_ports cpm_pcie_perstn]
set_property IOSTANDARD LVCMOS18 [get_ports cpm_pcie_perstn]

# PCIe lanes are routed through the CPM hard block — no user pin constraints

# ---------------------------------------------------------------------------
# DDR4 DIMM — optional external memory (not primary; HBM is primary)
# ---------------------------------------------------------------------------
# DDR4 PLL reference clock
set_property PACKAGE_PIN D2  [get_ports ddr4_act_n]
set_property IOSTANDARD POD12_DDR4 [get_ports ddr4_act_n]

# ---------------------------------------------------------------------------
# GPIO LEDs — for status indication
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN D20  [get_ports user_led[0]]
set_property PACKAGE_PIN E20  [get_ports user_led[1]]
set_property PACKAGE_PIN F20  [get_ports user_led[2]]
set_property PACKAGE_PIN G20  [get_ports user_led[3]]

foreach led_idx {0 1 2 3} {
    set_property IOSTANDARD LVCMOS18 [get_ports user_led[$led_idx]]
}

# ---------------------------------------------------------------------------
# UART — PL-side UART for debug (via USB-to-UART bridge)
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN AA19 [get_ports pl_uart_txd]
set_property PACKAGE_PIN AA20 [get_ports pl_uart_rxd]
set_property IOSTANDARD LVCMOS18 [get_ports pl_uart_txd]
set_property IOSTANDARD LVCMOS18 [get_ports pl_uart_rxd]

# ---------------------------------------------------------------------------
# Timing Constraints
# ---------------------------------------------------------------------------
# Asynchronous clock domains
set_clock_groups -asynchronous \
    -group [get_clocks -include_generated_clocks sys_clk_300mhz] \
    -group [get_clocks -include_generated_clocks -of_objects [get_nets -hierarchical -filter {NAME =~ *hbm*}]] \
    -group [get_clocks -include_generated_clocks cpm_pcie_refclk]

# Max delay for NOC cross-domain crossing
set_max_delay -datapath_only -from [get_clocks -include_generated_clocks sys_clk_300mhz] \
               -to [get_clocks -include_generated_clocks hbm_user_clk_0] 10.0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
set_property CONFIG_MODE SPIx4 [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]
