# ===========================================================================
# SMGP FPGA Constraints — Xilinx Artix-7 A7 (Arty A7-100)
# ===========================================================================
# Board: Digilent Arty A7-100
# FPGA:  xc7a100tcsg324-1
# ============================================================================

# ---------------------------------------------------------------------------
# Clock — 100 MHz system clock on E3 (Bank 35)
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

# ---------------------------------------------------------------------------
# Reset — active-low push button on C12 (Bank 33)
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN C12 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# ---------------------------------------------------------------------------
# LEDs — active-high LEDs for status output (Bank 33)
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN R12 [get_ports led[0]]
set_property PACKAGE_PIN T12 [get_ports led[1]]
set_property PACKAGE_PIN U12 [get_ports led[2]]
set_property PACKAGE_PIN U13 [get_ports led[3]]
set_property PACKAGE_PIN V13 [get_ports led[4]]
set_property PACKAGE_PIN V14 [get_ports led[5]]
set_property PACKAGE_PIN U14 [get_ports led[6]]
set_property PACKAGE_PIN U15 [get_ports led[7]]

foreach led_idx {0 1 2 3 4 5 6 7} {
    set_property IOSTANDARD LVCMOS33 [get_ports led[$led_idx]]
}

# LED[0] — Busy indicator
# LED[1] — DMA active
# LED[2] — HD engine active
# LED[3] — Spectral engine active
# LED[4:7] — Status code (binary)

# ---------------------------------------------------------------------------
# Buttons — user push buttons (Bank 33)
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN D19 [get_ports btn[0]]
set_property PACKAGE_PIN D20 [get_ports btn[1]]
set_property PACKAGE_PIN L20 [get_ports btn[2]]
set_property PACKAGE_PIN L19 [get_ports btn[3]]

foreach btn_idx {0 1 2 3} {
    set_property IOSTANDARD LVCMOS33 [get_ports btn[$btn_idx]]
}

# ---------------------------------------------------------------------------
# UART — USB-UART on JTAG USB connector (Bank 14)
# ---------------------------------------------------------------------------
set_property PACKAGE_PIN A18 [get_ports uart_txd]
set_property PACKAGE_PIN B18 [get_ports uart_rxd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_txd]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rxd]

# ---------------------------------------------------------------------------
# DDR3 Memory — if using the on-board DDR3 (MT41K128M16)
# ---------------------------------------------------------------------------
# System clock for DDR3 PLL reference
set_property PACKAGE_PIN E3  [get_ports ddr3_sys_clk_p]
set_property PACKAGE_PIN E2  [get_ports ddr3_sys_clk_n]
set_property IOSTANDARD DIFF_SSTL15 [get_ports ddr3_sys_clk_p]
set_property IOSTANDARD DIFF_SSTL15 [get_ports ddr3_sys_clk_n]

# ---------------------------------------------------------------------------
# PMOD Headers — for external memristor crossbar / sensor interface
# ---------------------------------------------------------------------------
# PMOD JA (Bank 15)
set_property PACKAGE_PIN G13 [get_ports pmod_ja[0]]
set_property PACKAGE_PIN B11 [get_ports pmod_ja[1]]
set_property PACKAGE_PIN A11 [get_ports pmod_ja[2]]
set_property PACKAGE_PIN D12 [get_ports pmod_ja[3]]
set_property PACKAGE_PIN D13 [get_ports pmod_ja[4]]
set_property PACKAGE_PIN B18 [get_ports pmod_ja[5]]
set_property PACKAGE_PIN A18 [get_ports pmod_ja[6]]
set_property PACKAGE_PIN K16 [get_ports pmod_ja[7]]

foreach pidx {0 1 2 3 4 5 6 7} {
    set_property IOSTANDARD LVCMOS33 [get_ports pmod_ja[$pidx]]
}

# ---------------------------------------------------------------------------
# Configuration — SPI flash for bitstream storage
# ---------------------------------------------------------------------------
set_property CONFIG_MODE SPIx4 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
