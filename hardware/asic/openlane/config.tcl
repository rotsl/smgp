# ===========================================================================
# OpenLane Configuration for SMGPU ASIC Synthesis
# ===========================================================================
# Design:    SMGP Core (Spectral Manifold Graph Processor)
# Foundry:   SkyWater 130nm (sky130hd)
# Flow:      OpenLane 2.0+
# ===========================================================================

# -- PDK Selection ----------------------------------------------------------
set ::env(PDK) "sky130A"
set ::env(STD_CELL_LIBRARY) "sky130_fd_sc_hd"

# -- Design Metadata --------------------------------------------------------
set ::env(DESIGN_NAME) "smgp_core"
set ::env(DESIGN_DIR) [file normalize $::env(DESIGN_DIR)]
set ::env(VERILOG_FILES) "\
    [glob $::env(DESIGN_DIR)/../../rtl/**/*.sv]\
    [glob $::env(DESIGN_DIR)/../../rtl/**/*.v]\
"
set ::env(DESIGN_IS_CORE) 1

# -- Clock Configuration ----------------------------------------------------
set ::env(CLOCK_PERIOD) "4.0"
set ::env(CLOCK_PORT) "clk"
set ::env(CLOCK_NET) $::env(CLOCK_PORT)
set ::env(SYNTH_CLOCK_UNCERTAINTY) "0.25"
set ::env(SYNTH_CLOCK_TRANSITION) "0.15"
set ::env(SYNTH_MAX_FANOUT) 20
set ::env(SYNTH_BUFFERING) 1

# -- Synthesis (Yosys/ABC) --------------------------------------------------
set ::env(SYNTH_STRATEGY) "AREA 0"
set ::env(SYNTH_MAX_TRAN) "0.75"
set ::env(SYNTH_SIZING) 0

# -- Floorplanning ----------------------------------------------------------
set ::env(FP_PDN_AUTO_ADJUST) 1
set ::env(FP_SIZING) "absolute"
# Die area: 2.0mm x 2.0mm — conservative for initial synthesis
set ::env(DIE_AREA) "0 0 2000 2000"
# Core area with margin
set ::env(CORE_AREA) "20 20 1980 1980"
# Pin placement
set ::env(FP_PIN_ORDER_CFG) $::env(DESIGN_DIR)/pin_order.cfg
set ::env(FP_CONTEXT_DEF) ""
set ::env(FP_CONTEXT_LEF) ""

# -- Power Delivery Network -------------------------------------------------
set ::env(VDD_NETS) "VDD"
set ::env(GND_NETS) "VSS"
set ::env(FP_PDN_VPITCH) 153.6
set ::env(FP_PDN_HPITCH) 153.6
set ::env(FP_PDN_VWIDTH) 3.0
set ::env(FP_PDN_HWIDTH) 3.0
set ::env(FP_PDN_VSPACING) 7.8
set ::env(FP_PDN_HSPACING) 7.8

# -- Macro Configuration (HBM controller, crossbar) -------------------------
set ::env(MACRO_PLACEMENT_CFG) $::env(DESIGN_DIR)/macro.cfg
set ::env(EXTRA_LEFS) ""
set ::env(EXTRA_GDS_FILES) ""

# -- Placement ---------------------------------------------------------------
set ::env(PL_TARGET_DENSITY) 0.45
set ::env(PL_MAX_DISPLACEMENT_X) 500
set ::env(PL_MAX_DISPLACEMENT_Y) 500
set ::env(PL_CELL_PADDING) 2
set ::env(GPL_CELL_PADDING) 2
set ::env(DIODE_INSERTION_STRATEGY) 3

# -- CTS (Clock Tree Synthesis) ---------------------------------------------
set ::env(CTS_TARGET_SKEW) 200
set ::env(CTS_ROOT_BUFFER) "sky130_fd_sc_hd__clkbuf_16"
set ::env(CTS_TOLERANCE) 100
set ::env(CTS_SINK_CLUSTERING_SIZE) 25

# -- Routing ----------------------------------------------------------------
set ::env(ROUTING_STRATEGY) 0
set ::env(GLB_RESIZER_TIMING_OPTIMIZATIONS) 1
set ::env(GLB_RESIZER_DESIGN_OPTIMIZATIONS) 1
set ::env(USE_ARC_ANTENNA_CHECK) 1

# -- DRC / LVS -------------------------------------------------------------
set ::env(MAGIC_DRC_USE_GDS) 1
set ::env(MAGIC_EXT_USE_GDS) 1
set ::env(QUIT_ON_LVS_ERROR) 0
set ::env(QUIT_ON_MAGIC_DRC) 0

# -- Fill Insertion ---------------------------------------------------------
set ::env(FP_PDN_CHECK_NODES) 1

# -- OpenRAM (if on-chip SRAM is needed) ------------------------------------
# set ::env(SRAM_TAG) "sram"
# set ::env(FP_SRAM_CARVEOUT) 1

# -- Reporting --------------------------------------------------------------
set ::env(RUN_STANDALONE) 0
set ::env(PRIMARY_SIGNOFF_TOOL) "magic"
