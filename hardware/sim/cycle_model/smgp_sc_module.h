/**
 * SMGP Cycle-Accurate SystemC Model Header
 *
 * Behavioral cycle-accurate model of the SMGP (Spectral Manifold Graph Processor)
 * core for early performance estimation and functional verification.
 */

#ifndef SMGP_SC_MODULE_H
#define SMGP_SC_MODULE_H

#include <systemc.h>

// ---------------------------------------------------------------------------
// Opcodes matching smgp_isa_pkg.sv
// ---------------------------------------------------------------------------
namespace smgp_opcodes {
    constexpr uint32_t COMPUTE_LAPLACIAN    = 0x01;
    constexpr uint32_t EIGEN_DECOMPOSITION  = 0x02;
    constexpr uint32_t HD_BIND              = 0x03;
    constexpr uint32_t GRAPH_REWRITE        = 0x04;
    constexpr uint32_t DMA_READ             = 0x80;
    constexpr uint32_t DMA_WRITE            = 0x81;
    constexpr uint32_t NOP                  = 0x00;
} // namespace smgp_opcodes

// ---------------------------------------------------------------------------
// SC_MODULE: SMGPCore
// ---------------------------------------------------------------------------
SC_MODULE(SMGPCore) {
    // -- Clock & reset -------------------------------------------------------
    sc_in<bool>        clk;
    sc_in<bool>        rst_n;

    // -- Instruction interface ------------------------------------------------
    sc_in<bool>        instr_valid;
    sc_in<uint32_t>    instr_opcode;
    sc_in<uint32_t>    instr_sub_opcode;
    sc_in<uint32_t>    instr_operand;

    // -- Status interface -----------------------------------------------------
    sc_out<bool>       busy;
    sc_out<uint32_t>   status_code;

    // -- Memory interface (simplified) ----------------------------------------
    sc_inout<uint32_t> mem_addr;
    sc_inout<uint32_t> mem_data_in;
    sc_out<uint32_t>   mem_data_out;
    sc_out<bool>       mem_wr_en;

    // -- Cycle counter -------------------------------------------------------
    unsigned long cycle_count;

    // -- Public API ----------------------------------------------------------
    void cycle_accurate_exec();

    // -- Constructor / Destructor --------------------------------------------
    SC_CTOR(SMGPCore);
    ~SMGPCore();

private:
    // Internal state
    enum State {
        IDLE,
        EXECUTING,
        DONE
    };

    State      m_state;
    uint32_t   m_current_opcode;
    uint32_t   m_cycles_remaining;
    uint32_t   m_operand;

    // Model parameters (configurable)
    uint32_t   m_nnz;           // non-zero entries in the graph
    uint32_t   m_k_eigen;       // number of eigenpairs requested
    uint32_t   m_latency_laplacian_per_nnz;
    uint32_t   m_latency_eigen_per_k_nnz;
    uint32_t   m_latency_hd_bind;
    uint32_t   m_latency_graph_rewrite_base;

    // Internal helper methods
    void compute_latency(uint32_t opcode, uint32_t sub_opcode, uint32_t operand);
    void drive_outputs();
};

// ---------------------------------------------------------------------------
// SC_MODULE: SMGPTestbench
// ---------------------------------------------------------------------------
SC_MODULE(SMGPTestbench) {
    sc_in<bool> done;

    void stimulus();

    SC_CTOR(SMGPTestbench);
};

#endif // SMGP_SC_MODULE_H
