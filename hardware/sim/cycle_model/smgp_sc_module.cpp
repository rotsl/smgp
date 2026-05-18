/**
 * SMGP Cycle-Accurate SystemC Model Implementation
 *
 * Behavioral model that estimates cycle counts for core SMGP operations:
 *   - Compute Laplacian      : nnz cycles
 *   - Eigen-decomposition    : k * nnz cycles
 *   - HD Bind                : 1 cycle  (memristor crossbar is instantaneous)
 *   - Graph Rewrite          : base + nnz/10 cycles
 */

#include "smgp_sc_module.h"
#include <iostream>
#include <iomanip>

using namespace smgp_opcodes;

// ===========================================================================
// SMGPCore Implementation
// ===========================================================================

SMGPCore::SMGPCore(sc_module_name name)
    : sc_module(name),
      cycle_count(0),
      m_state(IDLE),
      m_current_opcode(0),
      m_cycles_remaining(0),
      m_operand(0),
      m_nnz(1024),            // default graph size
      m_k_eigen(16),          // default eigenpairs
      m_latency_laplacian_per_nnz(1),
      m_latency_eigen_per_k_nnz(1),
      m_latency_hd_bind(1),
      m_latency_graph_rewrite_base(10)
{
    SC_METHOD(cycle_accurate_exec);
    sensitive_pos << clk;

    std::cout << "[SMGP] Core model instantiated: \"" << name << "\"" << std::endl;
}

SMGPCore::~SMGPCore() {
    std::cout << "[SMGP] Core model destroyed. Total cycles observed: "
              << cycle_count << std::endl;
}

// ---------------------------------------------------------------------------
// Compute operation latency based on graph parameters
// ---------------------------------------------------------------------------
void SMGPCore::compute_latency(uint32_t opcode, uint32_t sub_opcode,
                               uint32_t operand) {
    m_operand = operand;

    switch (opcode) {
        case COMPUTE_LAPLACIAN:
            // Laplacian computation: O(nnz) per row -> nnz cycles total
            m_cycles_remaining = m_latency_laplacian_per_nnz * m_nnz;
            break;

        case EIGEN_DECOMPOSITION:
            // Power iteration / QR for k eigenpairs: roughly k*nnz cycles
            m_cycles_remaining = m_latency_eigen_per_k_nnz * m_k_eigen * m_nnz;
            break;

        case HD_BIND:
            // Memristor crossbar binding: essentially 1 cycle (analog compute)
            m_cycles_remaining = m_latency_hd_bind;
            break;

        case GRAPH_REWRITE:
            // Graph rewriting: base overhead + linear in nnz
            m_cycles_remaining = m_latency_graph_rewrite_base + (m_nnz / 10);
            break;

        case DMA_READ:
        case DMA_WRITE:
            // DMA: proportional to data size (operand = number of words)
            m_cycles_remaining = operand + 2;  // setup + 1 cycle/word
            break;

        case NOP:
            m_cycles_remaining = 1;
            break;

        default:
            std::cerr << "[SMGP] WARNING: Unknown opcode 0x"
                      << std::hex << opcode << std::dec << std::endl;
            m_cycles_remaining = 1;
            break;
    }
}

// ---------------------------------------------------------------------------
// Drive output ports
// ---------------------------------------------------------------------------
void SMGPCore::drive_outputs() {
    busy.write(m_state == EXECUTING);
    status_code.write(static_cast<uint32_t>(m_state));
}

// ---------------------------------------------------------------------------
// Main cycle-accurate execution method (called every rising clock edge)
// ---------------------------------------------------------------------------
void SMGPCore::cycle_accurate_exec() {
    cycle_count++;

    // Handle reset
    if (rst_n.read() == false) {
        m_state = IDLE;
        m_cycles_remaining = 0;
        m_current_opcode = 0;
        drive_outputs();
        return;
    }

    // State machine
    switch (m_state) {
        case IDLE:
            if (instr_valid.read()) {
                m_current_opcode = instr_opcode.read();
                uint32_t sub_op = instr_sub_opcode.read();
                uint32_t operand = instr_operand.read();

                compute_latency(m_current_opcode, sub_op, operand);
                m_state = EXECUTING;

                std::cout << "[SMGP] Cycle " << std::setw(8) << cycle_count
                          << " | Dispatched opcode 0x" << std::hex
                          << std::setw(2) << std::setfill('0')
                          << m_current_opcode << std::dec
                          << std::setfill(' ')
                          << " (" << m_cycles_remaining << " cycles estimated)"
                          << std::endl;
            }
            break;

        case EXECUTING:
            if (m_cycles_remaining > 0) {
                m_cycles_remaining--;
            }

            if (m_cycles_remaining == 0) {
                m_state = DONE;

                std::cout << "[SMGP] Cycle " << std::setw(8) << cycle_count
                          << " | Completed opcode 0x" << std::hex
                          << std::setw(2) << std::setfill('0')
                          << m_current_opcode << std::dec
                          << std::setfill(' ') << std::endl;
            }
            break;

        case DONE:
            // Auto-return to idle on next cycle
            m_state = IDLE;
            m_current_opcode = 0;
            break;
    }

    drive_outputs();
}

// ===========================================================================
// SMGPTestbench Implementation
// ===========================================================================

SMGPTestbench::SMGPTestbench(sc_module_name name)
    : sc_module(name)
{
    SC_THREAD(stimulus);
    sensitive << done.pos();
}

void SMGPTestbench::stimulus() {
    // This thread is triggered when 'done' goes high.
    // The actual stimulus logic is in sc_main via the top-level testbench.
    std::cout << "[TB] Simulation complete signal received." << std::endl;
}
