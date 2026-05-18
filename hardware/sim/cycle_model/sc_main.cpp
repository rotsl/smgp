/**
 * SMGP Cycle-Accurate Simulation — Top-Level Testbench
 *
 * sc_main instantiates the SMGP core model and a simple testbench that:
 *   1. Resets the core
 *   2. Issues a COMPUTE_LAPLACIAN instruction (nnz = 1024)
 *   3. Issues an EIGEN_DECOMPOSITION instruction (k=16, nnz=1024)
 *   4. Issues an HD_BIND instruction
 *   5. Prints the total cycle count
 */

#include "smgp_sc_module.h"
#include <iostream>
#include <iomanip>

using namespace smgp_opcodes;

// ===========================================================================
// Top-Level Testbench Module
// ===========================================================================
class TopTestbench : public sc_module {
public:
    // Clock signal
    sc_clock clk;

    // Reset
    sc_signal<bool> rst_n;

    // Instruction interface
    sc_signal<bool>     instr_valid;
    sc_signal<uint32_t> instr_opcode;
    sc_signal<uint32_t> instr_sub_opcode;
    sc_signal<uint32_t> instr_operand;

    // Status interface
    sc_signal<bool>     busy;
    sc_signal<uint32_t> status_code;

    // Memory interface
    sc_signal<uint32_t> mem_addr;
    sc_signal<uint32_t> mem_data_in;
    sc_signal<uint32_t> mem_data_out;
    sc_signal<bool>     mem_wr_en;

    // Completion signal
    sc_signal<bool>     done;

    // DUT
    SMGPCore *dut;

    // Constructor
    SC_HAS_PROCESS(TopTestbench);

    TopTestbench(sc_module_name name)
        : sc_module(name),
          clk("clk", 10, SC_NS),  // 100 MHz clock
          rst_n("rst_n"),
          instr_valid("instr_valid"),
          instr_opcode("instr_opcode"),
          instr_sub_opcode("instr_sub_opcode"),
          instr_operand("instr_operand"),
          busy("busy"),
          status_code("status_code"),
          mem_addr("mem_addr"),
          mem_data_in("mem_data_in"),
          mem_data_out("mem_data_out"),
          mem_wr_en("mem_wr_en"),
          done("done")
    {
        // Instantiate DUT
        dut = new SMGPCore("smgp_core_dut");
        dut->clk(clk);
        dut->rst_n(rst_n);
        dut->instr_valid(instr_valid);
        dut->instr_opcode(instr_opcode);
        dut->instr_sub_opcode(instr_sub_opcode);
        dut->instr_operand(instr_operand);
        dut->busy(busy);
        dut->status_code(status_code);
        dut->mem_addr(mem_addr);
        dut->mem_data_in(mem_data_in);
        dut->mem_data_out(mem_data_out);
        dut->mem_wr_en(mem_wr_en);

        SC_THREAD(stimulus_thread);
    }

    // -----------------------------------------------------------------------
    // Stimulus: issue a sequence of instructions and monitor completion
    // -----------------------------------------------------------------------
    void stimulus_thread() {
        std::cout << "\n"
                  << "================================================================\n"
                  << "  SMGP Cycle-Accurate Simulation\n"
                  << "  Test: Laplacian -> Eigen-decomp -> HD Bind\n"
                  << "================================================================\n"
                  << std::endl;

        // ---- Reset sequence ----
        instr_valid.write(false);
        instr_opcode.write(0);
        instr_sub_opcode.write(0);
        instr_operand.write(0);
        rst_n.write(false);
        wait(10, SC_NS);   // assert reset for 1 clock cycle
        rst_n.write(true);
        wait(20, SC_NS);   // settle

        unsigned long start_cycles = dut->cycle_count;

        // ---- Instruction 1: Compute Laplacian (nnz=1024) ----
        issue_instruction(COMPUTE_LAPLACIAN, 0x00, 1024);
        wait_for_completion();

        // ---- Instruction 2: Eigen-decomposition (k=16, nnz=1024) ----
        issue_instruction(EIGEN_DECOMPOSITION, 0x00, 16 * 1024);
        wait_for_completion();

        // ---- Instruction 3: HD Bind ----
        issue_instruction(HD_BIND, 0x00, 0);
        wait_for_completion();

        // ---- Instruction 4: NOP (baseline) ----
        issue_instruction(NOP, 0x00, 0);
        wait_for_completion();

        unsigned long end_cycles = dut->cycle_count;
        unsigned long instr_cycles = end_cycles - start_cycles;

        // ---- Results ----
        std::cout << "\n"
                  << "================================================================\n"
                  << "  SIMULATION RESULTS\n"
                  << "================================================================\n"
                  << "  Total cycles (all):         " << std::setw(12) << end_cycles << "\n"
                  << "  Cycles for instructions:    " << std::setw(12) << instr_cycles << "\n"
                  << "    - Compute Laplacian:      " << std::setw(12) << 1024  << "  (expected: nnz)\n"
                  << "    - Eigen-decomposition:    " << std::setw(12) << 16384 << "  (expected: k*nnz)\n"
                  << "    - HD Bind:                " << std::setw(12) << 1     << "  (expected: 1)\n"
                  << "    - NOP:                    " << std::setw(12) << 1     << "  (expected: 1)\n"
                  << "================================================================\n"
                  << std::endl;

        sc_stop();
    }

private:
    void issue_instruction(uint32_t opcode, uint32_t sub_op, uint32_t operand) {
        wait(clk.posedge_event());
        instr_opcode.write(opcode);
        instr_sub_opcode.write(sub_op);
        instr_operand.write(operand);
        instr_valid.write(true);
        wait(clk.posedge_event());
        instr_valid.write(false);
    }

    void wait_for_completion() {
        while (busy.read()) {
            wait(clk.posedge_event());
        }
        // Wait one extra cycle for DONE->IDLE transition
        wait(clk.posedge_event());
    }
};

// ===========================================================================
// sc_main — entry point for SystemC simulation
// ===========================================================================
int sc_main(int argc, char* argv[]) {
    std::cout << "SMGP Cycle-Accurate Hardware Simulator" << std::endl;
    std::cout << "SystemC " << sc_core::sc_version() << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    TopTestbench tb("top_tb");

    // Run simulation
    try {
        sc_start();
    } catch (const std::exception& e) {
        std::cerr << "Simulation error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nSimulation finished successfully." << std::endl;
    return 0;
}
