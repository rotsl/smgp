// =============================================================================
// SMGP Userspace Hardware Library — Header
// =============================================================================
// Provides a C API for interacting with the SMGP FPGA/PCIe device.
// =============================================================================

#ifndef SMGP_HW_LIB_H
#define SMGP_HW_LIB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque device handle
typedef struct smgp_hw_ctx smgp_hw_ctx_t;

// Status codes
typedef enum {
    SMGP_OK = 0,
    SMGP_ERR_OPEN,
    SMGP_ERR_WRITE,
    SMGP_ERR_READ,
    SMGP_ERR_MMAP,
    SMGP_ERR_TIMEOUT,
    SMGP_ERR_IRQ,
    SMGP_ERR_INVALID_ARG
} smgp_status_t;

// Register addresses (matching smgp_system.sv)
#define SMGP_CTRL      0x0000
#define SMGP_STATUS    0x0004
#define SMGP_INSTR     0x0008
#define SMGP_RESULT    0x000C
#define SMGP_HD_DIM    0x0010
#define SMGP_MAX_NODES 0x0014
#define SMGP_FRAC_BITS 0x0018
#define SMGP_IRQ_EN    0x001C

// Open/close device
smgp_status_t smgp_hw_open(smgp_hw_ctx_t **ctx, const char *device_path);
smgp_status_t smgp_hw_close(smgp_hw_ctx_t *ctx);

// Register read/write
smgp_status_t smgp_hw_read_reg(smgp_hw_ctx_t *ctx, uint32_t offset, uint32_t *val);
smgp_status_t smgp_hw_write_reg(smgp_hw_ctx_t *ctx, uint32_t offset, uint32_t val);

// Instruction execution
smgp_status_t smgp_hw_execute(smgp_hw_ctx_t *ctx, uint32_t instruction, uint32_t timeout_ms);
smgp_status_t smgp_hw_get_status(smgp_hw_ctx_t *ctx, uint32_t *status);

// Memory access (via mmap)
smgp_status_t smgp_hw_mem_write(smgp_hw_ctx_t *ctx, uint64_t offset,
                                 const void *data, size_t len);
smgp_status_t smgp_hw_mem_read(smgp_hw_ctx_t *ctx, uint64_t offset,
                                void *data, size_t len);

// Utility: encode instruction (matches ISA package)
static inline uint32_t smgp_encode_instr(uint8_t opcode, uint8_t sub_opcode,
                                           uint8_t flags, uint16_t operand) {
    return ((uint32_t)opcode << 28) | ((uint32_t)sub_opcode << 24) |
           ((uint32_t)flags << 16) | operand;
}

#ifdef __cplusplus
}
#endif

#endif // SMGP_HW_LIB_H
