// =============================================================================
// SMGP Userspace Hardware Library — Implementation
// =============================================================================

#include "smgp_hw_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <poll.h>

struct smgp_hw_ctx {
    int fd;
    void *reg_map;
    void *mem_map;
    size_t mem_size;
};

smgp_status_t smgp_hw_open(smgp_hw_ctx_t **ctx, const char *device_path) {
    smgp_hw_ctx_t *c;

    if (!ctx || !device_path)
        return SMGP_ERR_INVALID_ARG;

    c = calloc(1, sizeof(*c));
    if (!c) return SMGP_ERR_OPEN;

    c->fd = open(device_path, O_RDWR);
    if (c->fd < 0) {
        free(c);
        return SMGP_ERR_OPEN;
    }

    // Map register space
    c->reg_map = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, 0);
    if (c->reg_map == MAP_FAILED) {
        close(c->fd);
        free(c);
        return SMGP_ERR_MMAP;
    }

    *ctx = c;
    return SMGP_OK;
}

smgp_status_t smgp_hw_close(smgp_hw_ctx_t *ctx) {
    if (!ctx) return SMGP_ERR_INVALID_ARG;
    if (ctx->reg_map && ctx->reg_map != MAP_FAILED)
        munmap(ctx->reg_map, 0x1000);
    if (ctx->mem_map && ctx->mem_map != MAP_FAILED)
        munmap(ctx->mem_map, ctx->mem_size);
    if (ctx->fd >= 0) close(ctx->fd);
    free(ctx);
    return SMGP_OK;
}

smgp_status_t smgp_hw_read_reg(smgp_hw_ctx_t *ctx, uint32_t offset, uint32_t *val) {
    if (!ctx || !val || offset >= 0x1000) return SMGP_ERR_INVALID_ARG;
    *val = *(volatile uint32_t *)((uint8_t *)ctx->reg_map + offset);
    return SMGP_OK;
}

smgp_status_t smgp_hw_write_reg(smgp_hw_ctx_t *ctx, uint32_t offset, uint32_t val) {
    if (!ctx || offset >= 0x1000) return SMGP_ERR_INVALID_ARG;
    *(volatile uint32_t *)((uint8_t *)ctx->reg_map + offset) = val;
    return SMGP_OK;
}

smgp_status_t smgp_hw_execute(smgp_hw_ctx_t *ctx, uint32_t instruction,
                                uint32_t timeout_ms) {
    smgp_status_t rc;
    uint32_t status;
    struct pollfd pfd;

    if (!ctx) return SMGP_ERR_INVALID_ARG;

    // Write instruction
    rc = smgp_hw_write_reg(ctx, SMGP_INSTR, instruction);
    if (rc != SMGP_OK) return rc;

    // Poll for completion
    pfd.fd = ctx->fd;
    pfd.events = POLLIN;

    int ret = poll(&pfd, timeout_ms);
    if (ret < 0) return SMGP_ERR_IRQ;
    if (ret == 0) return SMGP_ERR_TIMEOUT;

    rc = smgp_hw_get_status(ctx, &status);
    return rc;
}

smgp_status_t smgp_hw_get_status(smgp_hw_ctx_t *ctx, uint32_t *status) {
    return smgp_hw_read_reg(ctx, SMGP_STATUS, status);
}

smgp_status_t smgp_hw_mem_write(smgp_hw_ctx_t *ctx, uint64_t offset,
                                 const void *data, size_t len) {
    if (!ctx || !data || len == 0) return SMGP_ERR_INVALID_ARG;
    memcpy((uint8_t *)ctx->reg_map + 0x1000 + offset, data, len);
    return SMGP_OK;
}

smgp_status_t smgp_hw_mem_read(smgp_hw_ctx_t *ctx, uint64_t offset,
                                void *data, size_t len) {
    if (!ctx || !data || len == 0) return SMGP_ERR_INVALID_ARG;
    memcpy(data, (uint8_t *)ctx->reg_map + 0x1000 + offset, len);
    return SMGP_OK;
}
