// =============================================================================
// SMGP PCIe Linux Kernel Driver
// =============================================================================
// Minimal Linux kernel driver for the SMGP Graph Processing Unit via PCIe.
// Supports:
//   - PCIe device enumeration and BAR mapping
//   - mmap() for userspace access to device registers
//   - read/write for configuration
//   - Interrupt handling (MSI)
//
// Build:
//   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
//
// References:
//   - Corbet, J., et al. (2005). "Linux Device Drivers." 3rd Ed., O'Reilly.
//   - PCIe SIG (2019). "PCI Express Base Specification 5.0."
// =============================================================================

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

#define SMGP_VENDOR_ID    0x1337  // Custom vendor ID
#define SMGP_DEVICE_ID    0xSMGP  // Custom device ID
#define SMGP_DRIVER_NAME  "smgp"
#define SMGP_CLASS_NAME   "smgp_class"

// Register offsets (matches AXI4-Lite register file in smgp_system.sv)
#define SMGP_REG_CTRL      0x0000
#define SMGP_REG_STATUS    0x0004
#define SMGP_REG_INSTR     0x0008
#define SMGP_REG_RESULT    0x000C
#define SMGP_REG_HD_DIM    0x0010
#define SMGP_REG_MAX_NODES 0x0014
#define SMGP_REG_FRAC      0x0018
#define SMGP_REG_IRQ_EN    0x001C
#define SMGP_REG_SPACE     0x0020  // Total register space

// Device state
struct smgp_device {
    struct pci_dev *pdev;
    void __iomem *bar0;           // BAR0: MMIO register space
    void __iomem *bar2;           // BAR2: Graph memory (large)
    struct cdev cdev;
    struct device *device;
    dev_t dev_num;
    int irq;
    wait_queue_head_t waitq;
    bool irq_received;
};

static struct smgp_device *smgp_dev;

// ---------------------------------------------------------------------------
// Register read/write helpers
// ---------------------------------------------------------------------------
static inline u32 smgp_read_reg(struct smgp_device *dev, u32 offset)
{
    return ioread32(dev->bar0 + offset);
}

static inline void smgp_write_reg(struct smgp_device *dev, u32 offset, u32 val)
{
    iowrite32(val, dev->bar0 + offset);
}

// ---------------------------------------------------------------------------
// File operations
// ---------------------------------------------------------------------------
static ssize_t smgp_read(struct file *filp, char __user *buf,
                          size_t count, loff_t *f_pos)
{
    struct smgp_device *dev = filp->private_data;
    u32 val;

    if (*f_pos >= SMGP_REG_SPACE)
        return 0;

    val = smgp_read_reg(dev, (u32)*f_pos);
    if (copy_to_user(buf, &val, sizeof(val)))
        return -EFAULT;

    *f_pos += sizeof(val);
    return sizeof(val);
}

static ssize_t smgp_write(struct file *filp, const char __user *buf,
                           size_t count, loff_t *f_pos)
{
    struct smgp_device *dev = filp->private_data;
    u32 val;

    if (*f_pos >= SMGP_REG_SPACE)
        return 0;
    if (count < sizeof(val))
        return -EINVAL;

    if (copy_from_user(&val, buf, sizeof(val)))
        return -EFAULT;

    smgp_write_reg(dev, (u32)*f_pos, val);
    *f_pos += sizeof(val);
    return sizeof(val);
}

static int smgp_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct smgp_device *dev = filp->private_data;
    unsigned long pfn;

    if (vma->vm_end - vma->vm_start > pci_resource_len(dev->pdev, 2))
        return -EINVAL;

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    pfn = (pci_resource_start(dev->pdev, 2) >> PAGE_SHIFT) + vma->vm_pgoff;

    return remap_pfn_range(vma, vma->vm_start, pfn,
                           vma->vm_end - vma->vm_start, vma->vm_page_prot);
}

static const struct file_operations smgp_fops = {
    .owner   = THIS_MODULE,
    .read    = smgp_read,
    .write   = smgp_write,
    .mmap    = smgp_mmap,
};

// ---------------------------------------------------------------------------
// Interrupt handler
// ---------------------------------------------------------------------------
static irqreturn_t smgp_irq_handler(int irq, void *dev_id)
{
    struct smgp_device *dev = dev_id;
    u32 status;

    status = smgp_read_reg(dev, SMGP_REG_STATUS);
    if (status & 0x2) {  // Done flag
        dev->irq_received = true;
        wake_up_interruptible(&dev->waitq);
    }

    return IRQ_HANDLED;
}

// ---------------------------------------------------------------------------
// PCI probe / remove
// ---------------------------------------------------------------------------
static int smgp_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    int rc;
    struct smgp_device *dev;

    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->pdev = pdev;
    pci_set_drvdata(pdev, dev);
    init_waitqueue_head(&dev->waitq);

    // Enable device
    rc = pci_enable_device(pdev);
    if (rc) goto err_free;

    // Request regions
    rc = pci_request_regions(pdev, SMGP_DRIVER_NAME);
    if (rc) goto err_disable;

    // Map BARs
    dev->bar0 = pci_iomap(pdev, 0, SMGP_REG_SPACE);
    if (!dev->bar0) { rc = -EIO; goto err_release; }

    dev->bar2 = pci_iomap(pdev, 2, 0);
    if (!dev->bar2) { rc = -EIO; goto err_unmap0; }

    // Request interrupt
    rc = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI);
    if (rc < 0) goto err_unmap2;

    dev->irq = pci_irq_vector(pdev, 0);
    rc = request_irq(dev->irq, smgp_irq_handler, IRQF_SHARED,
                     SMGP_DRIVER_NAME, dev);
    if (rc) goto err_free_irq;

    // Create char device
    rc = alloc_chrdev_region(&dev->dev_num, 0, 1, SMGP_DRIVER_NAME);
    if (rc) goto err_free_irq_handler;

    cdev_init(&dev->cdev, &smgp_fops);
    rc = cdev_add(&dev->cdev, dev->dev_num, 1);
    if (rc) goto err_unreg_chrdev;

    dev->device = device_create(SMGP_CLASS_NAME, &pdev->dev,
                               dev->dev_num, NULL, SMGP_DRIVER_NAME);
    if (IS_ERR(dev->device)) {
        rc = PTR_ERR(dev->device);
        goto err_del_cdev;
    }

    smgp_dev = dev;
    dev_info(&pdev->dev, "SMGP GPU2.0 initialized (IRQ %d)\n", dev->irq);
    return 0;

err_del_cdev:
    cdev_del(&dev->cdev);
err_unreg_chrdev:
    unregister_chrdev_region(dev->dev_num, 1);
err_free_irq_handler:
    free_irq(dev->irq, dev);
err_free_irq:
    pci_free_irq_vectors(pdev);
err_unmap2:
    pci_iounmap(pdev, dev->bar2);
err_unmap0:
    pci_iounmap(pdev, dev->bar0);
err_release:
    pci_release_regions(pdev);
err_disable:
    pci_disable_device(pdev);
err_free:
    kfree(dev);
    return rc;
}

static void smgp_remove(struct pci_dev *pdev)
{
    struct smgp_device *dev = pci_get_drvdata(pdev);
    if (!dev) return;

    device_destroy(SMGP_CLASS_NAME, dev->dev_num);
    cdev_del(&dev->cdev);
    unregister_chrdev_region(dev->dev_num, 1);
    free_irq(dev->irq, dev);
    pci_free_irq_vectors(pdev);
    pci_iounmap(pdev, dev->bar2);
    pci_iounmap(pdev, dev->bar0);
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    kfree(dev);
    dev_info(&pdev->dev, "SMGP GPU2.0 removed\n");
}

// ---------------------------------------------------------------------------
// PCI device ID table
// ---------------------------------------------------------------------------
static const struct pci_device_id smgp_id_table[] = {
    { PCI_DEVICE(SMGP_VENDOR_ID, SMGP_DEVICE_ID) },
    { 0, }
};
MODULE_DEVICE_TABLE(pci, smgp_id_table);

static struct pci_driver smgp_driver = {
    .name     = SMGP_DRIVER_NAME,
    .id_table = smgp_id_table,
    .probe    = smgp_probe,
    .remove   = smgp_remove,
};

// ---------------------------------------------------------------------------
// Module init/exit
// ---------------------------------------------------------------------------
static int __init smgp_init(void)
{
    int rc;
    struct class *cl;

    cl = class_create(THIS_MODULE, SMGP_CLASS_NAME);
    if (IS_ERR(cl)) return PTR_ERR(cl);

    rc = pci_register_driver(&smgp_driver);
    if (rc) {
        class_destroy(cl);
        return rc;
    }
    return 0;
}

static void __exit smgp_exit(void)
{
    pci_unregister_driver(&smgp_driver);
    class_destroy(SMGP_CLASS_NAME);
}

module_init(smgp_init);
module_exit(smgp_exit);

MODULE_AUTHOR("Rohan R <rohan@smgp.dev>");
MODULE_DESCRIPTION("SMGP Graph Processing Unit PCIe Driver");
MODULE_LICENSE("Apache-2.0");
MODULE_VERSION("0.1.0");
