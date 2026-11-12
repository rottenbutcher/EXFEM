"""Multi-GPU execution.

`shard_ops` distributes element-by-element operators (element tangents, contact elements, ...)
over all visible devices along the element axis (padded with zero matrices), so the K*v
products inside PCG / LOBPCG run SPMD on every GPU and XLA inserts the all-reduce of the
scatter-add automatically (NCCL on NVIDIA multi-GPU nodes).

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    gfem run big.inp --solver amg --multi-gpu

Testing on CPU:  XLA_FLAGS=--xla_force_host_platform_device_count=4
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def device_mesh(devices=None):
    devs = np.array(devices or jax.devices())
    return Mesh(devs, ("e",))


def _pad(a, n, value=0):
    pad = (-a.shape[0]) % n
    if not pad:
        return a
    width = [(0, pad)] + [(0, 0)] * (a.ndim - 1)
    return jnp.pad(a, width, constant_values=value)


def shard_ops(ops, mesh=None):
    mesh = mesh or device_mesh()
    n = mesh.devices.size
    if n <= 1:
        return ops
    out = []
    for d, K in ops:
        d = _pad(jnp.asarray(d), n, 0)
        K = _pad(jnp.asarray(K), n, 0)
        out.append((jax.device_put(d, NamedSharding(mesh, P("e", None))),
                    jax.device_put(K, NamedSharding(mesh, P("e", *([None] * (K.ndim - 1)))))))
    return out


def replicate(x, mesh=None):
    mesh = mesh or device_mesh()
    return jax.device_put(x, NamedSharding(mesh, P()))


def info():
    d = jax.devices()
    return f"{len(d)} device(s): " + ", ".join(f"{x.platform}:{x.id}" for x in d)
