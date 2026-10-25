#!/usr/bin/env python
import sys
import math
import numpy as np
from mpi4py import MPI

from sieve_mpi import block_range, cross_out


def main():
    world = MPI.COMM_WORLD
    node = world.Split_type(MPI.COMM_TYPE_SHARED)
    if node.Get_size() != world.Get_size():
        if world.Get_rank() == 0:
            print("Wersja z pamięcią współdzieloną wymaga uruchomienia na jednym węźle (-N 1)")
        world.Abort(1)

    rank = node.Get_rank()
    size = node.Get_size()

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10**7
    show = "--print" in sys.argv
    root = math.isqrt(n)

    itemsize = np.dtype(np.uint8).itemsize
    nbytes = (n + 1) * itemsize if rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node)
    buf, _ = win.Shared_query(0)
    is_prime = np.ndarray(buffer=buf, dtype=np.uint8, shape=(n + 1,))

    node.Barrier()
    t0 = MPI.Wtime()

    start, end = block_range(root + 1, n + 1, rank, size)
    is_prime[start:end] = 1
    win.Fence()

    if rank == 0:
        is_prime[:root + 1] = 1
        is_prime[:2] = 0
        for q in range(2, math.isqrt(root) + 1):
            if is_prime[q]:
                is_prime[q * q:root + 1:q] = 0
    win.Fence()

    divisors = np.nonzero(is_prime[:root + 1])[0].tolist()
    cross_out(is_prime, 0, start, end, divisors)
    win.Fence()
    t_sieve = MPI.Wtime() - t0

    lo = 0 if rank == 0 else start
    local = np.nonzero(is_prime[lo:end])[0]
    count = len(local)
    offset = node.exscan(count) or 0
    total = node.allreduce(count)

    itemsize = np.dtype(np.int64).itemsize
    win_p = MPI.Win.Allocate_shared(total * itemsize if rank == 0 else 0, itemsize, comm=node)
    buf, _ = win_p.Shared_query(0)
    primes = np.ndarray(buffer=buf, dtype=np.int64, shape=(total,))
    win_p.Fence()
    primes[offset:offset + count] = local + lo
    win_p.Fence()
    t_total = MPI.Wtime() - t0

    if rank == 0:
        print(f"N={n} procs={size} primes={total} "
              f"sieve={t_sieve:.4f}s total={t_total:.4f}s")
        if show:
            print(primes)

    win_p.Free()
    win.Free()


if __name__ == "__main__":
    main()
