#!/usr/bin/env python
import sys
import math
import numpy as np
from mpi4py import MPI

SEGMENT = 1 << 20


def simple_sieve(m):
    is_prime = np.ones(m + 1, dtype=bool)
    is_prime[:2] = False
    for q in range(2, math.isqrt(m) + 1):
        if is_prime[q]:
            is_prime[q * q::q] = False
    return np.nonzero(is_prime)[0]


def block_range(lo, hi, rank, size):
    length = hi - lo
    start = lo + rank * length // size
    end = lo + (rank + 1) * length // size
    return start, end


def cross_out(is_prime, offset, start, end, divisors):
    for seg_start in range(start, end, SEGMENT):
        seg_end = min(seg_start + SEGMENT, end)
        for q in divisors:
            if q * q >= seg_end:
                break
            first = max(q * q, (seg_start + q - 1) // q * q)
            is_prime[first - offset:seg_end - offset:q] = False


def sieve_segment(start, end, divisors):
    is_prime = np.ones(end - start, dtype=bool)
    cross_out(is_prime, start, start, end, divisors)
    return np.nonzero(is_prime)[0].astype(np.int64) + start


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10**7
    show = "--print" in sys.argv

    comm.Barrier()
    t0 = MPI.Wtime()

    root = math.isqrt(n)
    divisors = simple_sieve(root).tolist()
    start, end = block_range(root + 1, n + 1, rank, size)
    local_primes = sieve_segment(start, end, divisors)
    t_compute = MPI.Wtime() - t0

    counts = comm.gather(len(local_primes), root=0)
    if rank == 0:
        result = np.empty(sum(counts), dtype=np.int64)
        recvbuf = [result, counts, MPI.INT64_T]
    else:
        recvbuf = None
    comm.Gatherv(local_primes, recvbuf, root=0)

    t_total = MPI.Wtime() - t0
    max_compute = comm.reduce(t_compute, op=MPI.MAX, root=0)

    if rank == 0:
        primes = np.concatenate([np.array(divisors, dtype=np.int64), result])
        print(f"N={n} procs={size} primes={len(primes)} "
              f"compute={max_compute:.4f}s total={t_total:.4f}s")
        if show:
            print(primes)


if __name__ == "__main__":
    main()
