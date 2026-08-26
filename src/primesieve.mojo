"""Odd-only segmented sieve kernels exposed through a C ABI."""

from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import simd_width_of as simdwidthof


comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int, num_workers: Int):
    """Run work on the runtime pool without depending on max.algorithm."""
    initialize_runtime()
    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    @__parameter
    async def run_worker(worker: Int):
        var start = worker * chunk_size + min(worker, extra_items)
        var count = chunk_size + Int(worker < extra_items)
        for i in range(count):
            func(start + i)

    var tasks = TaskGroup()
    for worker in range(min(num_workers, num_work_items)):
        tasks.create_task(run_worker(worker))
    tasks.wait()


def fill_ones(flags: BPtr, n: Int):
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = 8 * W
    var i = 0
    var ones = SIMD[DType.uint8, BYTE_W](1)
    while i + BYTE_W <= n:
        flags.store(i, ones)
        i += BYTE_W
    while i < n:
        flags[i] = 1
        i += 1


def fill_presieved(flags: BPtr, low: Int, n: Int):
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = 8 * W
    comptime VECTORS = 15
    comptime BLOCK = BYTE_W * VECTORS
    var prefix = min(n, BLOCK)
    var i = 0
    while i < prefix:
        var value = low + 2 * i
        flags[i] = UInt8(value % 3 != 0 and value % 5 != 0)
        i += 1
    while i + BLOCK <= n:
        comptime for vector in range(VECTORS):
            flags.store(
                i + vector * BYTE_W,
                flags.load[width=BYTE_W](vector * BYTE_W),
            )
        i += BLOCK
    while i < n:
        flags[i] = flags[i % BLOCK]
        i += 1
    if low <= 3 and 3 <= low + 2 * (n - 1):
        flags[(3 - low) // 2] = 1
    if low <= 5 and 5 <= low + 2 * (n - 1):
        flags[(5 - low) // 2] = 1


def count_ones(flags: BPtr, n: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = 8 * W
    var total = 0
    var i = 0
    while i + BYTE_W <= n:
        total += Int(flags.load[width=BYTE_W](i).reduce_add())
        i += BYTE_W
    while i < n:
        if flags[i] != 0:
            total += 1
        i += 1
    return total


def sieve_segment(
    low: Int,
    high: Int,
    base: IPtr,
    nbase: Int,
    flags: BPtr,
    n: Int,
) -> Int:
    var presieved = nbase > 10000
    if presieved:
        fill_presieved(flags, low, n)
    else:
        fill_ones(flags, n)
    for i in range(nbase):
        var p = Int(base[i])
        if p == 2 or (presieved and p <= 5):
            continue
        if p > high // p:
            break
        var first = p * p
        if first < low:
            first = (low // p) * p
            if first < low:
                first += p
        if first % 2 == 0:
            first += p
        var j = (first - low) // 2
        while j < n:
            flags[j] = 0
            j += p
    return count_ones(flags, n)


def collect_segment(
    low: Int,
    flags: BPtr,
    n: Int,
    values: IPtr,
) -> Int:
    var count = 0
    for i in range(n):
        if flags[i] != 0:
            values[count] = Int64(low + 2 * i)
            count += 1
    return count


@export("mps_small_sieve")
def mps_small_sieve(stop: Int, flags_addr: Int, n: Int) abi("C") -> Int:
    if stop < 0 or n < 0 or (n > 0 and flags_addr == 0):
        return -1
    if n == 0:
        return 0
    var flags = BPtr(unsafe_from_address=flags_addr)
    fill_ones(flags, n)
    var i = 0
    while i < n:
        var p = 2 * i + 3
        if p > stop // p:
            break
        if flags[i] != 0:
            var j = (p * p - 3) // 2
            while j < n:
                flags[j] = 0
                j += p
        i += 1
    return count_ones(flags, n)


@export("mps_segment_sieve")
def mps_segment_sieve(
    low: Int,
    high: Int,
    base_addr: Int,
    nbase: Int,
    flags_addr: Int,
    n: Int,
) abi("C") -> Int:
    if (
        low < 3 or low % 2 == 0 or high < low or n < 1
        or nbase < 0 or base_addr == 0 or flags_addr == 0
        or n != (high - low) // 2 + 1
    ):
        return -1
    var base = IPtr(unsafe_from_address=base_addr)
    var flags = BPtr(unsafe_from_address=flags_addr)
    return sieve_segment(low, high, base, nbase, flags, n)


@export("mps_sieve_batch")
def mps_sieve_batch(
    low: Int,
    stop: Int,
    base_addr: Int,
    nbase: Int,
    flags_addr: Int,
    segment_capacity: Int,
    counts_addr: Int,
    nsegments: Int,
    num_workers: Int,
) abi("C") -> Int:
    if (
        low < 3 or low % 2 == 0 or stop < low
        or nbase < 0 or base_addr == 0 or flags_addr == 0
        or segment_capacity < 1 or counts_addr == 0
        or nsegments < 1 or num_workers < 1
    ):
        return -1
    var total_n = (stop - low) // 2 + 1
    if nsegments != (total_n + segment_capacity - 1) // segment_capacity:
        return -1
    @parameter
    def work(segment: Int):
        var base = IPtr(unsafe_from_address=base_addr)
        var flags = BPtr(unsafe_from_address=flags_addr)
        var counts = IPtr(unsafe_from_address=counts_addr)
        var offset = segment * segment_capacity
        var segment_low = low + 2 * offset
        var n = min(segment_capacity, (stop - segment_low) // 2 + 1)
        var segment_high = segment_low + 2 * (n - 1)
        counts[segment] = Int64(
            sieve_segment(
                segment_low,
                segment_high,
                base,
                nbase,
                flags + offset,
                n,
            )
        )

    if num_workers > 1:
        parallelize[work](nsegments, num_workers)
    else:
        for segment in range(nsegments):
            work(segment)
    return 0


@export("mps_collect_segment")
def mps_collect_segment(
    low: Int,
    flags_addr: Int,
    n: Int,
    values_addr: Int,
) abi("C") -> Int:
    if low < 3 or low % 2 == 0 or n < 0:
        return -1
    if n == 0:
        return 0
    if flags_addr == 0 or values_addr == 0:
        return -1
    var flags = BPtr(unsafe_from_address=flags_addr)
    var values = IPtr(unsafe_from_address=values_addr)
    return collect_segment(low, flags, n, values)


@export("mps_collect_batch")
def mps_collect_batch(
    low: Int,
    flags_addr: Int,
    n: Int,
    segment_capacity: Int,
    offsets_addr: Int,
    nsegments: Int,
    values_addr: Int,
    num_workers: Int,
) abi("C") -> Int:
    if (
        low < 3 or low % 2 == 0 or n < 0 or segment_capacity < 1
        or nsegments < 1 or num_workers < 1
        or flags_addr == 0 or offsets_addr == 0
    ):
        return -1
    if nsegments != (n + segment_capacity - 1) // segment_capacity:
        return -1
    if values_addr == 0:
        # NumPy supplies a non-null sentinel for empty arrays, but reject a
        # genuinely null output pointer before constructing UnsafePointer.
        return -1
    @parameter
    def work(segment: Int):
        var flags = BPtr(unsafe_from_address=flags_addr)
        var offsets = IPtr(unsafe_from_address=offsets_addr)
        var values = IPtr(unsafe_from_address=values_addr)
        var flag_offset = segment * segment_capacity
        var segment_n = min(segment_capacity, n - flag_offset)
        _ = collect_segment(
            low + 2 * flag_offset,
            flags + flag_offset,
            segment_n,
            values + Int(offsets[segment]),
        )

    if num_workers > 1:
        parallelize[work](nsegments, num_workers)
    else:
        for segment in range(nsegments):
            work(segment)
    return 0


@export("mps_select_flag")
def mps_select_flag(
    low: Int,
    flags_addr: Int,
    n: Int,
    rank: Int,
) abi("C") -> Int:
    if low < 3 or low % 2 == 0 or n < 1 or rank < 1 or flags_addr == 0:
        return -1
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = 8 * W
    var flags = BPtr(unsafe_from_address=flags_addr)
    var remaining = rank
    var i = 0
    while i + BYTE_W <= n:
        var block_count = Int(flags.load[width=BYTE_W](i).reduce_add())
        if remaining > block_count:
            remaining -= block_count
            i += BYTE_W
        else:
            break
    while i < n:
        if flags[i] != 0:
            remaining -= 1
            if remaining == 0:
                return low + 2 * i
        i += 1
    return -1


@export("mps_count_constellations")
def mps_count_constellations(
    values_addr: Int,
    n: Int,
    kind: Int,
) abi("C") -> Int:
    if n < 0 or kind < 2 or kind > 6:
        return -1
    if n == 0:
        return 0
    if values_addr == 0:
        return -1
    var values = IPtr(unsafe_from_address=values_addr)
    var total = 0
    if kind == 2:
        for i in range(1, n):
            if values[i] - values[i - 1] == 2:
                total += 1
        return total
    if kind == 3:
        for i in range(2, n):
            var a = values[i - 2]
            var b = values[i - 1]
            var c = values[i]
            if (b - a == 2 and c - b == 4) or (b - a == 4 and c - b == 2):
                total += 1
        return total
    if kind == 4:
        for i in range(3, n):
            if (
                values[i - 2] - values[i - 3] == 2
                and values[i - 1] - values[i - 2] == 4
                and values[i] - values[i - 1] == 2
            ):
                total += 1
        return total
    if kind == 5:
        for i in range(4, n):
            var d0 = values[i - 3] - values[i - 4]
            var d1 = values[i - 2] - values[i - 3]
            var d2 = values[i - 1] - values[i - 2]
            var d3 = values[i] - values[i - 1]
            if (
                (d0 == 2 and d1 == 4 and d2 == 2 and d3 == 4)
                or (d0 == 4 and d1 == 2 and d2 == 4 and d3 == 2)
            ):
                total += 1
        return total
    if kind == 6:
        for i in range(5, n):
            if (
                values[i - 4] - values[i - 5] == 4
                and values[i - 3] - values[i - 4] == 2
                and values[i - 2] - values[i - 3] == 4
                and values[i - 1] - values[i - 2] == 2
                and values[i] - values[i - 1] == 4
            ):
                total += 1
        return total
    return 0


@export("mps_count_flag_constellations")
def mps_count_flag_constellations(
    flags_addr: Int,
    n: Int,
    kind: Int,
) abi("C") -> Int:
    if n < 0 or kind < 2 or kind > 6:
        return -1
    if n == 0:
        return 0
    if flags_addr == 0:
        return -1
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = 8 * W
    var flags = BPtr(unsafe_from_address=flags_addr)
    var total = 0
    var i = 0
    if kind == 2:
        while i + BYTE_W + 1 <= n:
            total += Int(
                (
                    flags.load[width=BYTE_W](i)
                    & flags.load[width=BYTE_W](i + 1)
                ).reduce_add()
            )
            i += BYTE_W
        while i + 1 < n:
            if flags[i] != 0 and flags[i + 1] != 0:
                total += 1
            i += 1
        return total
    if kind == 3:
        while i + BYTE_W + 3 <= n:
            var first = (
                flags.load[width=BYTE_W](i)
                & flags.load[width=BYTE_W](i + 1)
                & flags.load[width=BYTE_W](i + 3)
            )
            var second = (
                flags.load[width=BYTE_W](i)
                & flags.load[width=BYTE_W](i + 2)
                & flags.load[width=BYTE_W](i + 3)
            )
            total += Int((first | second).reduce_add())
            i += BYTE_W
        while i + 3 < n:
            if (
                (
                    flags[i] != 0
                    and flags[i + 1] != 0
                    and flags[i + 3] != 0
                )
                or (
                    flags[i] != 0
                    and flags[i + 2] != 0
                    and flags[i + 3] != 0
                )
            ):
                total += 1
            i += 1
        return total
    if kind == 4:
        while i + BYTE_W + 4 <= n:
            total += Int(
                (
                    flags.load[width=BYTE_W](i)
                    & flags.load[width=BYTE_W](i + 1)
                    & flags.load[width=BYTE_W](i + 3)
                    & flags.load[width=BYTE_W](i + 4)
                ).reduce_add()
            )
            i += BYTE_W
        while i + 4 < n:
            if (
                flags[i] != 0
                and flags[i + 1] != 0
                and flags[i + 3] != 0
                and flags[i + 4] != 0
            ):
                total += 1
            i += 1
        return total
    if kind == 5:
        while i + BYTE_W + 6 <= n:
            var first = (
                flags.load[width=BYTE_W](i)
                & flags.load[width=BYTE_W](i + 1)
                & flags.load[width=BYTE_W](i + 3)
                & flags.load[width=BYTE_W](i + 4)
                & flags.load[width=BYTE_W](i + 6)
            )
            var second = (
                flags.load[width=BYTE_W](i)
                & flags.load[width=BYTE_W](i + 2)
                & flags.load[width=BYTE_W](i + 3)
                & flags.load[width=BYTE_W](i + 5)
                & flags.load[width=BYTE_W](i + 6)
            )
            total += Int((first | second).reduce_add())
            i += BYTE_W
        while i + 6 < n:
            if (
                (
                    flags[i] != 0
                    and flags[i + 1] != 0
                    and flags[i + 3] != 0
                    and flags[i + 4] != 0
                    and flags[i + 6] != 0
                )
                or (
                    flags[i] != 0
                    and flags[i + 2] != 0
                    and flags[i + 3] != 0
                    and flags[i + 5] != 0
                    and flags[i + 6] != 0
                )
            ):
                total += 1
            i += 1
        return total
    if kind == 6:
        while i + BYTE_W + 8 <= n:
            total += Int(
                (
                    flags.load[width=BYTE_W](i)
                    & flags.load[width=BYTE_W](i + 2)
                    & flags.load[width=BYTE_W](i + 3)
                    & flags.load[width=BYTE_W](i + 5)
                    & flags.load[width=BYTE_W](i + 6)
                    & flags.load[width=BYTE_W](i + 8)
                ).reduce_add()
            )
            i += BYTE_W
        while i + 8 < n:
            if (
                flags[i] != 0
                and flags[i + 2] != 0
                and flags[i + 3] != 0
                and flags[i + 5] != 0
                and flags[i + 6] != 0
                and flags[i + 8] != 0
            ):
                total += 1
            i += 1
        return total
    return 0
