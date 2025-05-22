import time
import random
from functools import reduce
from math import gcd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import statistics
from math import prod, lcm
def timed_run(fn, a, m, repeat=5):
    times = []
    for _ in range(repeat):
        fn(a, m)  # warm-up
        start = time.perf_counter()
        fn(a, m)
        end = time.perf_counter()
        times.append(end - start)
    return statistics.mean(times), statistics.stdev(times)
# --- 基本工具函数 ---
def extended_gcd(a, b):
    """非递归扩展欧几里得算法"""
    x0, x1 = 1, 0
    y0, y1 = 0, 1
    while b != 0:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0

def modinv(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m

# --- 1. Naive CRT ---
def crt_naive(a, m):
    M = reduce(lambda x, y: x * y, m)
    result = 0
    for ai, mi in zip(a, m):
        Mi = M // mi
        inv = modinv(Mi, mi)
        result = (result + ai * Mi * inv) % M
    return result

# --- 2. Garner ---
def crt_garner(a, m):
    x = a[0]
    M = m[0]
    for i in range(1, len(a)):
        inv = modinv(M, m[i])
        t = ((a[i] - x) * inv) % m[i]
        x = x + M * t
        M *= m[i]
    return x % M

# --- 3. Divide & Conquer ---
def crt_merge_pair(a1, m1, a2, m2):
    """合并两个模方程 a ≡ a1 mod m1 和 a ≡ a2 mod m2"""
    g = gcd(m1, m2)
    if g != 1:
        raise ValueError("Moduli not coprime")

    inv_m1 = modinv(m1, m2)  # 直接使用全局modinv，内部调用非递归extended_gcd
    x = ((a2 - a1) * inv_m1) % m2
    result = (a1 + x * m1) % (m1 * m2)
    return result, m1 * m2



def crt_iterative(a, m):
    assert len(a) == len(m)
    n = len(a)
    a = a[:]
    m = m[:]

    while n > 1:
        new_a = []
        new_m = []
        i = 0
        while i + 1 < n:
            a1, m1 = a[i], m[i]
            a2, m2 = a[i+1], m[i+1]
            inv = pow(m1, -1, m2)
            x = ((a2 - a1) * inv) % m2
            new_a.append((a1 + x * m1) % (m1 * m2))
            new_m.append(m1 * m2)
            i += 2
        if i < n:
            new_a.append(a[i])
            new_m.append(m[i])
        a, m = new_a, new_m
        n = len(a)
    return a[0]


def is_pairwise_coprime(moduli):
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                return False
    return True

def gen_data(n):
    base = 10**5
    m = []
    a = []
    current = base
    while len(m) < n:
        # 尝试加入 current 这个模数
        if all(gcd(current, mi) == 1 for mi in m):  # 保证两两互素
            m.append(current)
            a.append(random.randint(1, current - 1))
        current += 1
    return a, m
def crt_merge_pair(a1, m1, a2, m2):
    """合并两个模方程 a ≡ a1 mod m1 和 a ≡ a2 mod m2（模数需互素）"""
    g = gcd(m1, m2)
    if g != 1:
        raise ValueError("Moduli not coprime")

    inv = modinv(m1, m2)
    x = ((a2 - a1) * inv) % m2
    result = (a1 + x * m1) % (m1 * m2)
    return result, m1 * m2

def crt_iterative_parallel(a_list, m_list, max_workers=None):
    """并行化版本的 Divide-Conq CRT"""
    assert len(a_list) == len(m_list)
    a = a_list[:]
    m = m_list[:]
    n = len(a)

    while n > 1:
        pairs = []
        for i in range(0, n, 2):
            if i + 1 == n:
                # 奇数项，最后一个单独留下
                pairs.append(((a[i], m[i]), None))
            else:
                pairs.append(((a[i], m[i]), (a[i+1], m[i+1])))

        def merge_task(pair):
            left, right = pair
            if right is None:
                return left
            return crt_merge_pair(left[0], left[1], right[0], right[1])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(merge_task, pairs))
        # 解包结果
        a, m = zip(*results)
        a = list(a)
        m = list(m)
        n = len(a)

    return a[0]
def crt_merge_pair(a1, m1, a2, m2):
    """合并两个模方程 a ≡ a1 mod m1 和 a ≡ a2 mod m2（模数需互素）"""
    g = gcd(m1, m2)
    if g != 1:
        raise ValueError("Moduli not coprime")

    inv = modinv(m1, m2)
    x = ((a2 - a1) * inv) % m2
    result = (a1 + x * m1) % (m1 * m2)
    return result, m1 * m2

def merge_task_mp(pair):
    left, right = pair
    if right is None:
        return left
    return crt_merge_pair(left[0], left[1], right[0], right[1])

def crt_iterative_parallel_mp(a_list, m_list, processes=None):
    """多进程并行 Divide-Conquer CRT"""
    assert len(a_list) == len(m_list)
    a = a_list[:]
    m = m_list[:]
    n = len(a)

    if processes is None:
        processes = cpu_count()

    while n > 1:
        pairs = []
        for i in range(0, n, 2):
            if i + 1 == n:
                pairs.append(((a[i], m[i]), None))
            else:
                pairs.append(((a[i], m[i]), (a[i+1], m[i+1])))

        with Pool(processes=processes) as pool:
            results = pool.map(merge_task_mp, pairs)

        a, m = zip(*results)
        a = list(a)
        m = list(m)
        n = len(a)

    return a[0]
def crt_rank_based(a, m):
    from math import prod

    k = len(a)
    M = prod(m)
    Mi = [M // mi for mi in m]
    inv_Mi = [modinv(Mi[i], m[i]) for i in range(k)]
    chi_ik = [(inv_Mi[i] * a[i]) % m[i] for i in range(k)]

    # 近似秩（floor of fractional sum）
    approx_rank = sum(chi_ik[i] / m[i] for i in range(k))
    rho_hat = int(approx_rank)

    # 冗余通道：X ≡ x̂ mod 2
    x_hat = sum(Mi[i] * chi_ik[i] for i in range(k)) - rho_hat * M
    chi0 = x_hat % 2

    # 校正项 δ
    chi0_hat = (sum(c % 2 for c in chi_ik) + (rho_hat % 2)) % 2
    delta = (chi0 + chi0_hat) % 2

    # 最终重建
    rank = rho_hat + delta
    x = (sum(Mi[i] * chi_ik[i] for i in range(k)) - rank * M) % M
    return x
def crt_rank_based_integer(a, m):
    """
    使用纯整数Rank估计的优化版中国剩余定理。
    基于最小冗余RNS思想，避免浮点除法。
    """
    from math import prod, lcm

    k = len(a)
    M = prod(m)
    Mi = [M // mi for mi in m]
    inv_Mi = [modinv(Mi[i], m[i]) for i in range(k)]
    chi_ik = [(inv_Mi[i] * a[i]) % m[i] for i in range(k)]

    # 使用最小公倍数扩大精度，避免除法
    scale = lcm(*m)
    approx_rank_scaled = sum((chi_ik[i] * (scale // m[i])) for i in range(k))
    rho_hat = approx_rank_scaled // scale

    x_hat = sum(Mi[i] * chi_ik[i] for i in range(k))
    chi0 = (x_hat - rho_hat * M) % 2

    chi0_hat = (sum(c % 2 for c in chi_ik) + (rho_hat % 2)) % 2
    delta = (chi0 + chi0_hat) % 2

    rank = rho_hat + delta
    x = (x_hat - rank * M) % M
    return x
def crt_combined_garner_rank(a, m, split_at=None):
    """
    将模数分为两段：前段用 Garner，后段用 Rank-Based（Int）合并结果。
    split_at: 指定前段使用 Garner 的模数个数；默认为一半
    """
    from math import prod, lcm

    n = len(a)
    if split_at is None:
        split_at = n // 2  # 默认前一半 Garner，后一半 Rank-Based

    # --- 第一段用 Garner 构造初解 x0 ---
    a1 = a[:split_at]
    m1 = m[:split_at]
    x0 = crt_garner(a1, m1)
    M0 = prod(m1)

    # --- 第二段用 Rank-Based 构造校正 delta ---
    a2 = a[split_at:]
    m2 = m[split_at:]
    if not m2:
        return x0 % prod(m)  # 所有模数都被 Garner 处理

    # 构造新的方程组： x ≡ x0 mod M0，与 a2 mod m2 合并
    a_ext = [x0] + a2
    m_ext = [M0] + m2

    # 使用整数化 Rank-Based 方法处理
    M = prod(m_ext)
    Mi = [M // mi for mi in m_ext]
    inv_Mi = [modinv(Mi[i], m_ext[i]) for i in range(len(m_ext))]
    chi_ik = [(inv_Mi[i] * a_ext[i]) % m_ext[i] for i in range(len(m_ext))]

    scale = lcm(*m_ext)
    approx_rank_scaled = sum(chi_ik[i] * (scale // m_ext[i]) for i in range(len(m_ext)))
    rho_hat = approx_rank_scaled // scale

    x_hat = sum(Mi[i] * chi_ik[i] for i in range(len(m_ext)))
    chi0 = (x_hat - rho_hat * M) % 2
    chi0_hat = (sum(c % 2 for c in chi_ik) + (rho_hat % 2)) % 2
    delta = (chi0 + chi0_hat) % 2

    rank = rho_hat + delta
    x = (x_hat - rank * M) % M
    return x
def _rank_worker(args):
    i, a_i, m_i, M = args
    Mi = M // m_i
    inv_Mi = modinv(Mi, m_i)
    chi_ik = (inv_Mi * a_i) % m_i
    return (i, chi_ik, m_i, Mi)

def crt_rank_based_parallel(a, m, processes=None):
    """
    并行版本的 Rank-Based (Int)，使用多进程计算 Mi 和 chi_ik。
    """
    assert len(a) == len(m)
    n = len(a)
    M = prod(m)
    scale = lcm(*m)

    if processes is None:
        processes = min(cpu_count(), n)

    with Pool(processes=processes) as pool:
        results = pool.map(_rank_worker, [(i, a[i], m[i], M) for i in range(n)])

    # 排序结果（因进程打乱）
    results.sort()
    chi_ik_list = [x[1] for x in results]
    m_list = [x[2] for x in results]
    Mi_list = [x[3] for x in results]

    approx_rank_scaled = sum(chi_ik_list[i] * (scale // m_list[i]) for i in range(n))
    rho_hat = approx_rank_scaled // scale

    x_hat = sum(Mi_list[i] * chi_ik_list[i] for i in range(n))
    chi0 = (x_hat - rho_hat * M) % 2
    chi0_hat = (sum(c % 2 for c in chi_ik_list) + (rho_hat % 2)) % 2
    delta = (chi0 + chi0_hat) % 2

    rank = rho_hat + delta
    x = (x_hat - rank * M) % M
    return x
def crt_divide_rank_combined(a_list, m_list, group_size=16):
    """
    分组版中国剩余定理：
    每 group_size 个模数作为一个子组，使用 Rank-Based (Int) 合并为一个局部解；
    然后使用 Divide-Conq 合并所有局部解。
    """
    assert len(a_list) == len(m_list)
    n = len(a_list)

    grouped_solutions = []
    grouped_moduli = []

    for i in range(0, n, group_size):
        a_chunk = a_list[i:i + group_size]
        m_chunk = m_list[i:i + group_size]

        # 用 Rank-Based (Int) 合并每组
        x_i = crt_rank_based_integer(a_chunk, m_chunk)
        M_i = 1
        for mi in m_chunk:
            M_i *= mi

        grouped_solutions.append(x_i)
        grouped_moduli.append(M_i)

    # 最后用 Divide-Conq 合并这些 (x_i mod M_i)
    final_result = crt_iterative(grouped_solutions, grouped_moduli)
    return final_result
def crt_divide_rank_combined_fast(a_list, m_list, group_size=16):
    """
    内联优化版本的 Divide+Rank：
    直接在主函数内完成每组 Rank-Based 的合并逻辑，减少函数调用和对象开销。
    """
    from math import prod, lcm

    assert len(a_list) == len(m_list)
    n = len(a_list)

    grouped_solutions = []
    grouped_moduli = []

    for i in range(0, n, group_size):
        a_chunk = a_list[i:i + group_size]
        m_chunk = m_list[i:i + group_size]
        k = len(a_chunk)

        # --- 内联 Rank-Based ---
        M = 1
        for mi in m_chunk:
            M *= mi

        Mi = [M // m_chunk[j] for j in range(k)]
        inv_Mi = [modinv(Mi[j], m_chunk[j]) for j in range(k)]
        chi_ik = [(inv_Mi[j] * a_chunk[j]) % m_chunk[j] for j in range(k)]

        scale = lcm(*m_chunk)
        approx_rank_scaled = sum(chi_ik[j] * (scale // m_chunk[j]) for j in range(k))
        rho_hat = approx_rank_scaled // scale

        x_hat = sum(Mi[j] * chi_ik[j] for j in range(k))
        chi0 = (x_hat - rho_hat * M) % 2
        chi0_hat = (sum(c % 2 for c in chi_ik) + (rho_hat % 2)) % 2
        delta = (chi0 + chi0_hat) % 2
        rank = rho_hat + delta

        x_group = (x_hat - rank * M) % M

        grouped_solutions.append(x_group)
        grouped_moduli.append(M)

    # --- 最后用 Divide-Conq 合并所有组 ---
    return crt_iterative(grouped_solutions, grouped_moduli)

# --- 测试函数 ---
def benchmark():
    print("Benchmarking CRT Solvers — Optimized Accuracy + Stability\n")
    for size in [10, 100, 300,1000]:
        a, m = gen_data(size)
        print(f"\n--- Testing with n = {size} ---")

        methods = [
            ("Naive", crt_naive),
            ("Garner", crt_garner),
            ("Divide-Conq", crt_iterative),
            ("Divide-Conq Thread", lambda a, m: crt_iterative_parallel(a, m)),
            ("Divide-Conq MP", lambda a, m: crt_iterative_parallel_mp(a, m)),
            ("Rank-Based (Int)", crt_rank_based_integer),
            ("Garner+Rank", lambda a, m: crt_combined_garner_rank(a, m)),
            ("Rank-Based Parallel", lambda a, m: crt_rank_based_parallel(a, m)),
            ("crt_divide_rank_combined",lambda a,m:crt_divide_rank_combined(a,m)),
            ("Divide+Rank (Fast)", lambda a, m: crt_divide_rank_combined_fast(a, m, group_size=16))

        ]

        # 随机打乱算法顺序，减少顺序偏差
        random.shuffle(methods)

        # 选 Naive 为参考正确值
        reference = crt_naive(a, m)

        for name, fn in methods:
            try:
                avg_time, stddev = timed_run(fn, a, m, repeat=5)
                result = fn(a, m)
                status = "✅" if result == reference else f"❌ Wrong! Expected: {reference}"
                print(f"{name:20s} Avg: {avg_time:.6f}s  StdDev: {stddev:.6f}s   {status}")
            except Exception as e:
                print(f"{name:20s} Error: {e}")




if __name__ == "__main__":
    benchmark()
