"""High-signal diagnostic debugging, AI/weight-space mechanics, and meta self-improvement generators.

Every generator produces non-trivial, parameterized first-principles reasoning traces structured around
RPR (`[L1 Foundations/Invariants] -> [L2 Hypothesis / Naive-Trap Check] -> [AUDIT / Root Cause] -> [L3 Verified Fix]`),
with executable Python unit tests verified at corpus build time.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any


# ===========================================================================
# PILLAR A: DIAGNOSTIC ROOT-CAUSE DEBUGGING & COGNITIVE-TRAP REASONING
# ===========================================================================
def make_diagnostic_debug_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Generate high-signal root-cause bug-fixing and cognitive-trap problems."""
    case = idx % 10

    if case == 0:
        # Upper-bound binary search infinite-loop bug hunt & repair
        fn = rng.choice(["last_feasible_rate", "max_passing_threshold", "upper_valid_index"])
        target_Example = rng.randint(5, 25)
        buggy = (
            f"def {fn}(arr: list[int], limit: int) -> int:\n"
            "    # Return largest index i with arr[i] <= limit, or -1 if none.\n"
            "    if not arr or arr[0] > limit:\n"
            "        return -1\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] <= limit:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo"
        )
        fixed = (
            f"def {fn}(arr: list[int], limit: int) -> int:\n"
            "    if not arr or arr[0] > limit:\n"
            "        return -1\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi + 1) // 2\n"
            "        if arr[mid] <= limit:\n"
            "            lo = mid\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return lo"
        )
        prompt = (
            f"Diagnose why the following sorted-array search function hangs indefinitely on certain inputs, "
            f"prove the failure with a minimal 2-element counterexample, and output ONLY the fixed ```python block:\n\n"
            f"```python\n{buggy}\n```"
        )
        think = (
            f"[L1] Loop invariant: answer lies in [lo, hi] and interval width (hi - lo) must strictly decrease every iteration.\n"
            f"[L2] Test 2-element interval lo=0, hi=1 with arr=[{target_Example}, {target_Example + 4}], limit={target_Example + 2}: "
            f"floor division gives mid = (0 + 1) // 2 = 0. Since arr[0]={target_Example} <= {target_Example + 2}, branch sets lo = mid = 0, leaving [lo, hi] = [0, 1] unchanged -> infinite loop!\n"
            f"[AUDIT] Floor midpoint `mid = (lo + hi) // 2` violates progress whenever `lo = mid` is an update branch.\n"
            f"[REBUILD] Bias midpoint upward: `mid = (lo + hi + 1) // 2`. On [0, 1], mid=1: either `lo=1` (terminates) or `hi=0` (terminates)."
        )
        test_code = (
            f"assert {fn}([{target_Example}, {target_Example + 4}], {target_Example + 2}) == 0\n"
            f"assert {fn}([{target_Example}, {target_Example + 4}], {target_Example + 4}) == 1\n"
            f"assert {fn}([{target_Example}, {target_Example + 4}], {target_Example - 1}) == -1\n"
            f"assert {fn}([1, 3, 3, 5, 8], 3) == 2"
        )
        return {
            "pillar": "diagnostic_debug",
            "messages": [
                {"role": "system", "content": "Diagnose root causes from invariants inside <think>, verify with a concrete counterexample, and output the fixed Python code."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{fixed}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 1:
        # Diamond DAG vs True Cycle 3-Color DFS Bug Hunt
        fn = rng.choice(["has_dependency_cycle", "detect_directed_cycle", "contains_graph_cycle"])
        buggy = (
            f"def {fn}(graph: dict[str, list[str]]) -> bool:\n"
            "    visited = set()\n"
            "    def dfs(u: str) -> bool:\n"
            "        if u in visited:\n"
            "            return True\n"
            "        visited.add(u)\n"
            "        return any(dfs(v) for v in graph.get(u, []))\n"
            "    return any(dfs(node) for node in graph)"
        )
        fixed = (
            f"def {fn}(graph: dict[str, list[str]]) -> bool:\n"
            "    state: dict[str, int] = {}  # 0/absent=unvisited, 1=visiting, 2=done\n"
            "    def dfs(u: str) -> bool:\n"
            "        st = state.get(u, 0)\n"
            "        if st == 1:\n"
            "            return True\n"
            "        if st == 2:\n"
            "            return False\n"
            "        state[u] = 1\n"
            "        for v in graph.get(u, ()):\n"
            "            if dfs(v):\n"
            "                return True\n"
            "        state[u] = 2\n"
            "        return False\n"
            "    return any(state.get(n, 0) == 0 and dfs(n) for n in graph)"
        )
        think = (
            "[L1] A directed graph has a cycle iff DFS encounters a back-edge to an ancestor currently on the active recursion stack.\n"
            "[L2] Test diamond DAG A->[B, C], B->[D], C->[D]: `dfs(A)` visits B then D, leaving D in `visited`. When `dfs(C)` reaches D, "
            "the buggy check `if u in visited: return True` mistakes a harmless cross/forward edge to completed node D for a cycle!\n"
            "[AUDIT] Single boolean `visited` conflates `actively on stack` (gray=1) with `completely explored` (black=2).\n"
            "[REBUILD] Use 3-state coloring: return True only when `state[u] == 1` (active ancestor), return False immediately when `state[u] == 2`."
        )
        test_code = (
            f"assert {fn}({{'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}}) is False\n"
            f"assert {fn}({{'A': ['B'], 'B': ['C'], 'C': ['A']}}) is True\n"
            f"assert {fn}({{'X': ['X']}}) is True"
        )
        return {
            "pillar": "diagnostic_debug",
            "messages": [
                {"role": "system", "content": "Diagnose root causes from invariants inside <think> and output ONLY the fixed ```python block."},
                {"role": "user", "content": f"This cycle detector falsely flags valid DAGs as cyclic. Find the bug and return ONLY the fixed ```python block:\n\n```python\n{buggy}\n```"},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{fixed}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 2:
        # Numerical Log-Sum-Exp & Log-Softmax Overflow / Underflow Fix
        fn = rng.choice(["stable_log_softmax", "safe_log_probs", "numerically_stable_log_softmax"])
        code = (
            "import math\n\n"
            f"def {fn}(logits: list[float]) -> list[float]:\n"
            "    if not logits:\n"
            "        return []\n"
            "    m = max(logits)\n"
            "    lse = m + math.log(sum(math.exp(x - m) for x in logits))\n"
            "    return [x - lse for x in logits]"
        )
        prompt = (
            f"Naive `math.log(math.exp(x) / sum(math.exp(z) for z in logits))` crashes with `OverflowError` on `logits = [1000.0, 1001.0]` "
            f"and underflows to `ZeroDivisionError` on `logits = [-1000.0, -999.0]`. Write `{fn}(logits: list[float]) -> list[float]` "
            "that is algebraically exact and numerically stable across both extremes. Output ONLY a ```python block."
        )
        think = (
            "[L1] Shift invariance: log(exp(x_i) / sum_j exp(x_j)) = x_i - log(sum_j exp(x_j)).\n"
            "[L2] Factor out m = max(logits): log(sum_j exp(x_j)) = m + log(sum_j exp(x_j - m)). Since x_j - m <= 0 for all j, "
            "every `exp(x_j - m)` lies in (0, 1], preventing overflow, and at least one term equals exp(0) = 1, preventing sum underflow to 0!\n"
            "[L3] Return `[x - lse for x in logits]`; handle empty list."
        )
        test_code = (
            f"res_hi = {fn}([1000.0, 1000.0])\n"
            "assert abs(res_hi[0] - (-0.69314718)) < 1e-5 and abs(res_hi[1] - (-0.69314718)) < 1e-5\n"
            f"res_lo = {fn}([-1000.0, -1000.0])\n"
            "assert abs(res_lo[0] - (-0.69314718)) < 1e-5"
        )
        return {
            "pillar": "diagnostic_debug",
            "messages": [
                {"role": "system", "content": "Derive numerically stable implementations from mathematical invariants inside <think>."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 3:
        # Shortest Subarray Sum >= K WITH Negative Numbers (Why Two-Pointers Fails -> Monotonic Deque)
        fn = rng.choice(["shortest_subarray_at_least_k", "min_len_sum_ge_k"])
        code = (
            "from collections import deque\n\n"
            f"def {fn}(nums: list[int], k: int) -> int:\n"
            "    pref = [0]\n"
            "    for x in nums:\n"
            "        pref.append(pref[-1] + x)\n"
            "    dq: deque[int] = deque()\n"
            "    best = len(nums) + 1\n"
            "    for j, pj in enumerate(pref):\n"
            "        while dq and pj - pref[dq[0]] >= k:\n"
            "            best = min(best, j - dq.popleft())\n"
            "        while dq and pref[dq[-1]] >= pj:\n"
            "            dq.pop()\n"
            "        dq.append(j)\n"
            "    return best if best <= len(nums) else -1"
        )
        prompt = (
            f"Write `{fn}(nums: list[int], k: int) -> int` returning the minimum length of a non-empty contiguous subarray "
            "with sum >= `k` (or `-1` if none exists), where `nums` MAY contain negative integers. Explain in `<think>` why "
            "standard sliding window fails on negatives, then output ONLY a ```python block."
        )
        think = (
            "[L1] Subarray sum(i..j-1) = pref[j] - pref[i] >= k, minimizing j - i.\n"
            "[L2] Naive two-pointer shrink assumes prefix sums are monotonic! Counterexample: nums=[3, -2, 3], k=4: window [0..2] has sum 4 (len 3), "
            "shrinking left to [1..2] gives 1 < 4 so naive loop stops, missing nothing here, OR on [2, -1, 2, 1], k=3: non-monotonic dips hide shorter suffixes.\n"
            "[REBUILD] Maintain a monotonic increasing deque of indices `dq` over `pref`: (1) while `pref[j] - pref[dq[0]] >= k`, pop left (any later j' > j gives longer subarray); "
            "(2) while `pref[dq[-1]] >= pref[j]`, pop right (`j` has both a smaller/equal prefix value and a larger index, strictly dominating `dq[-1]`). O(n) time."
        )
        test_code = (
            f"assert {fn}([2, -1, 2], 3) == 3\n"
            f"assert {fn}([84, -37, 32, 40, 95], 167) == 3\n"
            f"assert {fn}([1, -2, 1], 3) == -1"
        )
        return {
            "pillar": "diagnostic_debug",
            "messages": [
                {"role": "system", "content": "Catch algorithmic counterexamples in <think> and output verified O(n) Python code."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 4:
        # Python Late-Binding Closure + Mutable Default Double Bug Repair
        fn = rng.choice(["make_step_pipeline", "build_offset_adders"])
        buggy = (
            f"def {fn}(offsets: list[int], history: list[int] = []):\n"
            "    fns = [lambda x: x + off for off in offsets]\n"
            "    history.append(len(offsets))\n"
            "    return fns, history"
        )
        fixed = (
            f"def {fn}(offsets: list[int], history: list[int] | None = None):\n"
            "    hist = [] if history is None else history\n"
            "    fns = [lambda x, off=off: x + off for off in offsets]\n"
            "    hist.append(len(offsets))\n"
            "    return fns, hist"
        )
        think = (
            "[L1] Two independent bugs exist in `buggy`: (1) mutable default `history: list[int] = []` persists across separate calls when omitted; "
            "(2) list-comprehension `lambda x: x + off` captures the loop cell `off` by reference (late binding), so every lambda adds `offsets[-1]`!\n"
            "[REBUILD] Default `history: list[int] | None = None` -> instantiate fresh `[]` per call when `None`; bind current loop value eagerly via `lambda x, off=off: x + off`."
        )
        test_code = (
            f"f1, h1 = {fn}([1, 5, 10])\n"
            f"f2, h2 = {fn}([2, 4])\n"
            "assert [fn(10) for fn in f1] == [11, 15, 20]\n"
            "assert h1 == [3] and h2 == [2]"
        )
        return {
            "pillar": "diagnostic_debug",
            "messages": [
                {"role": "system", "content": "Audit all hidden bugs in <think> and return ONLY the corrected ```python block."},
                {"role": "user", "content": f"Find and fix BOTH bugs in this Python helper. Output ONLY a ```python block:\n\n```python\n{buggy}\n```"},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{fixed}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 5:
        # Dijkstra vs Negative Edge Weight Bug Hunt & Bellman-Ford / Queue Re-Relaxation Repair
        fn = rng.choice(["shortest_path_with_rebates", "min_cost_directed_path"])
        code = (
            "import math\n\n"
            f"def {fn}(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int | None:\n"
            "    dist = [math.inf] * n\n"
            "    dist[src] = 0\n"
            "    for _ in range(max(0, n - 1)):\n"
            "        updated = False\n"
            "        for u, v, w in edges:\n"
            "            if dist[u] != math.inf and dist[u] + w < dist[v]:\n"
            "                dist[v] = dist[u] + w\n"
            "                updated = True\n"
            "        if not updated:\n"
            "            break\n"
            "    return None if dist[dst] == math.inf else int(dist[dst])"
        )
        prompt = (
            f"A teammate used standard Dijkstra (skipping popped nodes via a `seen` set) on a DAG/negative-rebate graph with `n` vertices `0..n-1`, "
            f"and it returned `2` instead of `1` on `edges=[(0, 1, 2), (0, 2, 5), (2, 1, -4)]` (`src=0, dst=1`). "
            f"Explain why greedy locking fails in `<think>`, and implement `{fn}(n, edges, src, dst) -> int | None` (assuming no negative cycles; `None` if unreachable). "
            "Output ONLY a ```python block."
        )
        think = (
            "[L1] Standard Dijkstra assumes edge weights w >= 0 so the minimum-dist unvisited vertex can never be improved via a longer detour.\n"
            "[L2] On (0,1,2), (0,2,5), (2,1,-4): Dijkstra pops node 1 first at dist=2 and marks `1 in seen`, then later pops node 2 at dist=5 "
            "and ignores `5 + (-4) = 1 < 2` because 1 was greedily locked!\n"
            "[REBUILD] Use Bellman-Ford with early-exit over at most n-1 relaxation passes: handles negative edges cleanly without greedy locking."
        )
        test_code = (
            f"assert {fn}(3, [(0, 1, 2), (0, 2, 5), (2, 1, -4)], 0, 1) == 1\n"
            f"assert {fn}(4, [(0, 1, 4), (1, 2, -2)], 0, 3) is None"
        )
        return {
            "pillar": "diagnostic_debug",
            "messages": [
                {"role": "system", "content": "Diagnose algorithmic invariant violations in <think> and output verified Python code."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 6:
        # Harmonic Mean Cognitive Trap (Round-trip / Equal-Distance Speeds)
        v1, v2 = rng.choice([(30, 60), (20, 80), (40, 60), (15, 60), (24, 72), (36, 72)])
        harm = (2 * v1 * v2) // (v1 + v2)
        naive = (v1 + v2) // 2
        dist = v1 * v2
        t1_h = dist // v1
        t2_h = dist // v2
        q = (
            f"An autonomous shuttle travels from depot A to depot B at {v1} km/h and returns along the exact same route from B to A at {v2} km/h. "
            'A junior engineer claims the round-trip average speed is the arithmetic mean. Audit that claim and return raw JSON: '
            '{"naive_mean_wrong": true, "exact_avg_kmh": <int>}.'
        )
        think = (
            f"[L1] Average speed = (total distance) / (total time), NOT the unweighted average of speeds.\n"
            f"[L2] Test naive claim ({v1}+{v2})/2 = {naive}: pick route distance D = {v1}*{v2} = {dist} km. "
            f"Outbound time = {dist}/{v1} = {t1_h} h; return time = {dist}/{v2} = {t2_h} h. "
            f"Total distance = 2*{dist} = {2*dist} km; total time = {t1_h}+{t2_h} = {t1_h+t2_h} h.\n"
            f"[AUDIT] Naive arithmetic mean {naive} fails because more time ({t1_h}h vs {t2_h}h) is spent at the slower speed {min(v1, v2)} km/h.\n"
            f"[REBUILD] Harmonic mean = {2*dist} / {t1_h+t2_h} = {harm} km/h."
        )
        exp = {"naive_mean_wrong": True, "exact_avg_kmh": harm}
        return {
            "pillar": "diagnostic_trap_math",
            "messages": [
                {"role": "system", "content": "Test intuitive claims against physical definitions in <think> and output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 7:
        # Bayes Base-Rate Fallacy Trap (Exact Integer Cohort Derivation)
        pop = 10000
        infected = rng.choice([50, 100, 200])  # 0.5%, 1%, 2%
        healthy = pop - infected
        fp_rate_pct = rng.choice([2, 5, 10])
        true_pos = infected  # 100% sensitivity for exact integer clarity or 90%
        sens_pct = rng.choice([90, 100])
        tp = (infected * sens_pct) // 100
        fp = (healthy * fp_rate_pct) // 100
        tot_pos = tp + fp
        q = (
            f"In a fleet of {pop} optical transceivers, exactly {infected} have a latent laser fault. "
            f"An automated bench test flags {sens_pct}% of faulty units as positive, and falsely flags {fp_rate_pct}% of healthy units as positive. "
            f"Compute the exact integer counts of true positives (`tp`), false positives (`fp`), and total flagged units (`flagged`) across the {pop} units. "
            'Output raw JSON: {"tp": <int>, "fp": <int>, "flagged": <int>}.'
        )
        think = (
            f"[L1] Partition {pop} units into faulty={infected} and healthy={pop}-{infected}={healthy} (avoid base-rate neglect!).\n"
            f"[L2] True positives: {infected} * {sens_pct}% = {tp}. False positives from large healthy pool: {healthy} * {fp_rate_pct}% = {fp}.\n"
            f"[L3] Total flagged = {tp} + {fp} = {tot_pos} (note fp={fp} rivals/exceeds tp={tp} due to low base prevalence)."
        )
        exp = {"tp": tp, "fp": fp, "flagged": tot_pos}
        return {
            "pillar": "diagnostic_trap_math",
            "messages": [
                {"role": "system", "content": "Avoid base-rate fallacies by computing exact cohort counts in <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 8:
        # Distributed Quorum & Split-Brain Sizing Trap (Paxos / Raft f faults + Read/Write Quorum Intersection)
        f_tol = rng.randint(1, 5)
        n_crash = 2 * f_tol + 1
        n_byz = 3 * f_tol + 1
        q = (
            f"A distributed consensus cluster must tolerate up to `f = {f_tol}` simultaneous node failures without split-brain. "
            f"Compute: (1) minimum nodes `n_crash` under crash-recovery (Raft/Paxos), (2) majority quorum size `quorum_crash`, "
            f"and (3) minimum nodes `n_bft` under arbitrary Byzantine faults (PBFT). "
            'Output raw JSON: {"n_crash": <int>, "quorum_crash": <int>, "n_bft": <int>}.'
        )
        quorum = f_tol + 1
        think = (
            f"[L1] Crash-fault safety requires any two quorums Q1, Q2 to intersect in >=1 node (`2*Q > N`) while surviving f={f_tol} dead nodes (`N - f >= Q`).\n"
            f"[L2] Setting Q = N - {f_tol} gives 2*(N - {f_tol}) > N => N >= 2*{f_tol} + 1 = {n_crash}, with quorum Q = {f_tol} + 1 = {quorum}. "
            f"For Byzantine faults, intersection must contain >= f+1 nodes so >=1 honest node overlaps after f liars: N >= 3*f + 1 = 3*{f_tol} + 1 = {n_byz}.\n"
            f"[L3] Verify crash: 2*{quorum}={2*quorum} > {n_crash} and {n_crash}-{f_tol}={quorum}."
        )
        exp = {"n_crash": n_crash, "quorum_crash": quorum, "n_bft": n_byz}
        return {
            "pillar": "diagnostic_trap_math",
            "messages": [
                {"role": "system", "content": "Derive distributed-systems safety bounds from quorum-intersection invariants in <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    # case == 9: 2-Turn Self-Correction on Fencepost + Off-by-One Rate-Limiter Bug!
    window_s = rng.choice([10, 60])
    max_req = rng.choice([3, 5])
    u1 = (
        f"Write `allow_request(timestamps: list[int], now: int) -> bool` for a sliding-window limiter allowing at most "
        f"`{max_req}` requests in the inclusive interval `[now - {window_s}, now]` (`timestamps` holds prior accepted times; "
        f"if allowed, append `now` in-place and return `True`, else leave `timestamps` unchanged and return `False`). Output ONLY a ```python block."
    )
    c1 = (
        f"def allow_request(timestamps: list[int], now: int) -> bool:\n"
        f"    cutoff = now - {window_s}\n"
        "    recent = sum(1 for t in timestamps if cutoff <= t <= now)\n"
        f"    if recent >= {max_req}:\n"
        "        return False\n"
        "    timestamps.append(now)\n"
        "    return True"
    )
    u2 = (
        f"Spec Revision: Product security clarified that a window of length `{window_s}` seconds is **half-open** `(now - {window_s}, now]` "
        f"(so an event at exact `t = now - {window_s}` has already expired and MUST NOT count against the cap `{max_req}`), "
        "AND expired timestamps `<= now - window_s` must be pruned in-place from `timestamps` on every call to prevent memory leaks. "
        "Rebuild `allow_request` cleanly. Output ONLY a ```python block."
    )
    c2 = (
        f"def allow_request(timestamps: list[int], now: int) -> bool:\n"
        f"    cutoff = now - {window_s}\n"
        "    timestamps[:] = [t for t in timestamps if cutoff < t <= now]\n"
        f"    if len(timestamps) >= {max_req}:\n"
        "        return False\n"
        "    timestamps.append(now)\n"
        "    return True"
    )
    test_code = (
        f"ts = [100] * {max_req}\n"
        f"assert allow_request(ts, 100 + {window_s}) is True\n"
        f"assert ts == [100 + {window_s}]\n"
        f"ts2 = [200] * {max_req}\n"
        f"assert allow_request(ts2, 200 + {window_s} - 1) is False\n"
        f"assert len(ts2) == {max_req}"
    )
    return {
        "pillar": "diagnostic_debug_2turn",
        "messages": [
            {"role": "system", "content": "Audit boundary conditions (`<` vs `<=`) and state-mutation invariants carefully inside <think>."},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": f"<think>[L1] Inclusive [now-{window_s}, now], cap={max_req}. [L2] Count cutoff <= t <= now; append only when count < {max_req}.</think>\n```python\n{c1}\n```"},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": f"<think>[AUDIT] Two assumptions changed: (1) boundary `cutoff <= t` INVALID -> strict `cutoff < t` for half-open `(now-{window_s}, now]`; (2) passive counting INVALID -> prune in-place via `timestamps[:] = [...]` before cap check so memory stays O(cap). [REBUILD] Prune in-place, check `len(timestamps) >= {max_req}`, append if under cap.</think>\n```python\n{c2}\n```"},
        ],
        "verifier": {"kind": "python_code", "test": test_code},
    }


# ===========================================================================
# PILLAR B: HOW AI & NEURAL WEIGHTS WORK (TRANSFORMERS, HYBRID CONV, LORA, BF16)
# ===========================================================================
def make_ai_weights_mechanics_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Deep, mathematically exact & executable mechanics of neural weights, KV cache, LoRA, SwiGLU, and quantization."""
    case = idx % 8

    if case == 0:
        # Exact GQA vs Hybrid Conv-Attention KV-Cache Memory Derivation!
        n_attn = rng.choice([8, 11, 16])
        n_conv = rng.choice([12, 13, 16])
        n_kv = rng.choice([2, 4, 8])
        d_head = rng.choice([64, 128])
        seq_len = rng.choice([2048, 4096, 8192])
        # BF16 = 2 bytes per element; K and V = 2 tensors per attention layer
        kv_bytes_per_tok_layer = 2 * n_kv * d_head * 2
        total_attn_kv_kb = (n_attn * seq_len * kv_bytes_per_tok_layer) // 1024
        q = (
            f"A {n_attn + n_conv}-layer hybrid language model has `{n_attn}` Grouped-Query Attention (GQA) layers "
            f"(`n_kv_heads={n_kv}`, `head_dim={d_head}`) and `{n_conv}` short-convolution layers (which store fixed-size state instead of O(L) KV cache). "
            f"In `bfloat16` (2 bytes/element), compute: (1) `bytes_per_token_per_attn_layer` for K+V, and (2) `total_attn_kv_kib` (exact KiB = bytes / 1024) "
            f"for a single sequence of length `L = {seq_len}`. Output raw JSON: "
            '{"bytes_per_token_per_attn_layer": <int>, "total_attn_kv_kib": <int>}.'
        )
        think = (
            f"[L1] Only the {n_attn} attention layers allocate O(L) KV cache; each stores K and V (factor 2) across {n_kv} KV heads of dim {d_head} at 2 bytes (bfloat16).\n"
            f"[L2] bytes_per_token_per_attn_layer = 2 (K,V) * {n_kv} * {d_head} * 2 (bytes) = {kv_bytes_per_tok_layer} bytes. "
            f"Across L={seq_len} tokens ({seq_len // 1024} * 1024): per-layer KiB = ({seq_len} // 1024) * {kv_bytes_per_tok_layer} = {(seq_len // 1024) * kv_bytes_per_tok_layer} KiB. "
            f"Multiply by {n_attn} attention layers: {n_attn} * {(seq_len // 1024) * kv_bytes_per_tok_layer} = {total_attn_kv_kb} KiB.\n"
            f"[L3] Note the {n_conv} conv layers save {n_conv}/{n_attn + n_conv} of sequence-scaling KV memory."
        )
        exp = {"bytes_per_token_per_attn_layer": kv_bytes_per_tok_layer, "total_attn_kv_kib": total_attn_kv_kb}
        return {
            "pillar": "ai_weights_mechanics",
            "messages": [
                {"role": "system", "content": "Compute neural architecture tensor shapes and memory footprints from first principles inside <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 1:
        # Executable LoRA Forward + Lossless Weight Fusion Verification
        fn_fwd = "lora_linear_forward_and_fuse"
        rank = rng.choice([8, 16, 32])
        alpha = rank * 2
        code = (
            "import numpy as np\n\n"
            f"def {fn_fwd}(x: np.ndarray, w_base: np.ndarray, lora_a: np.ndarray, lora_b: np.ndarray, rank: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:\n"
            "    \"\"\"w_base: (out_dim, in_dim), lora_a: (in_dim, rank), lora_b: (rank, out_dim), x: (seq, in_dim).\n"
            "    Return (y_unmerged, w_fused) where FusedLinear(x) == y_unmerged.\"\"\"\n"
            "    scale = alpha / float(rank)\n"
            "    y_base = x @ w_base.T\n"
            "    y_lora = scale * ((x @ lora_a) @ lora_b)\n"
            "    w_fused = w_base + scale * (lora_b.T @ lora_a.T)\n"
            "    return y_base + y_lora, w_fused"
        )
        prompt = (
            f"Implement `{fn_fwd}(x, w_base, lora_a, lora_b, rank, alpha) -> tuple[np.ndarray, np.ndarray]` in NumPy given "
            "`w_base` shape `(out_dim, in_dim)`, MLX-convention `lora_a` shape `(in_dim, rank)`, `lora_b` shape `(rank, out_dim)`, "
            "and `x` shape `(seq, in_dim)`. Return `(y_unmerged, w_fused)` such that `x @ w_fused.T` matches `y_unmerged` up to floating-point precision. "
            "Verify transpose algebra carefully in `<think>` and output ONLY a ```python block."
        )
        think = (
            "[L1] Check matrix dimensions: x is (S, D_in), w_base is (D_out, D_in) -> x @ w_base.T is (S, D_out). "
            "In MLX PEFT convention, lora_a is (D_in, r) and lora_b is (r, D_out), so (x @ lora_a) is (S, r) and (x @ lora_a) @ lora_b is (S, D_out).\n"
            "[L2] Equate x @ w_fused.T = x @ w_base.T + scale * x @ (lora_a @ lora_b). Taking transpose of the delta: "
            "(lora_a @ lora_b).T = lora_b.T @ lora_a.T, which has shape (D_out, r) @ (r, D_in) = (D_out, D_in), matching w_base!\n"
            "[L3] At init lora_b = 0 => w_fused == w_base (exact identity warm-start)."
        )
        test_code = (
            "import numpy as np\n"
            "rng = np.random.default_rng(7)\n"
            f"x = rng.standard_normal((5, 32)); w = rng.standard_normal((48, 32))\n"
            f"a = rng.standard_normal((32, {rank})); b = rng.standard_normal(({rank}, 48))\n"
            f"y_u, w_f = {fn_fwd}(x, w, a, b, {rank}, {float(alpha)})\n"
            "assert w_f.shape == (48, 32)\n"
            "assert np.max(np.abs(y_u - (x @ w_f.T))) < 1e-10"
        )
        return {
            "pillar": "ai_weights_code",
            "messages": [
                {"role": "system", "content": "Verify tensor shapes and transpose identities in <think> before writing code."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 2:
        # Vision-Nullspace Orthogonal Task-Vector Projection (Preserving Multimodal Subspaces During Weight Merging!)
        fn = "project_delta_into_nullspace"
        code = (
            "import numpy as np\n\n"
            f"def {fn}(delta_w: np.ndarray, vis_activations: np.ndarray, k: int) -> np.ndarray:\n"
            "    \"\"\"delta_w: (d_out, d_in); vis_activations: (n_tokens, d_in); k: protected top-k right singular directions.\"\"\"\n"
            "    if k <= 0 or vis_activations.size == 0:\n"
            "        return delta_w.copy()\n"
            "    _u, _s, vt = np.linalg.svd(vis_activations, full_matrices=False)\n"
            "    v_top = vt[:k].T  # (d_in, k) orthonormal columns\n"
            "    return delta_w - (delta_w @ v_top) @ v_top.T"
        )
        prompt = (
            f"When merging a pure-text weight delta `delta_w` `(d_out, d_in)` into a Vision-Language Model linear layer `y = x @ W.T`, "
            "naive addition shifts visual token outputs `vis_activations @ delta_w.T`. Write a NumPy function "
            f"`{fn}(delta_w: np.ndarray, vis_activations: np.ndarray, k: int) -> np.ndarray` that projects `delta_w` onto the orthogonal complement "
            "of the top-`k` principal input directions of `vis_activations` `(n_tokens, d_in)` so `vis_activations @ delta_perp.T` vanishes on that top-`k` subspace. "
            "Output ONLY a ```python block."
        )
        think = (
            "[L1] Linear layer maps row input v in R^{d_in} to v @ delta_w.T = (delta_w @ v.T).T. To leave the top-k visual input subspace V_k (shape d_in x k, orthonormal columns) untouched, we need delta_perp @ V_k = 0.\n"
            "[L2] Take thin SVD of `vis_activations` = U S V^T; the first k rows of `vt` form V_k^T (`v_top = vt[:k].T`, shape d_in x k, V_k^T V_k = I_k). "
            "Right-multiplying by projector (I - V_k V_k^T) in O(d_out * d_in * k) without forming d_in x d_in matrix: `delta_w - (delta_w @ v_top) @ v_top.T`.\n"
            "[L3] Check: `delta_perp @ v_top = delta_w @ v_top - (delta_w @ v_top) @ (v_top.T @ v_top) = 0`."
        )
        test_code = (
            "import numpy as np\n"
            "rng = np.random.default_rng(11)\n"
            "vis = rng.standard_normal((20, 16)) @ np.diag([10]*4 + [0]*12)\n"
            "dw = rng.standard_normal((24, 16))\n"
            f"dw_p = {fn}(dw, vis, 4)\n"
            "assert np.max(np.abs(vis @ dw_p.T)) < 1e-10"
        )
        return {
            "pillar": "ai_weights_code",
            "messages": [
                {"role": "system", "content": "Derive subspace projections from linear algebra first principles in <think>."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 3:
        # SwiGLU FFN Associative Key-Value Memory & Rank-1 Weight-Space Surgery
        fn = "rank1_w2_direction_suppression"
        code = (
            "import numpy as np\n\n"
            f"def {fn}(w2: np.ndarray, u_bad: np.ndarray, neuron_idx: list[int], alpha: float) -> np.ndarray:\n"
            "    \"\"\"w2: (d_model, d_ff); u_bad: (d_model,); suppress component along unit-normalized u_bad on columns neuron_idx.\"\"\"\n"
            "    norm = float(np.linalg.norm(u_bad))\n"
            "    if norm < 1e-12 or not neuron_idx:\n"
            "        return w2.copy()\n"
            "    u = u_bad / norm\n"
            "    out = w2.copy()\n"
            "    cols = out[:, neuron_idx]\n"
            "    out[:, neuron_idx] = cols - alpha * np.outer(u, u @ cols)\n"
            "    return out"
        )
        prompt = (
            "In a SwiGLU block `y = x + W2 @ (SiLU(W1 @ x) * (W3 @ x))`, each column `W2[:, j]` writes a vector into the residual stream scaled by neuron `j`'s gate activation. "
            f"Write `{fn}(w2: np.ndarray, u_bad: np.ndarray, neuron_idx: list[int], alpha: float) -> np.ndarray` that normalizes `u_bad` to unit length `u` "
            "and damps only the component along `u` on selected columns `J = neuron_idx` via `W2[:, J] <- W2[:, J] - alpha * u (u^T W2[:, J])`, "
            "leaving all directions orthogonal to `u` and all columns outside `J` bit-exact. Output ONLY a ```python block."
        )
        think = (
            "[L1] In SwiGLU key-value memory, W1/W3 compute scalar gate coefficient g_j(x), and column W2[:, j] in R^{d_model} is the value vector added to residual stream.\n"
            "[L2] For unit vector u (||u||_2 = 1) and orthogonal v (u^T v = 0): v^T (c - alpha * u (u^T c)) = v^T c - 0 = v^T c (orthogonal features preserved!), "
            "while u^T c_new = (1 - alpha) (u^T c) (at alpha=1.0, exact projection onto u^perp).\n"
            "[L3] Normalize u = u_bad / ||u_bad||, update `out[:, neuron_idx] = cols - alpha * np.outer(u, u @ cols)`."
        )
        test_code = (
            "import numpy as np\n"
            "w2 = np.array([[3.0, 5.0], [4.0, 12.0]])\n"
            "u_bad = np.array([3.0, 4.0])\n"
            f"edited = {fn}(w2, u_bad, [0], 1.0)\n"
            "assert np.allclose(edited[:, 1], w2[:, 1])\n"
            "assert abs(np.dot(u_bad, edited[:, 0])) < 1e-10"
        )
        return {
            "pillar": "ai_weights_code",
            "messages": [
                {"role": "system", "content": "Explain mechanistic FFN weight geometry in <think> and output verified NumPy code."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 4:
        # Full-Logit Vocabulary Allocation Spike vs Chunked CE Memory Math!
        seq_len = rng.choice([1024, 2048])
        vocab = 128000
        chunk = 256
        full_mib = (seq_len * vocab * 4) // (1024 * 1024)
        chunk_mib = (chunk * vocab * 4) // (1024 * 1024)
        saved_mib = full_mib - chunk_mib
        q = (
            f"During fine-tuning of a language model with `vocab_size = {vocab}` on sequence length `L = {seq_len}`, "
            f"materializing full-sequence `float32` (4 bytes) logits `[1, {seq_len}, {vocab}]` causes a large memory spike. "
            f"Compute exact integer MiB (`1 MiB = 1048576 bytes`) for: (1) `full_logits_mib`, (2) `chunked_logits_mib` when computing cross-entropy "
            f"in slices of `chunk_size = {chunk}` tokens, and (3) `saved_mib = full_logits_mib - chunked_logits_mib`. "
            'Output raw JSON: {"full_logits_mib": <int>, "chunked_logits_mib": <int>, "saved_mib": <int>}.'
        )
        # Note: 128000 * 4 = 512000 bytes; 1024 * 512000 / 1048576 = 500 MiB exact! 256 * 512000 / 1048576 = 125 MiB exact!
        think = (
            f"[L1] 1 MiB = 1024 * 1024 = 1,048,576 bytes. Per-token float32 logit row = {vocab} * 4 = 512,000 bytes = 500 * 1024 bytes.\n"
            f"[L2] For 256 tokens: 256 * 500 * 1024 bytes = 128,000 * 1024 bytes = (128,000 / 1024) MiB = 125 MiB! "
            f"Since L={seq_len} is {seq_len // 256} * 256: full_logits_mib = {seq_len // 256} * 125 = {full_mib} MiB; "
            f"chunked_logits_mib = {chunk_mib} MiB; saved_mib = {full_mib} - {chunk_mib} = {saved_mib} MiB.\n"
            f"[L3] Verify: {full_mib} - {chunk_mib} = {saved_mib} MiB (and backward pass saves another 2x on logit grad buffers)."
        )
        exp = {"full_logits_mib": full_mib, "chunked_logits_mib": chunk_mib, "saved_mib": saved_mib}
        return {
            "pillar": "ai_weights_mechanics",
            "messages": [
                {"role": "system", "content": "Compute exact GPU tensor memory allocations in <think> and output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 5:
        # LoRA Parameter Count & Compression Ratio Exact Calculation
        d_model = 2048
        d_ff = 7680
        rank = rng.choice([16, 32])
        # SwiGLU w1, w2, w3 base params vs LoRA params
        base_ffn = 3 * d_model * d_ff
        lora_ffn = 3 * rank * (d_model + d_ff)
        q = (
            f"Consider a single SwiGLU FFN block with `d_model = {d_model}` and `d_ff = {d_ff}` containing three bias-free matrices `w1, w2, w3`. "
            f"Compute: (1) `base_params` across all three matrices, and (2) `lora_params` when attaching rank `r = {rank}` LoRA (`A` and `B` per matrix) to `w1, w2, w3`. "
            'Output raw JSON: {"base_params": <int>, "lora_params": <int>}.'
        )
        dm_dff = d_model + d_ff  # 9728
        think = (
            f"[L1] Each of w1 ({d_ff}x{d_model}), w3 ({d_ff}x{d_model}), and w2 ({d_model}x{d_ff}) has {d_model}*{d_ff} weights, and LoRA adds r*(d_model + d_ff) weights per matrix.\n"
            f"[L2] Per-matrix base: 2048 * 7680 = 2048 * 768 * 10 = 1,572,864 * 10 = 15,728,640. Times 3 = 47,185,920 (`base_params`). "
            f"Sum dims: {d_model} + {d_ff} = {dm_dff}. Per-matrix LoRA (r={rank}): {rank} * {dm_dff} = {rank * dm_dff}. "
            f"Times 3 matrices: 3 * {rank * dm_dff} = {lora_ffn} (`lora_params`).\n"
            f"[L3] Verify ratio: {lora_ffn} is <2% of {base_ffn}."
        )
        exp = {"base_params": base_ffn, "lora_params": lora_ffn}
        return {
            "pillar": "ai_weights_mechanics",
            "messages": [
                {"role": "system", "content": "Calculate exact neural parameter counts step-by-step in <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 6:
        # Why Hybrid Short-Conv + Attention Models Need Local Reasoning Witnesses (Architectural First-Principles)
        conv_k = rng.choice([5, 7])
        n_conv, n_slide, n_full = 13, 5, 6
        q = (
            f"A 24-layer tri-hybrid language model consists of `{n_conv}` causal short-convolution layers (`kernel_size={conv_k}`), "
            f"`{n_slide}` sliding-window attention layers, and `{n_full}` global full-attention layers. "
            "When evaluating 6 candidate options listed 400 tokens earlier in the prompt, jumping straight to `OPT-D: PASS` in `<think>` "
            "causes high hallucination rates, whereas writing `OPT-D: cost=270<=300, lat=48<=55 -> PASS` boosts accuracy from ~50% to >95%. "
            "Output raw JSON summarizing the architectural mechanism: "
            '`{"global_retrieval_layers": <int>, "local_compute_layers": <int>, "requires_local_witness": true}`.'
        )
        local_l = n_conv + n_slide
        think = (
            f"[L1] Only the {n_full} `full_attention` layers can attend directly across a 400-token distance to retrieve option attributes from the prompt; "
            f"the {n_conv} causal conv layers (`kernel_size={conv_k}`) only see the last {conv_k} tokens, and the {n_slide} sliding layers have bounded local receptivity ({n_conv}+{n_slide}={local_l} local layers).\n"
            f"[L2] If the model emits `OPT-D: PASS` directly, a single forward pass must retrieve 5 attributes AND compute 5 inequalities purely inside {n_full} global layers. "
            f"By first copying `cost=270<=300, lat=48<=55` into `<think>`, the {n_full} global layers perform simple retrieval, placing `270<=300` inside the {conv_k}-token causal window so all {local_l} local conv/sliding layers execute the inequality comparison!\n"
            f"[L3] global_retrieval_layers={n_full}, local_compute_layers={local_l}, requires_local_witness=true."
        )
        exp = {"global_retrieval_layers": n_full, "local_compute_layers": local_l, "requires_local_witness": True}
        return {
            "pillar": "ai_weights_mechanics",
            "messages": [
                {"role": "system", "content": "Analyze model behavior from layer-level inductive biases in <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    # case == 7: TIES-Merging Executable Implementation (Trim, Elect Sign, Disjoint Mean)
    fn = "ties_merge_deltas"
    code = (
        "import numpy as np\n\n"
        f"def {fn}(deltas: list[np.ndarray], keep_ratio: float = 0.5) -> np.ndarray:\n"
        "    if not deltas:\n"
        "        raise ValueError('deltas must be non-empty')\n"
        "    trimmed = []\n"
        "    for d in deltas:\n"
        "        flat = np.abs(d).ravel()\n"
        "        k = max(1, int(round(len(flat) * keep_ratio)))\n"
        "        thresh = np.partition(flat, len(flat) - k)[len(flat) - k]\n"
        "        trimmed.append(np.where(np.abs(d) >= thresh, d, 0.0))\n"
        "    stack = np.stack(trimmed, axis=0)\n"
        "    elected_sign = np.sign(np.sum(stack, axis=0))\n"
        "    agree = (np.sign(stack) == elected_sign) & (elected_sign != 0)\n"
        "    counts = np.sum(agree, axis=0)\n"
        "    summed = np.sum(np.where(agree, stack, 0.0), axis=0)\n"
        "    return np.where(counts > 0, summed / np.maximum(counts, 1), 0.0)"
    )
    prompt = (
        f"Implement `{fn}(deltas: list[np.ndarray], keep_ratio: float = 0.5) -> np.ndarray` for TIES model merging: "
        "(1) **Trim**: in each task vector `d`, zero out entries below the top-`keep_ratio` magnitude threshold (`k = max(1, round(N * keep_ratio))`); "
        "(2) **Elect Sign**: compute `np.sign(sum_t d_trimmed_t)` elementwise; "
        "(3) **Disjoint Merge**: average ONLY non-zero trimmed entries whose sign matches `elected_sign` (`0.0` where `elected_sign == 0`). "
        "Output ONLY a ```python block."
    )
    think = (
        "[L1] Naive averaging of multiple fine-tuned weight deltas suffers from (a) interference from noisy small weights and (b) sign conflict cancellation (+0.4 + -0.3 -> +0.05).\n"
        "[L2] Step 1: find top-k magnitude cutoff via `np.partition(flat, N - k)[N - k]`. Step 2: `elected_sign = np.sign(np.sum(stack, axis=0))` resolves sign by total mass. "
        "Step 3: mask `agree = (np.sign(stack) == elected_sign) & (elected_sign != 0)` and divide `summed` by `counts` (not len(deltas)!).\n"
        "[L3] Guard division by zero via `np.maximum(counts, 1)` inside `np.where`."
    )
    test_code = (
        "import numpy as np\n"
        "d1 = np.array([10.0, -8.0, 0.1, 0.05])\n"
        "d2 = np.array([6.0, 2.0, -0.2, 9.0])\n"
        f"m = {fn}([d1, d2], keep_ratio=0.5)\n"
        "assert np.isclose(m[0], 8.0) and np.isclose(m[1], -8.0) and np.isclose(m[3], 9.0) and m[2] == 0.0"
    )
    return {
        "pillar": "ai_weights_code",
        "messages": [
            {"role": "system", "content": "Derive model-merging algorithms cleanly in <think> and output verified NumPy code."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
        ],
        "verifier": {"kind": "python_code", "test": test_code},
    }


# ===========================================================================
# PILLAR C: AI SELF-IMPROVEMENT, POST-TRAINING DIAGNOSTICS & META-RPR
# ===========================================================================
def make_self_improvement_meta_item(rng: random.Random, idx: int) -> dict[str, Any]:
    """Teach the model how AI systems diagnose training/reasoning failures and systematically improve themselves."""
    case = idx % 8

    if case == 0:
        # Diagnosing Silent Zero-Gradient Checkpointing Closure Bug (Real MLX Autograd Pathology!)
        base_vl = round(rng.uniform(1.65, 2.15), 4)
        q = (
            f"Post-Training Diagnostic: You launch a 153-module LoRA fine-tuning run (`lora_a ~ N(0, sigma^2)`, `lora_b = 0`). "
            f"Every 40 steps, validation loss logs `{base_vl:.4f}` identically to 4 decimal places, and inspecting the saved checkpoint "
            "shows `153/153` `lora_a` tensors non-zero while `0/153` `lora_b` tensors are non-zero (`max_abs = 0.0`), even though `train_loss` printed `0.62` "
            "and `DecoderLayer.__call__` wrapped the layer in `mx.checkpoint(lambda x: orig_call(self, x))`. "
            "Identify the exact mathematical/autograd root cause and best fail-fast assertion as raw JSON: "
            '`{"root_cause": "<closure_param_grad_dropped|lr_too_small|b_init_bug>", "step0_grad_a_is_zero": true, "step1_assert": "max_abs_lora_b > 0"}`.'
        )
        think = (
            f"[L1] Forward delta is `(alpha/r) * (x @ A) @ B`. At step 0, `B = 0`, so `dL/dA = (alpha/r) * x^T @ (grad_y @ B^T) = 0` (`step0_grad_a_is_zero = true`), "
            f"WHEREAS `dL/dB = (alpha/r) * (x @ A)^T @ grad_y` MUST be non-zero on step 1 if gradients reach the layer!\n"
            f"[L2] Why did `lora_b` stay identically `0.0` while `lora_a` only looked non-zero from its random Gaussian initialization? "
            f"Because functional `mx.checkpoint(lambda x: orig_call(self, x))` only differentiates wrt explicit positional inputs (`x`), severing autograd edges to closure-captured `self` module parameters!\n"
            f"[REBUILD] Fix by passing explicit trainable parameter trees into `checkpoint` (or removing redundant layer checkpointing when chunked CE already fits RAM), "
            "and assert `max_abs_lora_b > 0` immediately after optimizer step 1."
        )
        exp = {
            "root_cause": "closure_param_grad_dropped",
            "step0_grad_a_is_zero": True,
            "step1_assert": "max_abs_lora_b > 0",
        }
        return {
            "pillar": "ai_self_improvement",
            "messages": [
                {"role": "system", "content": "Diagnose ML training pathologies from gradient calculus and autograd semantics inside <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 1:
        # Executable GDPO + Dr. GRPO Multi-Channel Advantage Estimator!
        fn = "compute_gdpo_dr_grpo_advantages"
        code = (
            f"def {fn}(rollout_channels: list[dict[str, float]], weights: dict[str, float]) -> list[float]:\n"
            "    \"\"\"Compute decoupled per-channel mean-centered advantages WITHOUT std division (Dr. GRPO + GDPO).\"\"\"\n"
            "    g = len(rollout_channels)\n"
            "    if g == 0:\n"
            "        return []\n"
            "    advs = [0.0] * g\n"
            "    for ch, w in weights.items():\n"
            "        vals = [float(r.get(ch, 0.0)) for r in rollout_channels]\n"
            "        mean_v = sum(vals) / g\n"
            "        for i, v in enumerate(vals):\n"
            "            advs[i] += w * (v - mean_v)\n"
            "    return advs"
        )
        prompt = (
            "Vanilla GRPO normalizes scalar summed rewards by group standard deviation `(R_i - mean(R)) / (std(R) + eps)`, "
            "which explodes small formatting noise when correctness ties across a group and conflates orthogonal reward channels. "
            f"Implement `{fn}(rollout_channels: list[dict[str, float]], weights: dict[str, float]) -> list[float]` "
            "that mean-centers each reward channel independently across the group (GDPO) without dividing by standard deviation (Dr. GRPO), "
            "returning the weighted sum of centered channel advantages per rollout. Output ONLY a ```python block."
        )
        think = (
            "[L1] Two biases in vanilla scalar GRPO: (1) dividing by `std(R)` magnifies 0.01 length/echo differences into +/-1.0 advantages when all G rollouts get `correct=1`; "
            "(2) summing channels before centering lets high-variance penalties drown low-variance format signals.\n"
            "[L2] Decouple per channel `c`: `A_i = sum_c w_c * (r_{i,c} - (1/G) * sum_j r_{j,c})`. Notice `sum_i A_i == 0` is guaranteed and constant channels contribute `0.0` without `0/0` instability!\n"
            "[L3] Handle empty rollout list cleanly."
        )
        test_code = (
            "chs = [{'correct': 1.0, 'concise': 0.4}, {'correct': 1.0, 'concise': 0.0}, {'correct': 0.0, 'concise': 0.2}, {'correct': 0.0, 'concise': 0.2}]\n"
            "w = {'correct': 1.0, 'concise': 0.5}\n"
            f"adv = {fn}(chs, w)\n"
            "assert abs(sum(adv)) < 1e-9\n"
            "assert abs(adv[0] - (0.5 + 0.5 * 0.2)) < 1e-9 and abs(adv[1] - (0.5 + 0.5 * (-0.2))) < 1e-9"
        )
        return {
            "pillar": "ai_self_improvement_code",
            "messages": [
                {"role": "system", "content": "Analyze RLVR advantage estimators in <think> and output verified Python code."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 2:
        # Executable ATMS Taint Ratio & Selective vs Full Option-B Pyramid Repair Planner!
        fn = "plan_rpr_reconstruction"
        code = (
            f"def {fn}(active_nodes: dict[str, list[str]], facts: dict[str, list[str]], invalidated_roots: set[str], threshold: float = 0.65) -> dict:\n"
            "    \"\"\"active_nodes[nid] = parent IDs; facts[fid] = parent IDs.\n"
            "    Return {'mode': 'selective_repair'|'option_b_rebuild', 'tainted_nodes': sorted list, 'preserved_facts': sorted list}.\"\"\"\n"
            "    bad = set(invalidated_roots)\n"
            "    tainted: set[str] = set()\n"
            "    changed = True\n"
            "    while changed:\n"
            "        changed = False\n"
            "        for nid, deps in active_nodes.items():\n"
            "            if nid not in tainted and any(d in bad for d in deps):\n"
            "                tainted.add(nid)\n"
            "                bad.add(nid)\n"
            "                changed = True\n"
            "    preserved = sorted(fid for fid, deps in facts.items() if not any(d in bad for d in deps))\n"
            "    r_taint = len(tainted) / max(1, len(active_nodes))\n"
            "    mode = 'selective_repair' if (active_nodes and r_taint <= threshold) else 'option_b_rebuild'\n"
            "    return {'mode': mode, 'tainted_nodes': sorted(tainted), 'preserved_facts': preserved}"
        )
        prompt = (
            f"Implement `{fn}(active_nodes: dict[str, list[str]], facts: dict[str, list[str]], invalidated_roots: set[str], threshold: float = 0.65) -> dict` "
            "for a Revisable Reasoning Pyramid: propagate transitive taint from `invalidated_roots` through `active_nodes`, preserve any fact in `facts` "
            "whose `deps` do not intersect `invalidated_roots | tainted_nodes`, and choose `'selective_repair'` when `active_nodes` is non-empty and "
            "`len(tainted_nodes) / len(active_nodes) <= threshold` else `'option_b_rebuild'`. Output ONLY a ```python block."
        )
        think = (
            "[L1] ATMS belief revision separates (a) invalidated premise roots + transitively dependent reasoning nodes (`bad`) from (b) independent verified facts (`deps` disjoint from `bad`).\n"
            "[L2] Compute transitive fixed point of `tainted` over `active_nodes`, adding newly tainted `nid` to `bad` so downstream nodes & dependent facts are caught. "
            "Then filter `facts` with `not any(d in bad for d in deps)` and compare `r_taint <= threshold`.\n"
            "[L3] Return sorted lists for determinism."
        )
        test_code = (
            "nodes = {'L2-1': ['A1'], 'L2-2': ['A2'], 'L2-3': ['A3'], 'L3-1': ['L2-1']}\n"
            "facts = {'F1': [], 'F2': ['A2'], 'F3': ['L2-1']}\n"
            f"r_sel = {fn}(nodes, facts, {{'A1'}}, threshold=0.65)\n"
            "assert r_sel == {'mode': 'selective_repair', 'tainted_nodes': ['L2-1', 'L3-1'], 'preserved_facts': ['F1', 'F2']}\n"
            f"r_full = {fn}(nodes, facts, {{'A1', 'A2'}}, threshold=0.65)\n"
            "assert r_full['mode'] == 'option_b_rebuild' and r_full['preserved_facts'] == ['F1']"
        )
        return {
            "pillar": "ai_self_improvement_code",
            "messages": [
                {"role": "system", "content": "Implement clean ATMS self-correction algorithms after verifying invariants in <think>."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 3:
        # Benchmark Decontamination & Prompt-Echo Anti-Gaming Guard (Executable!)
        fn = "audit_trace_quality"
        code = (
            f"def {fn}(prompt: str, think_text: str, bench_corpora: list[str]) -> dict:\n"
            "    def ngrams(txt: str, n: int) -> list[tuple[str, ...]]:\n"
            "        w = txt.lower().split()\n"
            "        return [tuple(w[i:i+n]) for i in range(max(0, len(w) - n + 1))]\n"
            "    p5 = set(ngrams(prompt, 5))\n"
            "    t5 = ngrams(think_text, 5)\n"
            "    echo_rate = (sum(1 for g in t5 if g in p5) / len(t5)) if t5 else 0.0\n"
            "    bench6 = {g for doc in bench_corpora for g in ngrams(doc, 6)}\n"
            "    combined6 = ngrams(prompt + ' ' + think_text, 6)\n"
            "    leaked = any(g in bench6 for g in combined6)\n"
            "    return {'echo_rate': round(echo_rate, 4), 'contaminated': leaked, 'high_quality': (not leaked and echo_rate <= 0.2)}"
        )
        prompt = (
            f"To prevent small reasoning models from faking progress via benchmark memorization or prompt-copying inside `<think>`, "
            f"write `{fn}(prompt: str, think_text: str, bench_corpora: list[str]) -> dict` returning:\n"
            "- `'echo_rate'`: rounded (4 decimals) fraction of lowercase whitespace word 5-grams in `think_text` that appear in `prompt` (`0.0` if `<5` words),\n"
            "- `'contaminated'`: `True` iff any 6-gram of `prompt + ' ' + think_text` appears in any string in `bench_corpora`,\n"
            "- `'high_quality'`: `not contaminated and echo_rate <= 0.2`.\n"
            "Output ONLY a ```python block."
        )
        think = (
            "[L1] Two failure modes in synthetic reasoning post-training: (1) prompt echoing (copying prompt sentences verbatim into `<think>` wastes token budget without computing anything); "
            "(2) benchmark n-gram leakage.\n"
            "[L2] Helper `ngrams(txt, n)` over `txt.lower().split()`. Compute `echo_rate = hits / len(t5)` when `t5` non-empty, and `contaminated` via set membership against 6-grams of `bench_corpora`.\n"
            "[L3] Return exact keys `'echo_rate'`, `'contaminated'`, `'high_quality'`."
        )
        test_code = (
            f"r1 = {fn}('a b c d e f g', 'x y z u v w', ['p q r s t u v'])\n"
            "assert r1 == {'echo_rate': 0.0, 'contaminated': False, 'high_quality': True}\n"
            f"r2 = {fn}('a b c d e f', 'a b c d e f', [])\n"
            "assert r2['echo_rate'] == 1.0 and r2['high_quality'] is False\n"
            f"r3 = {fn}('alpha beta gamma delta epsilon zeta', 'clean step', ['alpha beta gamma delta epsilon zeta'])\n"
            "assert r3['contaminated'] is True and r3['high_quality'] is False"
        )
        return {
            "pillar": "ai_self_improvement_code",
            "messages": [
                {"role": "system", "content": "Build rigorous evaluation & data-hygiene verifiers in Python after checking edge cases in <think>."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
            ],
            "verifier": {"kind": "python_code", "test": test_code},
        }

    if case == 4:
        # WiSE-FT / EMA Weight-Space Interpolation & Why Pure RLVR Forgets Without SFT Anchors
        q = (
            "During post-training of a 2B model, Stage-1 DFT reaches `80%` overall (`60%` math, `100%` schema), "
            "but running 24 unshuffled Stage-2 RLVR episodes ordered `[math x6, audit x6, tool x6, schema x6]` with no SFT replay "
            "causes `math` to drop from `60%` to `0%`. Identify all three necessary fixes in raw JSON: "
            '`{"shuffle_episodes": true, "mix_sft_anchor_loss": true, "wise_ft_checkpoint_interp": true}`.'
        )
        think = (
            "[L1] Sequential single-domain RL updates perform greedy policy gradient steps on the final domain (`schema`), shifting shared MLP/attention subspaces away from earlier domains (`math`).\n"
            "[L2] Three orthogonal stabilizers eliminate this forgetting: (1) `shuffle_episodes=true` so consecutive updates mix domains; "
            "(2) `mix_sft_anchor_loss=true` (`L = L_RLVR + beta * L_DFT`) so live rollouts stay anchored to verified multi-domain trajectories; "
            "(3) `wise_ft_checkpoint_interp=true` (`theta_final = (1 - lam)*theta_stage1_ema + lam*theta_stage2`) selected on a held-out multi-skill gate.\n"
            "[L3] Emit JSON."
        )
        exp = {"shuffle_episodes": True, "mix_sft_anchor_loss": True, "wise_ft_checkpoint_interp": True}
        return {
            "pillar": "ai_self_improvement",
            "messages": [
                {"role": "system", "content": "Diagnose continual-learning and RLVR forgetting mechanisms from optimization principles in <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 5:
        # SCoRe Multi-Turn Progress Shaping & Train/Test <think> Stripping Alignment
        r1_val = 0.0
        r2_val = 1.0
        alpha_p = rng.choice([1.5, 2.0, 2.5])
        shaped_fix = round(r2_val + alpha_p * (r2_val - r1_val), 2)
        shaped_regress = round(0.0 + alpha_p * (0.0 - 1.0), 2)
        q = (
            f"In two-turn belief-revision RL (SCoRe), Turn-2 reward uses progress shaping `r_hat_2 = r(y_2) + {alpha_p} * (r(y_2) - r(y_1))` "
            "where `r(y) in {0.0, 1.0}`. Compute: (1) `reward_when_fixed` (`r(y_1)=0.0 -> r(y_2)=1.0`), "
            "(2) `reward_when_broken` (`r(y_1)=1.0 -> r(y_2)=0.0` unnecessary collateral change!), and "
            "(3) `strip_prior_turn_think` (whether Turn-1 `<think>` must be stripped from Turn-2 context during training when inference chat templates strip prior `<think>`). "
            'Output raw JSON: {"reward_when_fixed": <number>, "reward_when_broken": <number>, "strip_prior_turn_think": true}.'
        )
        think = (
            f"[L1] Progress shaping rewards genuine error correction while heavily penalizing unnecessary edits to already-correct answers.\n"
            f"[L2] (1) Fixed (0 -> 1): r_hat_2 = 1.0 + {alpha_p}*(1.0 - 0.0) = {shaped_fix}. "
            f"(2) Broken (1 -> 0): r_hat_2 = 0.0 + {alpha_p}*(0.0 - 1.0) = {shaped_regress}. "
            f"(3) Train/test distribution alignment: if inference strips Turn-1 `<think>` (leaving only Turn-1 JSON), leaving Turn-1 `<think>` visible during training leaks reasoning state unavailable at test time -> `strip_prior_turn_think = true`.\n"
            f"[L3] Emit JSON."
        )
        exp = {"reward_when_fixed": shaped_fix, "reward_when_broken": shaped_regress, "strip_prior_turn_think": True}
        return {
            "pillar": "ai_self_improvement",
            "messages": [
                {"role": "system", "content": "Analyze multi-turn RL credit assignment and train-inference symmetry in <think>. Output raw JSON."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    if case == 6:
        # 2-Turn Self-Critique & Internal RPR Repair on a Flawed Draft Derivation!
        n_items = rng.choice([12, 15, 18])
        price = rng.choice([25, 30, 40])
        disc_pct = 20
        tax_pct = 10
        gross = n_items * price
        after_disc = (gross * (100 - disc_pct)) // 100
        exact_final = (after_disc * (100 + tax_pct)) // 100
        wrong_additive = (gross * (100 - disc_pct + tax_pct)) // 100  # classic -20% + 10% = -10% trap!

        u1 = (
            f"Audit and repair this flawed draft solution:\n"
            f"Problem: Buy `{n_items}` modules at `${price}` each (`${gross}` subtotal), apply a `{disc_pct}%` bulk discount first, "
            f"and then apply `{tax_pct}%` VAT on the discounted invoice.\n"
            f"Flawed Draft: 'A {disc_pct}% discount followed by {tax_pct}% tax nets -{disc_pct}% + {tax_pct}% = -10%, so total = {gross} * 0.90 = {wrong_additive}.'\n"
            'Pinpoint the mathematical flaw and return raw JSON: {"draft_error": "percentages_multiply_not_add", "correct_total": <int>}.'
        )
        think = (
            f"[AUDIT] Draft assumed sequential percentage operators add (`-0.20 + 0.10 = -0.10`), which is INVALID! "
            f"Multiplying factors gives `(1 - {disc_pct}/100) * (1 + {tax_pct}/100) = 0.80 * 1.10 = 0.88` (a 12% net reduction, NOT 10%!), "
            f"while subtotal F1 = {n_items}*{price} = {gross} is PRESERVED.\n"
            f"[REBUILD] Step 1 (after {disc_pct}% discount): {gross} * 80 / 100 = {after_disc}. "
            f"Step 2 (after {tax_pct}% VAT on {after_disc}): {after_disc} + {after_disc // 10} = {exact_final} (vs flawed {wrong_additive})."
        )
        exp = {"draft_error": "percentages_multiply_not_add", "correct_total": exact_final}
        return {
            "pillar": "ai_self_improvement",
            "messages": [
                {"role": "system", "content": "Critique flawed reasoning traces using [AUDIT] and [REBUILD] inside <think>, preserving valid intermediate steps. Output raw JSON."},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": f"<think>{think}</think>\n{json.dumps(exp)}"},
            ],
            "expected": exp,
        }

    # case == 7: Verifier False-Positive Audit (Substring Reward Bug Fix in Python!)
    fn = "has_real_tool_invocation"
    code = (
        "import re\n\n"
        "_REAL_TAG = re.compile(r'<function=|<parameter=|</tool_call>|<tool_call>\\s*(?:<|\\{|\\n)', re.IGNORECASE)\n\n"
        f"def {fn}(raw_response: str) -> bool:\n"
        "    low = raw_response.lower()\n"
        "    if '</think>' in low:\n"
        "        post = raw_response.split('</think>', 1)[1]\n"
        "        if '<tool_call' in post.lower():\n"
        "            return True\n"
        "    return bool(_REAL_TAG.search(raw_response))"
    )
    prompt = (
        "During RLVR training for tool abstention, a naive verifier `'<tool_call' in raw_response.lower()` falsely penalized "
        "a model that reasoned `<think>No external tool needed; never emit <tool_call>.</think>\\n{\"total\": 42}`, corrupting policy gradients. "
        f"Write `{fn}(raw_response: str) -> bool` that returns `False` on negative prose mentions of `<tool_call>` inside closed `<think>...</think>`, "
        "while returning `True` whenever `<tool_call` appears after `</think>` OR a real structured invocation (`<function=`, `<parameter=`, `</tool_call>`, or `<tool_call>` followed by `<`, `{`, or newline) appears anywhere. "
        "Output ONLY a ```python block."
    )
    think = (
        "[L1] Distinguish meta-linguistic mention inside `<think>` (`never emit <tool_call>.`) from actual XML/JSON invocation.\n"
        "[L2] Split on `</think>` when present: any `<tool_call` in the post-think surface is a real leak; anywhere in `raw_response`, "
        "`<function=`, `<parameter=`, `</tool_call>`, or `<tool_call>\\s*(?:<|\\{|\\n)` indicates an actual tool payload.\n"
        "[L3] Test both false-positive self-reminders and unclosed-think leaks."
    )
    test_code = (
        f"assert {fn}('<think>Do not emit <tool_call>.</think>\\n{{\"ok\": true}}') is False\n"
        f"assert {fn}('<think>Call tool</think>\\n<tool_call><function=f></function></tool_call>') is True\n"
        f"assert {fn}('<think><tool_call>\\n<function=leak>') is True"
    )
    return {
        "pillar": "ai_self_improvement_code",
        "messages": [
            {"role": "system", "content": "Design reward verifiers that are immune to false positives and reward hacking. Output verified Python code."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>{think}</think>\n```python\n{code}\n```"},
        ],
        "verifier": {"kind": "python_code", "test": test_code},
    }
