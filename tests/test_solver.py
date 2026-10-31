"""求解器一致性回归测试（python -m tests.test_solver）。

检查：对角价值为 0、反对称 V(qa,qp) = -V(qp,qa)、可剥削性 gap 足够小、
以及 BR 对 turtle 能有较高价值。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from courage.bots.scripts import bots_by_name
from courage.solvers.best_response import solve_best_response
from courage.solvers.equilibrium import build_tables, solve_equilibrium


def main() -> None:
    T = build_tables(0)
    assert len(T.states) == 36, len(T.states)

    res = solve_equilibrium(mode=0, gamma=0.95, iters=2000, tol=1e-3)
    V = res.values

    for q in range(6):
        assert abs(V[(q, q)]) < 1e-2, (q, V[(q, q)])

    antisym = max(abs(V[(a, b)] + V[(b, a)])
                  for a in range(6) for b in range(6))
    assert antisym < 5e-2, antisym
    assert res.meta["gap"] < 5e-2, res.meta["gap"]
    print(f"均衡 OK: gap={res.meta['gap']:.3f}  反对称误差={antisym:.3f}  "
          f"V(3,3)={V[(3, 3)]:+.4f}")

    opp = {(qa, qp): bots_by_name["turtle"].policy(qp, qa, mode=0)
           for qa in range(6) for qp in range(6)}
    br = solve_best_response(0, opp, gamma=0.95)
    assert br.values[(3, 3)] > 0.5, br.values[(3, 3)]
    print(f"BR(turtle) OK: V(3,3)={br.values[(3, 3)]:+.3f}")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
