#!/usr/bin/env python3

import numpy as np
import sympy


def inv_kin(r, l1, l2) -> np.ndarray:
    q2 = None
    c2 = ((r[0] ** 2 + r[1] ** 2) - (l1**2 + l2**2)) / (2 * l1 * l2)
    if c2 > 1:
        raise ValueError("Target is unreachable.")
    elif c2 == 1:
        q2 = 0
    else:
        q2 = np.arccos(c2)
    q1 = np.atan2(r[1], r[0]) + np.atan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2], dtype=np.float64)


if __name__ == "__main__":
    L1, L2 = sympy.symbols("L1, L2", positive=True)
    q1, q2 = sympy.symbols("q1, q2")

    c1 = sympy.cos(q1)
    c2 = sympy.cos(q2)
    c12 = sympy.cos(q1 + q2)
    s1 = sympy.sin(q1)
    s2 = sympy.sin(q2)
    s12 = sympy.sin(q1 + q2)

    J11 = L1 * c1 + L2 * c12
    J12 = L2 * c12
    J21 = -(L1 * s1 + L2 * s12)
    J22 = -L2 * s12

    J = sympy.Matrix([[J11, J12], [J21, J22]])

    print("J(q): ")
    sympy.pretty_print(J)
    print("\n")

    det_J = -L1 * L2 * s2
    print("|J(q)|:")
    sympy.pretty_print(det_J)
    print("\n")


    J_inv = J.inv()
    J_inv = sympy.simplify(J_inv)
    print("Inverse of J(q):")
    sympy.pretty_print(J_inv)
    print("\n")

    J_pinv = J.pinv()
    J_pinv = sympy.simplify(J_pinv)
    print("Pseudo-inverse of J(q):")
    sympy.pretty_print(J_pinv)
    print("\n")

    dq1, dq2 = sympy.symbols("dq1, dq2")
    dq = sympy.Matrix([dq1, dq2])

    r_d = np.array([[5, 0], [4, 0], [3, 0], [0, 3], [0, 4], [0, 5]])
    qbar = np.zeros(np.shape(r_d), dtype=np.float64)
    for i in range(6):
        qbar[i, :] = inv_kin(r_d[i], 4, 3)
    print("Exercice 2.1.a: Generate joint trajectories")
    print(f"joint trajectory (rad):\n{qbar}")
