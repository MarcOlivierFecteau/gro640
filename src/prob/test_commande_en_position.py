#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from gro640_robots import LaserRobot
from dosg0801_fecm0701 import CustomPositionController  # Empty template


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Test position controller for LaserRobot."
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        default=[0.0, -0.5],
        help="Target position for the end-effector (2 values)",
    )
    parsed_args = parser.parse_args(args)

    target = np.array(parsed_args.target)
    if np.shape(target) != (2,):
        raise ValueError("Target position MUST be a 2 x 1 matrix.")

    # Model cinématique du robot
    sys = LaserRobot()

    # Contrôleur en position de l'effecteur standard
    ctl = CustomPositionController(sys)

    # Cible de position pour l'effecteur
    ctl.rbar = target

    # Dynamique en boucle fermée
    clsys = ctl + sys

    # Configurations de départs
    clsys.x0 = np.array([0, 0.5, 0])  # crash
    # clsys.x0 = np.array([0, 0.7, 0])  # fonctionne

    # Simulation
    clsys.compute_trajectory()
    clsys.plot_trajectory("xu")
    clsys.animate_simulation()


if __name__ == "__main__":
    main()
