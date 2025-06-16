#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pyro.control import robotcontrollers
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################


def dh2T(r: float, d: float, theta: float, alpha: float) -> np.ndarray:
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1

    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """

    T = np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha),
                np.sin(theta) * np.sin(alpha),
                r * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha),
                -np.cos(theta) * np.sin(alpha),
                r * np.sin(theta),
            ],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    return T


def dhs2T(r: np.ndarray, d: np.ndarray, theta: np.ndarray, alpha: np.ndarray):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1

    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """

    WTT = np.eye(4, dtype=np.float64)

    n = r.shape[0]

    for i in range(n):
        WTT = WTT @ dh2T(r[i], d[i], theta[i], alpha[i])

    return WTT


def f(q):
    """


    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1
        Effector's cartesian position (x,y,z)

    """
    __r = np.zeros((3, 1))

    r = np.array([0.033, 0.155, 0.135, 0, 0, 0], dtype=np.float64)
    d = np.array([0.075, 0, 0, 0, 0.218, 0.0095 + q[5]], dtype=np.float64)
    theta = np.array([q[0], q[1]+np.pi/2, q[2], q[3]-np.pi/2, q[4], np.pi/2], dtype=np.float64)
    alpha = np.array([np.pi / 2, 0, 0, -np.pi / 2, np.pi/2, 0], dtype=np.float64)
   
    T = dhs2T(r, d, theta, alpha)

    # Position de l'effecteur (x, y, z)
    __r = T[0:3, 3].reshape((3, 1))

    return __r


###################
# Part 2
###################


class CustomPositionController(EndEffectorKinematicController):
    ############################
    def __init__(self, manipulator):
        """ """

        EndEffectorKinematicController.__init__(self, manipulator, 1)

        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################
        self.L1 = 1.2  # m
        self.L2 = 0.5  # m
        self.L3 = 0.5  # m
        self.gains = np.diag([1.0, 1.0])
        self.dq_max = np.pi  # Joint speed limits (rad/s). NOTE: optional.
        self.lambda_dls = 0.1  # Damping factor

    def fwd_kin(self, q: np.ndarray) -> np.ndarray:
        """Computes the forward kinematics of the planar robot arm.

        Args:
            q (np.ndarray): current joint configuration (rad)

        Raises:
            ValueError: if `q` is not of shape 3 x 1

        Returns:
            np.ndarray: the end-effector's pose [x, y], where x, y are in meters.
        """
        if q.shape != (3,):
            raise ValueError("Joint angles `q` MUST be a 1-D array of shape 3 x 1.")
        x = (
            self.L1 * np.cos(q[0])
            + self.L2 * np.cos(q[0] + q[1])
            + self.L3 * np.cos(q[0] + q[1] + q[2])
        )
        y = (
            self.L1 * np.sin(q[0])
            + self.L2 * np.sin(q[0] + q[1])
            + self.L3 * np.sin(q[0] + q[1] + q[2])
        )

        return np.array([x, y], dtype=np.float64)

    def J(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian matrix of the current configuration of the planar robot arm.

        Args:
            q (np.ndarray): current joint configuration (rad)

        Returns:
            np.ndarray: the Jacobian matrix `J`
        """
        # Pre-compute sines and cosines for convenience
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        s123, c123 = np.sin(q[0] + q[1] + q[2]), np.cos(q[0] + q[1] + q[2])

        # Partial derivatives for x
        dx_dq1 = -self.L1 * s1 - self.L2 * s12 - self.L3 * s123
        dx_dq2 = -self.L2 * s12 - self.L3 * s123
        dx_dq3 = -self.L3 * s123

        # Partial derivatives for y
        dy_dq1 = self.L1 * c1 + self.L2 * c12 + self.L3 * c123
        dy_dq2 = self.L2 * c12 + self.L3 * c123
        dy_dq3 = self.L3 * c123

        return np.array(
            [[dx_dq1, dx_dq2, dx_dq3], [dy_dq1, dy_dq2, dy_dq3]],
            dtype=np.float64,
        )

    #############################
    def c(self, y: np.ndarray, r: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Feedback law: u = c(y,r,t)

        INPUTS
        y = q   : sensor signal vector  = joint angular positions        3 x 1
        r = r_d : reference signal vector  = desired effector position   3 x 1
        t       : time                                                   1 x 1

        OUPUTS
        u = dq  : control inputs vector =  joint velocities              3 x 1

        """

        # Feedback from sensors
        q = y

        # Jacobian computation
        J = self.J(q)

        # Ref
        r_desired = r
        r_actual = self.fwd_kin(q)

        # Error
        e = r_desired - r_actual

        # Desired end-effector velocity
        dr_d = self.gains @ e

        dq = np.zeros(3, dtype=np.float64)

        J_dls_inv = J.T @ np.linalg.inv(
            J @ J.T + (self.lambda_dls**2) * np.identity(2, dtype=np.float64)
        )
        dq = J_dls_inv @ dr_d

        # (Bonus) Handle joint speed limits
        dq = np.clip(dq, -self.dq_max, self.dq_max)

        return dq


###################
# Part 3
###################


class CustomDrillingController(robotcontrollers.RobotController):
    """ """

    ############################
    def __init__(self, robot_model):
        """ """

        super().__init__(dof=3)

        self.robot_model = robot_model

        # Label
        self.name = "Custom Drilling Controller"
        self.case = 0

    #############################
    def c(self, y, r, t=0):
        """
        Feedback static computation u = c(y,r,t)

        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1

        OUPUTS
        u  : control inputs vector    m x 1

        """

        # Ref
        f_e = np.array([0,0,-200])

        # Feedback from sensors
        x = y
        [q, dq] = self.x2q(x)

        # Robot model
        r = self.robot_model.forward_kinematic_effector(q)  # End-effector actual position
        J = self.robot_model.J(q)  # Jacobian matrix
        g = self.robot_model.g(q)  # Gravity vector
        H = self.robot_model.H(q)  # Inertia matrix
        C = self.robot_model.C(q, dq)  # Coriolis matrix

        ##################################
        # Votre loi de commande ici !!!
        ##################################
        kp_approche = np.array([30,30,2])
        kp_sortie = np.array([200,200,1000])
        kp_fond = np.array([1000,1000,1])
        kd_approche = np.array([20,20,1])
        kd_fond = np.array([200,200,1])
        r_d_approche = np.array([0.25,0.25, 0.40])
        r_d_fond = np.array([0.25,0.25,0.20])
        # print(r)
        
        if np.linalg.norm(r_d_approche - r)<1e-2 and self.case == 0:
            self.case = 1
        elif abs(r_d_fond[2] - r[2]) < 1e-3 and self.case == 1:
            self.case = 2


        match self.case:
            case 0:
                u = J.T @ (kp_approche * (r_d_approche - r) + kd_approche * (-J@dq)) + g
            case 1:
                    u = J.T @ (kp_fond * (r_d_fond - r) + f_e + kd_fond * (-J@dq)) + g
                    # u = J.T @ f_e + g
            case 2:
                u = J.T @ (kp_sortie * (r_d_approche - r) + kd_fond * (-J@dq)) + g
                
        return u


###################
# Part 4
###################


def goal2r(r_0: np.ndarray, r_f: np.ndarray, t_f: float):
    """

    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000  # nb of time steps

    # Number of DoF for the effector only
    m = 3

    r = np.zeros((m, l))
    dr = np.zeros((m, l))
    ddr = np.zeros((m, l))

    #################################
    # Votre code ici !!!
    ##################################

    # NOTE: We use a third order polynomial interpolation to generate the trajectory.
    #       This offers a good compromise between smoothness of the trajectory and computing complexity.

    if r_0.shape != r_f.shape:
        raise ValueError("`r_0` and `r_f` MUST have the same shape.")

    t = np.linspace(0.0, t_f, l)

    for i in range(m):
        # Compute positions
        r[i, :] = (
            r_0[i]
            + (3 * (r_f[i] - r_0[i]) / (t_f**2)) * t**2
            + (-2 * (r_f[i] - r_0[i]) / (t_f**3)) * t**3
        )

        # Compute speeds
        dr[i, :] = (6 * (r_f[i] - r_0[i]) / (t_f**2)) * t + (
            -6 * (r_f[i] - r_0[i]) / (t_f**3)
        ) * t**2

        # Compute accelerations
        ddr[i, :] = (6 * (r_f[i] - r_0[i]) / (t_f**2)) + (
            -12 * (r_f[i] - r_0[i]) / (t_f**3)
        ) * t

    return r, dr, ddr


def r2q(r, dr, ddr, manipulator):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    manipulator : pyro object

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]

    # Number of DoF
    n = 3

    x, y, z = r
    L1, L2, L3 = manipulator.l1, manipulator.l2, manipulator.l3
    q1 = np.atan2(y, x)
    c3 = (x**2 + y**2 + (z-1)**2 - L2**2 - L3**2 ) / (2*L2*L3)
    s3 = np.sqrt(1 - c3**2)
    q3 = np.atan2(s3, c3)
    s2 = ((L2 + L3*c3)*(z - L1) - (L3*s3)*(x*np.cos(q1) + y*np.sin(q1))) / (L3**2 * s3**2 - (L2 + L3 * c3**2))
    c2 = np.sqrt(1 - s2**2)
    q2 = np.atan2(s2, c2)

    # Output dimensions
    dq = np.zeros((n, l))
    ddq = np.zeros((n, l))

    #################################
    # Votre code ici !!!
    ##################################

    q = np.array([q1, q2, q3])
    J = manipulator.J(q)
    dq = np.linalg.inv(J) @ dr
    ddq = 


    # TODO: depends on DH parameters of drilling robot

    return q, dq, ddq


def q2torque(q, dq, ddq, manipulator):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    manipulator : pyro object

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]

    # Number of DoF
    n = 3

    # Output dimensions
    tau = np.zeros((n, l))

    #################################
    # Votre code ici !!!
    ##################################

    # TODO: depends on DH parameters of drilling robot

    return tau
