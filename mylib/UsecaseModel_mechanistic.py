
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import odeint





class HitDeliveryModel:
    """
    Multi-hit T-cell killing model with T-cell death (uniform rate across states).

    State vector: C = [C0, C1, ..., C_nhits, C0:T, C1:T, ..., C_{nhits-1}:T, T]

    - C0..C_{nhits-1}: free tumor cells with 0..nhits-1 hits
    - C_nhits: dead/removed tumor cells (nhits hits received)
    - C0:T..C_{nhits-1}:T: tumor-T complexes at hit stage 0..nhits-1
    - T: free T cells

    T-cells die at rate d_T regardless of binding state. When a T-cell dies
    inside a bound complex, the tumor cell returns to the corresponding
    free pool C_i at the same hit level (hits already received are not undone).
    """

    def __init__(
        self,
        k1=1e-5,       # binding rate [1/(cell*min)]
        k2=2.5e-3,     # unbinding rate [1/hr]
        k3=1e-3,       # hit delivery rate [1/min]
        u0=5e-4,       # tumor growth rate [1/min]
        d=0.006,       # dead cell clearance rate [1/min]
        d_T=1e-10,      # T-cell death rate (uniform across all states) [1/min]
        nhits=4,        # number of hits to kill
        Kmax=500,       # carrying capacity
    ):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.u0 = u0
        self.d = d
        self.d_T = d_T
        self.nhits = nhits
        self.Kmax = Kmax

    def rhs(self, C, t):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        u0 = self.u0
        d = self.d
        d_T = self.d_T
        nhits = self.nhits
        Kmax = self.Kmax

        totalcell = np.sum(C[0:-1])
        v = nhits + 1  # number of free tumor compartments (0..nhits)

        derivs = []

        # Free tumor cells: C0 .. C_nhits
        for hit_idx in range(nhits + 1):
            if hit_idx == 0:
                # Proliferation of pristine cells (both free and bound forms),
                # binding loss, unbinding return, and return from T-cell death in complex.
                derivs.append(
                    u0 * (C[0] + C[v + 0]) * (1 - totalcell / Kmax)
                    - k1 * C[0] * C[-1]
                    + k2 * C[v + 0]
                    + d_T * C[v + 0]      # T-cell death in [C0:T] returns C0
                )
            elif hit_idx < nhits:
                # Free intermediates: binding loss, unbinding return, and
                # return from T-cell death in [C_i:T] (tumor stays at hit level i).
                derivs.append(
                    - k1 * C[hit_idx] * C[-1]
                    + k2 * C[v + hit_idx]
                    + d_T * C[v + hit_idx]   # T-cell death in [C_i:T] returns C_i
                )
            else:  # hit_idx == nhits (dead pool)
                # Cells that completed nhits hits, accumulating from the chain output
                # and being cleared at rate d.
                derivs.append(
                    k3 * C[-2] - d * C[hit_idx]
                )

        # Complexes: C0:T .. C_{nhits-1}:T
        for complex_id in range(nhits):
            if complex_id == 0:
                # First complex: binding, unbinding, hit-transition out,
                # and T-cell death loss.
                derivs.append(
                    k1 * C[0] * C[-1]
                    - k2 * C[v + 0]
                    - k3 * C[v + 0]
                    - d_T * C[v + 0]      # T-cell death in this complex
                )
            else:
                # Intermediate complexes: binding, unbinding, hit-transition in/out,
                # and T-cell death loss.
                derivs.append(
                    k1 * C[complex_id] * C[-1]
                    - k2 * C[v + complex_id]
                    - k3 * C[v + complex_id]
                    + k3 * C[v + complex_id - 1]
                    - d_T * C[v + complex_id]   # T-cell death in this complex
                )

        # Free T cells: binding loss, unbinding return, release after final hit,
        # and free-T-cell death.
        derivs.append(
            - k1 * C[-1] * np.sum(C[0:nhits])
            + k2 * np.sum(C[v:v + nhits])
            + k3 * C[-2]
            - d_T * C[-1]                # Free T-cell death
        )

        return np.array(derivs)

    def solve(self, C0_tumor, T0, t_span=(0, 2900), n_points=2000):
        nhits = self.nhits
        v = nhits + 1
        n_states = v + nhits + 1
        y0 = np.zeros(n_states)
        y0[0] = C0_tumor
        y0[-1] = T0

        t = np.linspace(t_span[0], t_span[1], n_points)
        sol = odeint(self.rhs, y0, t)

        free_tumor_alive = np.sum(sol[:, 0:nhits], axis=1)
        dead_tumor = sol[:, nhits]
        complexes = sol[:, v:v + nhits]
        complex_tumor = np.sum(complexes, axis=1)
        total_tumor = free_tumor_alive + complex_tumor
        free_T = sol[:, -1]
        total_T = free_T + complex_tumor

        return {
            't': t,
            'total_tumor': total_tumor,
            'free_tumor': free_tumor_alive,
            'complex_tumor': complex_tumor,
            'dead_tumor': dead_tumor,
            'free_T': free_T,
            'total_T': total_T,
        }

    def get_total_tumor_at_end(self, C0_tumor, T0, t_end=2900):
        res = self.solve(C0_tumor, T0, t_span=(0, t_end), n_points=500)
        return res['total_tumor'][-1]





class TwoStatePhenotypeModel:
    """
    Two-compartment phenotype-switching model:

        A --k1--> B
        A <--k2-- B

    Dynamics:
        dA/dt = -k1 * A + k2 * B
        dB/dt =  k1 * A - k2 * B
    """

    def __init__(self, k1, k2, A0=50, B0=50):
        """
        Parameters
        ----------
        k1 : float
            Transition rate from phenotype A to phenotype B.
        k2 : float
            Transition rate from phenotype B to phenotype A.
        A0 : float
            Initial number of cells in state A.
        B0 : float
            Initial number of cells in state B.
        """
        if k1 < 0 or k2 < 0:
            raise ValueError("Transition rates k1 and k2 must be non-negative.")

        if A0 < 0 or B0 < 0:
            raise ValueError("Initial cell populations must be non-negative.")

        self.k1 = k1
        self.k2 = k2
        self.A0 = A0
        self.B0 = B0
        self.total_population = A0 + B0

        # These attributes will be filled after simulation.
        self.time_history = None
        self.A_history = None
        self.B_history = None

    def dynamics(self, t, populations):
        """
        Right-hand side of the ODE system.
        """
        A, B = populations

        dA_dt = -self.k1 * A + self.k2 * B
        dB_dt = self.k1 * A - self.k2 * B

        return [dA_dt, dB_dt]

    def simulate(self, t_end=20, number_of_points=500):
        """
        Simulate the phenotype dynamics and store the time history.

        Parameters
        ----------
        t_end : float
            Final simulation time.
        number_of_points : int
            Number of time points stored in the solution.

        Returns
        -------
        time_history, A_history, B_history : numpy arrays
        """
        time_points = np.linspace(0, t_end, number_of_points)

        solution = solve_ivp(
            fun=self.dynamics,
            t_span=(0, t_end),
            y0=[self.A0, self.B0],
            t_eval=time_points,
            method="RK45"
        )

        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")

        self.time_history = solution.t
        self.A_history = solution.y[0]
        self.B_history = solution.y[1]

        return self.time_history, self.A_history, self.B_history

    def steady_state(self):
        """
        Return the analytical steady-state populations.
        """
        if self.k1 + self.k2 == 0:
            # There are no transitions. The initial condition is unchanged.
            return self.A0, self.B0

        A_steady = self.total_population * self.k2 / (self.k1 + self.k2)
        B_steady = self.total_population * self.k1 / (self.k1 + self.k2)

        return A_steady, B_steady

    def plot(self):
        """
        Plot the dynamics of A and B on a single graph.
        """
        if self.time_history is None:
            raise RuntimeError("Run the simulate() method before plotting.")

        A_steady, B_steady = self.steady_state()

        plt.figure(figsize=(7, 4.5))

        plt.plot(
            self.time_history,
            self.A_history,
            linewidth=2,
            label="Phenotype A"
        )

        plt.plot(
            self.time_history,
            self.B_history,
            linewidth=2,
            label="Phenotype B"
        )

        plt.axhline(
            A_steady,
            linestyle="--",
            alpha=0.5,
            label=f"A steady state = {A_steady:.2f}"
        )

        plt.axhline(
            B_steady,
            linestyle="--",
            alpha=0.5,
            label=f"B steady state = {B_steady:.2f}"
        )

        plt.xlabel("Time")
        plt.ylabel("Number of cells")
        plt.title("Two-state phenotype dynamics")
        plt.legend()
        plt.tight_layout()
        plt.show()