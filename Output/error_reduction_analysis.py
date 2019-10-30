import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


def make_time_slices():
    plt.figure()
    plt.title(r"Time-slice error at various time steps")
    plt.xlabel(r"$\dim X_\delta$")
    plt.ylabel(r"$||\gamma_t(u - u_\delta)||_{L_2(\Omega)}$")
    for i in range(0, 9, 2):
        plt.loglog(data['dofs'], [err[i] for err in data['errors']],
                   '-+',
                   label='t=%g' % (i / 8))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("error_reduction_time_slices.pdf")


def make_rates():
    plt.figure()
    plt.title(
        "Average error reduction rates as function of start- and end-index")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Average rate")

    np.linspace(0, 1, 9)
    len_dofs = len(data['dofs'])
    for start, end in product([0, 1], [len_dofs, len_dofs - 1, len_dofs - 2]):
        rates = [
            -np.polyfit(np.log(data['dofs'])[start:end],
                        np.log([err[i] for err in data['errors']])[start:end],
                        deg=1)[0] for i in range(9)
        ]
        plt.plot(np.linspace(0, 1, 9),
                 rates,
                 label="data[%g:%g]" % (start, end))
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_reduction_rates.pdf")


def make_linearity():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Time per apply")
    ax1.set_xlabel(r"$\# X_\delta + \# Y_\delta$")
    ax1.set_ylabel("time (s)")
    ax1.loglog(data['dofs'], [
        sum(dims) * t for (dims, t) in zip(data['dims'], data['time_per_dof'])
    ], '-x')
    ax1.tick_params(axis='y')
    ax1.grid()

    print(
        np.polyfit(np.log(data['dofs']),
                   np.log([t for t in data['time_per_dof']]),
                   deg=2))
    ax2.set_title("Time per apply per doublenode")
    ax2.set_xlabel(r"$\# X_\delta + \# Y_\delta$")
    ax2.set_ylabel("time (ms)")
    ax2.semilogx(data['dofs'], [1000 * t for t in data['time_per_dof']], '-x')
    ax2.tick_params(axis='y')
    ax2.set_ylim((0, 3.0))
    ax2.grid()
    plt.tight_layout()
    plt.savefig("error_reduction_linearity.pdf")


def make_minres():
    histories = data['residual_norm_histories']
    histories = histories[:8]

    plt.figure()
    plt.xlabel(r"$\# X_\delta + \# Y_\delta$")
    plt.ylabel(r"MINRES iterations")
    plt.loglog(data['dofs'], data['minres_iters'])
    plt.grid()
    plt.savefig("error_reduction_minres_iters.pdf")

    plt.figure()
    plt.ylim(1e-5, 1e-0)
    plt.ylabel(r"$||M x_k - b||_2$")
    plt.xlabel(r"MINRES iteration")
    for i, hist in enumerate(histories):
        plt.semilogy(hist, label="solve %s" % (i + 1))
    plt.grid()
    plt.legend()
    plt.savefig("error_reduction_minres_history_semilogy.pdf")

    plt.figure()
    plt.ylim(1e-5, 1e-0)
    plt.ylabel(r"$||M x_k - b||_2$")
    plt.xlabel(r"MINRES iteration")
    for i, hist in enumerate(histories):
        plt.loglog(hist, label="solve %s" % (i + 1))
    plt.grid()
    plt.legend()
    plt.savefig("error_reduction_minres_history_loglog.pdf")


data = pd.read_pickle("error_reduction_v2.pickle")
print(data)

make_time_slices()
make_rates()
make_linearity()
make_minres()
