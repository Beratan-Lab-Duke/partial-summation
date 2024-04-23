import numpy as np
from scipy import integrate

def noneq_fgr(c0, c1, c2, w, kbT, eD, eA, vDA, t1=1000):
    w = np.array(w)
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    coth = 1 / np.tanh(w / (2 * kbT))
    ωDA = eD - eA
    factor1 = - (c1 - c2) ** 2 / w ** 2
    # τ = t1 - t2, t1 > t2
    def exponent1(τ): return np.sum(
        factor1 * (coth * (1 - np.cos(w * τ)) + 1j * np.sin(τ * w)))

    factor2 = 2j * (c1 - c2) * (c1 -c0) / w ** 2
    def exponent2(τ): return np.sum(
        factor2 * (np.sin(w * t1) + np.sin(w * (τ - t1))))
    
    def corr(τ): return np.exp(exponent1(τ)) * \
        np.exp(exponent2(τ)) * np.exp(1j * ωDA * τ)

    def integrand(τ): return np.real(corr(τ))
    integral, _ = integrate.quad(integrand, 0, np.inf, limit=500000, epsabs=0, epsrel=4e-10)
    print(_)
    return 2 * (vDA ** 2) * integral


def marcus_rate(c: float, e: float, kbT: float, reorg_e: float):
    return 2 * np.pi * c ** 2 / np.sqrt(4 * np.pi * kbT * reorg_e) * np.exp(-(reorg_e - e) ** 2 / (4 * kbT * reorg_e))

if __name__ == "__main__":
    """
    Example calculation. Reproduce Figure 2 in dx.doi.org/10.1021/jp400462f | J. Phys. Chem. A 2013, 117, 6196−6204
    """

    import matplotlib.pyplot as plt
    from legendre_discretization import get_vn_squared, get_approx_func

    # Lorentzian spectral density parameters. Atomic units.
    reorg_e = 2.39e-2
    Omega = 3.5e-4
    kbT = 9.5e-4 * 0.5
    eta = 1.2e-3
    freq_domain = [0, 5e-3]
    C_DA = 5e-5
    j = lambda w: 0.5 * (4 * reorg_e) * Omega ** 2 * eta * w / ((Omega ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)

    w, v_sq = get_vn_squared(j, 100, freq_domain)
    print("Discrete Reorganization E", np.sum(v_sq / w / np.pi))

    x = np.linspace(*freq_domain, 1000) # type: ignore
    plt.plot(x, j(x), label='j')
    plt.savefig('spectral_density.png')
    plt.clf()

    e = np.linspace(reorg_e - 1 * reorg_e, reorg_e + 1 * reorg_e, 60)

    c0 = 0.1 * np.sqrt(v_sq / np.pi)
    c1 = 0 * c0
    c2 = np.sqrt(v_sq / np.pi)
    rate_fgr_plus = np.vectorize(lambda ei: noneq_fgr(c0, c1, c2, w, kbT, eD=ei, eA=0, vDA=C_DA, t1=10000)
                            )(e)
    rate_fgr_minus = np.vectorize(lambda ei: noneq_fgr(-c0, c1, c2, w, kbT, eD=ei, eA=0, vDA=C_DA, t1=10000)
                            )(e)
    rate_marcus = np.vectorize(lambda ei: marcus_rate(C_DA, ei, kbT, reorg_e))(e)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(e, rate_fgr_plus, 'o-', label=r'FGR rate, +$\rho_0$')
    ax.plot(e, rate_fgr_minus, 's-', label=r'FGR rate, -$\rho_0$')
    ax.plot(e, rate_marcus, 'x', label='Marcus rate')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('-$\Delta G$ (a.u.)')
    ax.set_ylabel('Rate (a.u.)')
    ax.set_xlim(e.min(), e.max())

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.4)

    fig.savefig('golden_rule_figure_2.png')