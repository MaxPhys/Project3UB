import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','notebook', 'grid'])
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

N = 10

init_random = np.random.random((N,N))
lattice = np.zeros((N, N))
lattice[init_random>=0.3] = 1
lattice[init_random<0.3] = -1


plt.imshow(lattice, cmap='Greys')
plt.title("Lattice 10x10")
plt.show()


def get_energy(lattice):
    # Applies the nearest neighbours summation, with using convolve mode "wrap" we install PBC
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='wrap', cval=0)
    # Divide result by two so we don't count doubles
    return arr.sum()/2


@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True, nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    for t in range(0, times - 1):
        # 2. pick random point on array and flip spin
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr[x, y]  # initial spin
        spin_f = spin_i * -1  # proposed spin flip

        # compute change in energy
        E_i = 0
        E_f = 0
        if x > 0:
            E_i += -spin_i * spin_arr[x - 1, y]
            E_f += -spin_f * spin_arr[x - 1, y]
        if x < N - 1:
            E_i += -spin_i * spin_arr[x + 1, y]
            E_f += -spin_f * spin_arr[x + 1, y]
        if y > 0:
            E_i += -spin_i * spin_arr[x, y - 1]
            E_f += -spin_f * spin_arr[x, y - 1]
        if y < N - 1:
            E_i += -spin_i * spin_arr[x, y + 1]
            E_f += -spin_f * spin_arr[x, y + 1]

        # 3 / 4. change state with designated probabilities
        dE = E_f - E_i
        if (dE > 0) * (np.random.random() < np.exp(-BJ * dE)):
            spin_arr[x, y] = spin_f
            energy += dE
        elif dE <= 0:
            spin_arr[x, y] = spin_f
            energy += dE

        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy

    return net_spins, net_energy


spins, energies = metropolis(lattice, 10**3, 0.001, get_energy(lattice))

fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins/N**2)
ax.set_xlabel('MC Steps')
ax.set_ylabel(r'Average Magnetization')
ax.grid()
ax = axes[1]
ax.plot(energies)
ax.set_xlabel('MC Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
plt.show()


def get_spin_energy(lattice, betajs):
    ms = np.zeros(len(betajs))
    E_means = np.zeros(len(betajs))
    E_stds = np.zeros(len(betajs))
    for i, betaj in enumerate(betajs):
        spins, energies = metropolis(lattice, 1000, betaj, get_energy(lattice))
        ms[i] = spins[-1000:].mean() / N ** 2
        E_means[i] = energies[-1000:].mean()
        E_stds[i] = energies[-1000:].std()
    return ms, E_means, E_stds


betajs = np.arange(0.1, 2, 0.05)
ms, E_means_n, E_stds_n = get_spin_energy(lattice, betajs)

plt.figure(figsize=(8,5))
plt.plot(1/betajs, ms, 'o--', label='70% Spins +1, 30% Spins -1')
plt.xlabel(r'$\left(\frac{k}{J}\right)T$')
plt.ylabel(r'$\bar{m}$')
plt.legend(facecolor='white', framealpha=1)
plt.show()
