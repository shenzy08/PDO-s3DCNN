# Wigner-D matrix
import os
import numpy as np


def z_rot_mat(angle, l):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).
    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    M = np.zeros((2 * l + 1, 2 * l + 1))
    inds = np.arange(0, 2 * l + 1, 1)
    reversed_inds = np.arange(2 * l, -1, -1)
    frequencies = np.arange(l, -l - 1, -1)
    M[inds, reversed_inds] = np.sin(frequencies * angle)
    M[inds, inds] = np.cos(frequencies * angle)
    return M

def rot_mat(alpha, beta, gamma, l, J):
    """
    Compute the representation matrix of a rotation by ZYZ-Euler
    angles (alpha, beta, gamma) in representation l in the basis
    of real spherical harmonics.
    The result is the same as the wignerD_mat function by Johann Goetz,
    when the sign of alpha and gamma is flipped.
    The forementioned function is here:
    https://sites.google.com/site/theodoregoetz/notes/wignerdfunction
    """
    Xa = z_rot_mat(alpha, l)
    Xb = z_rot_mat(beta, l)
    Xc = z_rot_mat(gamma, l)
    return Xa.dot(J).dot(Xb).dot(J).dot(Xc)


def wigner_D_matrix(l, alpha, beta, gamma,
                    field='real', normalization='quantum', order='centered', condon_shortley='cs'):
    """
    Evaluate the Wigner-d matrix D^l_mn(alpha, beta, gamma)
    :param l: the degree of the Wigner-d function. l >= 0
    :param alpha: the argument. 0 <= alpha <= 2 pi
    :param beta: the argument. 0 <= beta <= pi
    :param gamma: the argument. 0 <= gamma <= 2 pi
    :param field: 'real' or 'complex'
    :param normalization: 'quantum', 'seismology', 'geodesy' or 'nfft'
    :param order: 'centered' or 'block'
    :param condon_shortley: 'cs' or 'nocs'
    :return: D^l_mn(alpha, beta, gamma) in the chosen basis
    """
    base = '../J_dense_0-150.npy'
#     base = 'J_dense_0-150.npy'
    path = base
    # path = os.path.join(os.path.dirname(__file__), base)
    Jd = np.load(path, allow_pickle=True)
    D = rot_mat(alpha=alpha, beta=beta, gamma=gamma, l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != ('real', 'quantum', 'centered', 'cs'):
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        D = B.dot(D).dot(BB)

        if field == 'real':
            # print('WIGNER D IMAG PART:', np.sum(np.abs(D.imag)))
            assert np.isclose(np.sum(np.abs(D.imag)), 0.0)
            D = D.real

    return D