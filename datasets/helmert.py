"""
Created on 10/30/18

@author: Baoxiong Jia

Description:

"""
import numpy as np

def helmert_affine_3d(datum1, datum2):
    '''
    :param datum1: n*3 matrix
    :param datum2: n*3 matrix
    :return: 
    '''
    s1 = datum1.shape
    s2 = datum2.shape
    N = s1.shape[0]

    G = np.zeros((3 * N, 12), dtype=np.float)
    E1 = np.ones((N, 1), dtype=np.float)
    Z1 = np.zeros((N, 1), dtype=np.float)
    z3 = np.zeros(N, dtype=np.float)
    G

def helmert_3d(datum1, datum2, type='7p', without_scale=0, approx = np.zeros(3, dtype=np.float)):
    '''
    :param datum1: n*3 matrix
    :param datum2: n*3 matrix
    :param type: '7p' or '10p'
    :param without_scale: 0
    :param approx: (3,) vector
    :return:
    '''
    sof = 1
    # Check parameter validity
    assert(len(datum1.shape) == 2 and len(datum2.shape) == 2, 'datum1 and datum2 should be matrices')
    assert(len(approx.shape) == 2 and approx.shape[1] == 3, 'the approx vector should be 1 * 3')
    assert(isinstance(type, str), 'type parameter should be a string')
    if type is '7p':
        rc = np.zeros(3, dtype=np.float)
    else:
        # Case '10p'
        rc = np.mean(datum1, axis=0)

    # Check data validity
    N1 = datum1.shape[0]
    N2 = datum2.shape[0]
    N = N1
    assert(N1 == N2, 'datum1 and datum2 should have the same dimension')
    assert(datum1.shape[1] == 3 and datum2.shape[1] == 3, 'both datum matrix should be of N*3')

    # naeh should be (7,) vector, set the naeh vector
    naeh = np.concatenate((np.zeros(3, dtype=np.float), approx, np.array([1], dtype=np.float)))
    if np.array_equal(approx, np.zeros(3,dtype=np.float)) and N > 3:
        # TODO: add helmert_affine_3d transformation and debug
        pass
    if without_scale != 0:
        naeh[6] = without_scale

    wert_A = np.array([1e-8, 1e-8])
    zaehl = 0
    x0 = naeh[0]
    y0 = naeh[1]
    z0 = naeh[2]
    ex = naeh[3]
    ey = naeh[4]
    ez = naeh[5]
    m = naeh[6]
    tp = np.array([x0, y0, z0, ex, ey, ez, m])
    qbb = np.eye(3 * N)
    while True:
        A = np.zeros((3 * N, 7), dtype=np.float)
        w = np.zeros((3 * N, 1), dtype = np.float)
        for i in range(N):
            A[i * 3][0] = -1
            A[i * 3 + 1][1] = -1
            A[i * 3 + 2][2] = -1
            A[i * 3][3]= -m * ((np.cos(ex) * np.sin(ey) * np.cos(ez) - np.sin(ex) * np.sin(ez)) *(datum1[i][1] - rc[1])
                               + (np.sin(ex) * np.sin(ey) * np.cos(ez) + np.cos(ex) * np.sin(ey)) *(datum1[i][2] - rc[2]))
            A[i * 3][4] = -m * ((-np.sin(ey) * np.cos(ez)) * (datum1[i][0] - rc[0]) +
                                (np.sin(ex) * np.cos(ey) * np.cos(ez)) * (datum1[i][1] - rc[1]) +
                                (-np.cos(ex) * np.cos(ey) * np.cos(ez)) * (datum1[i][2] - rc[3]))
            A[i * 3][5] = -m * ((-np.cos(ey) * np.sin(ez)) * (datum1[i][0] - rc[0]) +
                                (-np.sin(ex) * np.sin(ey) * np.sin(ez) + np.cos(ex) * np.cos(ez)) * (datum1[i][1]-rc[2]) +
                                (np.cos(ex) * np.sin(ey) * np.sin(ez) + np.sin(ex)* np.cos(ex)) * (datum1[i][2]-rc[3]))
            A[i * 3][6] = -((np.cos(ey) * np.cos(ez)) * (datum1[i][0] - rc[0]) +
                            (np.sin(ex) * np.sin(ey) * np.cos(ez) + np.cos(ex) * np.sin(ez)) * (datum1[i][1] - rc[1]) +
                            (-np.cos(ex) * np.sin(ey) * np.cos(ez) + np.sin(ex) * np.sin(ez)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 1][3] = -m * ((-np.cos(ex) * np.sin(ey) * np.sin(ez) - np.sin(ex) * np.cos(ez)) * (datum1[i][1] - rc[1]) +
                                    (-np.sin(ex) * np.sin(ey) * np.sin(ez) + np.cos(ex) * np.cos(ez)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 1][4] = -m * ((np.sin(ey) * np.sin(ez)) * (datum1[i][0] - rc[0]) +
                                    (-np.sin(ex) * np.cos(ey) * np.sin(ez)) * (datum1[i][1] - rc[1]) +
                                    (np.cos(ex) * np.cos(ey) * np.sin(ez)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 1][5] = -m * ((-np.cos(ey) * np.cos(ez)) * (datum1[i][0] - rc[0]) +
                                    (-np.sin(ex) * np.sin(ey) * np.cos(ez) - np.cos(ex) * np.sin(ez)) * (datum1[i][1] - rc[1]) +
                                    (np.cos(ex) * np.sin(ey) * np.cos(ez) + np.sin(ex) * np.sin(ez)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 1][6] = -((-np.cos(ey) * np.sin(ez)) * (datum1[i][0] - rc[1]) +
                                (-np.sin(ex) * np.sin(ey) * np.sin(ez) + np.cos(ex) * np.cos(ez)) * (datum1[i][1] - rc[1]) +
                                (np.cos(ex) * np.sin(ey) * np.sin(ez) + np.sin(ex) * np.cos(ez)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 2][3] = -m * ((-np.cos(ex) * np.cos(ey)) * (datum1[i][1] - rc[1]) +
                                    (-np.sin(ex) * np.cos(ey)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 2][4] = -m * ((np.cos(ey)) * (datum1[i][0] - rc[0]) +
                                    (np.sin(ex) * np.sin(ey)) * (datum1[i][1] - rc[1]) +
                                    (-np.cos(ex) * np.sin(ey)) * (datum1[i][2] - rc[2]))
            A[i * 3 + 2][5] = 0
            A[i * 3 + 2][6] = -((np.sin(ey)) * (datum1[i][0] - rc[0]) +
                                (-np.sin(ex) * np.cos(ey)) * (datum1[i][1] - rc[1]) +
                                (np.cos(ex) * np.cos(ey)) * (datum1[i][2] - rc[2]))
            w[i * 3][0] = -rc[0] + datum2[i][0]- x0 - m * ((np.cos(ey) * np.cos(ez)) * (datum1[i][0] - rc[0]) +
                                                        (np.sin(ex) * np.sin(ey) * np.cos(ez) + np.cos(ex)* np.sin(ez)) * (datum1[i][1] - rc[1]) +
                                                        (-np.cos(ex) * np.sin(ey) * np.cos(ez) + np.sin(ex) * np.sin(ez)) * (datum1[i][2] - rc[2]))
            w[i * 3 + 1][0] = -rc[1] + datum2[i][1] - y0 - m * ((-np.cos(ey) * np.sin(ez)) * (datum1[i][0] - rc[0]) +
                                                                (-np.sin(ex) * np.sin(ey) * np.sin(ez) + np.cos(ex) * np.cos(ez)) * (datum1[i][1] - rc[1]) +
                                                                (np.cos(ex) * np.sin(ey) * np.sin(ez) + np.sin(ex) * np.cos(ez)) * (datum1[i][2] - rc[2]))
            w[i * 3 + 2][0] = -rc[2] + datum2[i][2] - z0 - m * ((np.sin(ey))*(datum1[i][0] - rc[0]) +
                                                                (-np.sin(ex) * np.cos(ey)) * (datum1[i][1] - rc[1]) +
                                                                (np.cos(ex) * np.cos(ey)) * (datum1[i][2] - rc[2]))
        if without_scale != 0:
            A = A[:, : -1]

        w = -1. * w
        r = A.shape[0] - A.shape[1]
        pbb = np.linalg.inv(qbb)
        quadra_A = np.matmul(np.matmul(A.T, pbb), A)
        inv_quadra_A = np.linalg.inv(quadra_A)
        delta_x = np.matmul(inv_quadra_A, np.matmul(np.matmul(A.T, pbb), w))
        v = np.matmul(A, delta_x) - w
        quadra_v = np.matmul(np.matmul(v.T, pbb), v)
        sig0p = np.sqrt(quadra_v / r)
        qxxda = inv_quadra_A
        kxxda = sig0p ** 2 * qxxda
        ac = np.sqrt(np.diag(kxxda))

        delta_x = delta_x.reshape((-1, ))   # reshape to row vector
        testv = np.sqrt((delta_x[0] ** 2 + delta_x[1] ** 2 + delta_x[2] ** 2) / 3.)
        testd = np.sqrt((delta_x[3] ** 2 + delta_x[4] ** 2 + delta_x[5] ** 2) / 3.)
        zaehl = zaehl + 1
        x0 = x0 + delta_x[0]
        y0 = y0 + delta_x[1]
        z0 = z0 + delta_x[2]
        ex = ex + delta_x[3]
        ey = ey + delta_x[4]
        ez = ez + delta_x[5]
        if without_scale == 0 and (m + delta_x[6]) > 1e-15: # This condition is to prevent numerical problems with m-->0
            m = m + delta_x[6]
        tp = np.array([x0, y0, z0, ex, ey, ez, m])
        if abs(testv) < wert_A[0] and abs(testd) < wert_A[1]:
            break
        elif zaehl > 1000:
            sof = 0
            print('Iteration Limit Warning: Calculation not converging after 1000 iterations. I am aborting. Results may be inaccurate.')
            break

    if len(np.argwhere(np.abs(tp[3:6]) > 2 * np.pi)) > 0:
        print('Approximate Accuracy Warning: Rotation angles seem to be big. A better approximation is regarded. Results will be inaccurate.')

    idz = np.zeros_like(datum1)
    for i in range(N):
        idz[i][1] = rc[1] + tp[1] + tp[6] * ((-np.cos(tp[4]) * np.sin(tp[5])) * (datum1[i][0] - rc[0]) +
                                             (-np.sin(tp[3]) * np.sin(tp[4]) * np.sin(tp[5]) + np.cos(tp[3]) * np.cos(tp[5])) * (datum1[i][1] - rc[1]) +
                                             (np.cos(tp[3]) * np.sin(tp[4]) * np.sin(tp[5]) + np.sin(tp[3]) * np.cos(tp[5]))*(datum1[i][2] - rc[2]))
        idz[i][0] = rc[0] + tp[0] + tp[6] * ((np.cos(tp[4]) * np.cos(tp[5])) * (datum1[i][0] - rc[0]) +
                                             (np.sin(tp[3]) * np.sin(tp[4]) * np.cos(tp[5]) + np.cos(tp[3]) * np.sin(tp[5])) *(datum1[i][1] - rc[1]) +
                                             (-np.cos(tp[3]) * np.sin(tp[4]) * np.cos(tp[5]) + np.sin(tp[3]) * np.sin(tp[5])) * (datum1[i][2] - rc[2]))
        idz[i][2] = rc[2] + tp[2] + tp[6] * ((np.sin(tp[4])) * (datum1[i][0] - rc[0]) +
                                             (-np.sin(tp[3]) * np.cos(tp[4])) * (datum1[i][1] - rc[1]) +
                                             (np.cos(tp[3]) * np.cos(tp[4])) * (datum1[i][2] - rc[2]))
    tr = datum2 - idz
    return tp, rc, ac, tr, sof

def helmert_2d():
    pass

def test():
    cases = [
                (
                    np.array([[0.0304347500000000, 0.271670000000000, 1.67570700000000],
                           [0.140380900000000, 0.314954300000000, 1.89607300000000],
                           [-0.153808100000000, -0.135794000000000, 1.85765100000000],
                           [0.0416980000000000, -0.239627400000000, 1.69971600000000]]),
                    np.array([[0.0304347500000000, 0.271670000000000, 1.67570700000000],
                           [0.140380900000000, 0.314954300000000, 1.89607300000000],
                           [-0.153808100000000, -0.135794000000000, 1.85765100000000],
                           [0.0416980000000000, -0.239627400000000, 1.69971600000000]])
                ),
                (
                    np.array([[0.0343117800000000, 0.219011300000000, 1.65202900000000],
                              [0.144150100000000, 0.265266300000000, 1.86847700000000],
                              [-0.174984600000000, -0.176452300000000, 1.83610600000000],
                              [0.0246211000000000, -0.278663500000000, 1.68597900000000]]),
                    np.array([[0.0304347500000000, 0.271670000000000, 1.67570700000000],
                              [0.140380900000000, 0.314954300000000, 1.89607300000000],
                              [-0.153808100000000, -0.135794000000000, 1.85765100000000],
                              [0.0416980000000000, -0.239627400000000, 1.69971600000000]])
                )
            ]
    for case in cases:
        output = helmert_3d(case[0], case[1])
if __name__ == '__main__':
    test()