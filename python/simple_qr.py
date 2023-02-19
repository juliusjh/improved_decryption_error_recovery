# from python.helpers import sub_vec, mat_mul


def copy_matrix(mat):
    return [mi.copy() for mi in mat]


def mat_mul(mat, vec, q=None):
    if q is None:
        return (sum((m_ij * v_j for m_ij, v_j in zip(m_i, vec))) for m_i in mat)
    else:
        return (
            sum(((m_ij * v_j) % q for m_ij, v_j in zip(m_i, vec))) % q for m_i in mat
        )


def mat_mat_mul(mat0, mat1, q=None):
    assert all((len(mi) == len(mat1) for mi in mat0))
    res = [[0 for _ in range(len(mat1))] for _ in range(len(mat0))]
    for i in range(len(mat0)):
        for j in range(len(mat1)):
            for k in range(len(mat1)):
                if q:
                    res[i][j] += (mat0[i][k] * mat1[k][j]) % q
                else:
                    res[i][j] += mat0[i][k] * mat1[k][j]
            if q:
                res[i][j] %= q
    return res


def sub_vec(v0, v1, q=None):
    if q is None:
        return map(lambda x: x[0] - x[1], zip(v0, v1))
    else:
        return map(lambda x: (x[0] - x[1]) % q, zip(v0, v1))


def dot(a, b, q):
    return sum((ai * bi % q for ai, bi in zip(a, b)))


def project(u, a, q):
    scalar = dot(u, a, q)
    norm_sq_inv = pow(dot(u, u, q), -1, q) if scalar != 0 else 0
    fac = (scalar * norm_sq_inv) % q
    vec = [(fac * ui) % q for ui in u]
    return vec, fac


def remove_component_along(v, vi, q):
    proj, fac = project(v, vi, q)
    return sub_vec(vi, proj, q), fac


def qr(a, q, inplace=False):
    if not inplace:
        a = copy_matrix(a)
    r = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]
    for i in range(0, len(a)):
        for j in range(i + 1, len(a)):
            v, fac = remove_component_along(a[i], a[j], q)
            a[j] = list(v)
            if i < len(r[j]):
                r[j][i] = fac
        if i < len(r[i]):
            r[i][i] = 1
    nz = 0
    for qi in a:
        if all((qij == 0 for qij in qi)):
            break
        nz += 1
    a = a[:nz]

    # if normalise:
    #    for i in range(len(a)):
    #        norm = math.sqrt(dot(a[i], a[i], q)) #TODO: square root mod q
    #        norm_inv = pow(norm, -1, q)
    #        a[i] = [(norm_inv*aij) % q for aij in a[i]]
    #        for j in range(len(r)):
    #            r[j][i] = (r[j][i]*norm) % q
    return a, r


def test():
    a = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    q, r = qr(a, 23)
    res = list(mat_mat_mul(r, q, 23))
    assert res == a

    a = [[1, 2, 3], [2, 5, 6], [0, 2, 1]]
    q, r = qr(a, 23)
    res = list(mat_mat_mul(r, q, 23))
    assert res == a

    a = [[1, 2, 3], [2, 5, 6], [0, 2, 1], [0, 1, 2]]
    q, r = qr(a, 23)
    res = list(mat_mat_mul(r, q, 23))
    assert res == a


if __name__ == "__main__":
    test()
