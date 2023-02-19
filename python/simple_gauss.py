# For testing purposes

# Redefinition because of pythons terrible import system
def mat_mul(mat, vec, q=None):
    if q is None:
        return (sum((m_ij * v_j for m_ij, v_j in zip(m_i, vec))) for m_i in mat)
    else:
        return (
            sum(((m_ij * v_j) % q for m_ij, v_j in zip(m_i, vec))) % q for m_i in mat
        )


def sub_vec(v0, v1, q=None):
    if q is None:
        return map(lambda x: x[0] - x[1], zip(v0, v1))
    else:
        return map(lambda x: (x[0] - x[1]) % q, zip(v0, v1))


def simple_gauss(a, b, q, key=None):
    ab = list(zip(a, b))
    n = len(ab[0][0])
    red = lambda x: x % q
    red_row = lambda row: (list(map(red, row[0])), red(row[1]))
    reduce_ab = lambda ab: list(map(red_row, ab))
    ab = reduce_ab(ab)
    ab = sorted(ab, key=lambda ai: leading_zeros(ai[0]))
    i = 0
    while i < n:
        ab_i = ab[i]
        j = leading_zeros(ab_i[0])
        if j == n:
            break
        pivot_inv = pow(ab_i[0][j], -1, q)
        row_sub = [pivot_inv * a_ij % q for a_ij in ab_i[0]]
        b_sub = (pivot_inv * ab_i[1]) % q
        for k, ab_k in list(enumerate(ab))[i + 1 :]:
            if ab_k[0][j] == 0:
                continue
            mul = (ab_k[0][j] * pivot_inv) % q
            row_sub = [(mul * a_ij) % q for a_ij in ab_i[0]]
            # b_sub = (pivot_inv*ab_i[1]) % q
            ab_k_0 = list(sub_vec(ab_k[0], row_sub, q))
            ab_k_1 = (ab_k[1] - mul * ab_i[1]) % q
            ab[k] = (ab_k_0, ab_k_1)
            if key:
                b = list(mat_mul([ab_k_0], key))[0]
                assert b % q == ab_k_1
        i += 1
        ab = sorted(ab, key=lambda ai: leading_zeros(ai[0]))
        if key:
            b = mat_mul(map(lambda x: x[0], ab), key, q)
            assert all((bi % q == abi[1] % q for bi, abi in zip(b, ab)))
    res = [None] * n
    left = n
    i = n - 1
    while i >= 0:
        while leading_zeros(ab[i][0]) == left:
            i -= 1
        if leading_zeros(ab[i][0]) < left - 1:
            return ab[:i], res
        left_side = (
            sum([(aij * res_j) % q for aij, res_j in zip(ab[i][0][left:], res[left:])])
            % q
        )
        a_il_inv = pow(ab[i][0][left - 1], -1, q)
        res[left - 1] = (a_il_inv * (ab[i][1] - left_side)) % q
        if key:
            assert key[left - 1] == res[left - 1]
        left -= 1
        i -= 1

    return res


def leading_zeros(ai):
    j = 0
    while j < len(ai) and ai[j] == 0:
        j += 1
    return j


def test():
    a = [[1, 2, 3], [2, 5, 6], [0, 2, 1]]
    s = [1, 2, 3]
    b = mat_mul(a, s, 23)
    res = simple_gauss(a, b, 23)
    assert res == s

    a = [[1, 2, 3], [2, 5, 6], [0, 2, 1], [1, 1, 1], [2, 3, 4]]
    s = [1, 2, 3]
    b = mat_mul(a, s, 23)
    res = simple_gauss(a, b, 23)
    assert res == s


if __name__ == "__main__":
    test()
