import math
import numpy as np
from importlib import import_module
import python.version as version
from python.helpers import (
    add_vec,
    euclidean_dist,
    mat_mul,
    reduce_sym_list,
    sub_vec,
    norm,
    most_likely_list,
    print_v,
)
# Inefficient but enough for testing
from python.simple_gauss import simple_gauss

python_kyber = import_module(f"python_kyber{version.KYBER_VERSION}")


def substitute_s(lwe_instance, known_s):
    lwe = lwe_instance.copy()
    for j, s_j in known_s:
        for i, (a_i, b_i) in enumerate(zip(lwe.a, lwe.b)):
            lwe.b[i] = (b_i - a_i[j] * s_j) % lwe.q
            a_i[j] = 0
    return lwe


def substitute_e(lwe_instance, known_e, eliminate_order_s, key=None):
    q = lwe_instance.q
    lwe = lwe_instance.copy()
    n = len(lwe.a[0])
    substitution_matrix = []
    substitution_matrix_vec = []
    rows_to_delete = []
    columns_to_delete = []
    for i, e_i in known_e:
        if not eliminate_order_s:
            break
        idx_s_to_eliminate = eliminate_order_s.pop()
        put_backs = []
        while lwe.a[i][idx_s_to_eliminate] == 0:
            put_backs.append(idx_s_to_eliminate)
            # TODO: Make sure this does not fail
            idx_s_to_eliminate = eliminate_order_s.pop()
        for pb in reversed(put_backs):
            eliminate_order_s.append(pb)
        a_inverse = pow(lwe.a[i][idx_s_to_eliminate], -1, q)
        b_sub = (a_inverse * (lwe.b[i] - e_i)) % q
        s_add = [(-a_inverse * a_i) % q for a_i in lwe.a[i]]
        s_add[idx_s_to_eliminate] = None
        substitution_matrix.append(lwe.a[i].copy())
        substitution_matrix_vec.append((lwe.b[i] - e_i) % q)
        if key:
            assert list(mat_mul([lwe.a[i]], key[n:], q))[0] == (lwe.b[i] - e_i) % q
            assert (
                list(mat_mul(substitution_matrix, key[n:], q))
                == substitution_matrix_vec
            )
            s_add[idx_s_to_eliminate] = 0
            assert (
                key[idx_s_to_eliminate + n] % q
                == (b_sub + list(mat_mul([s_add], key[n:], q))[0]) % q
            )
            s_add[idx_s_to_eliminate] = None
        for i_row in range(len(lwe.a)):
            if i_row == i:
                continue
            c = lwe.a[i_row][idx_s_to_eliminate]
            for j in range(len(lwe.a[i_row])):
                if idx_s_to_eliminate != j:
                    lwe.a[i_row][j] = (lwe.a[i_row][j] + c * s_add[j]) % q
            lwe.a[i_row][idx_s_to_eliminate] = 0
            lwe.b[i_row] = (lwe.b[i_row] - c * b_sub) % q
        rows_to_delete.append(i)
        columns_to_delete.append(idx_s_to_eliminate)
    rows_to_delete = sorted(rows_to_delete, reverse=True)
    if key:
        b = list(mat_mul(substitution_matrix, key[n:], q))
        assert b == substitution_matrix_vec
    mapping_e = [None for _ in range(n)]
    k = n - len(rows_to_delete) - 1
    for i in reversed(range(n)):
        if i in rows_to_delete:
            del lwe.a[i]
            del lwe.b[i]
            if key:
                del key[i]
        else:
            mapping_e[i] = k
            k -= 1
    assert lwe.is_solution(key)
    return (
        lwe,
        (substitution_matrix, substitution_matrix_vec),
        rows_to_delete,
        columns_to_delete,
        mapping_e,
        key,
    )


def remove_solved_s(lwe_instance, del_list, key=None):
    lwe = lwe_instance.copy()
    del_list = sorted(del_list)
    n = len(lwe.a)
    m = len(lwe.a[0])
    if key:
        assert lwe.is_solution(key)
    mapping = [None for _ in range(m)]
    if not del_list:
        return lwe, mapping, key
    k = m - len(del_list) - 1
    current_to_remove = del_list.pop()
    for j in reversed(range(m)):
        if j == current_to_remove:
            for i in range(len(lwe.a)):
                assert lwe.a[i][j] == 0
                del lwe.a[i][j]
            if key:
                del key[j + n]
            if del_list:
                current_to_remove = del_list.pop()
        else:
            mapping[j] = k
            k -= 1
    assert not del_list
    assert lwe.is_solution(key)
    return lwe, mapping, key


def solve_from_substituted(known_s, substituted_equations, lwe_instance, key=None):
    q = lwe_instance.q
    n = len(substituted_equations[0][0])
    for j, s_j in known_s:
        substituted_equations[0].append([1 if k == j else 0 for k in range(n)])
        substituted_equations[1].append(s_j)
    ###
    if key:
        b = list(mat_mul(substituted_equations[0], key[512:], q))
        assert reduce_sym_list(b) == reduce_sym_list(substituted_equations[1])
    ###
    key_s = simple_gauss(substituted_equations[0], substituted_equations[1], q)
    key_s = list(reduce_sym_list(key_s, q))
    b = list(mat_mul(substituted_equations[0], key_s, q))
    assert reduce_sym_list(b) == reduce_sym_list(substituted_equations[1])
    if key:
        assert key[n:] == key_s
    a_s = list(mat_mul(lwe_instance.a, key_s, q))
    key_e = list(reduce_sym_list(sub_vec(lwe_instance.b, a_s, q)))
    if key:
        assert lwe_instance.is_solution(key)
        a_s_test = list(mat_mul(lwe_instance.a, key[n:], q))
        assert list(a_s) == list(a_s_test)
        e_test = list(reduce_sym_list(sub_vec(lwe_instance.b, a_s_test, q)))
        assert e_test == key_e
        as_e = list(reduce_sym_list(add_vec(a_s, key_e, q), q))
        assert as_e == list(reduce_sym_list(lwe_instance.b, q))
    rec_key = key_e + key_s
    if key:
        assert key == rec_key
    return rec_key


def solve(
    propagation_data,
    block_size,
    run_reduction,
    max_beta,
    perform,
    add_fplll,
    step,
    max_enum,
    step_rank,
):
    print_v("Solving LWE instance..")
    key = propagation_data.key
    n = len(key) // 2
    if step == -1:
        current_step = sorted(propagation_data.steps.items(), key=lambda x: x[0])[step][
            1
        ]
    else:
        current_step = propagation_data.steps[step]
    recovered_positions = current_step.recovered_coefficients
    ordered_key_indices = current_step.ordered_key_indices
    known_indices = ordered_key_indices[:recovered_positions]
    known_e = list(
        map(
            lambda i: (i, current_step.guessed_key[i]),
            filter(lambda i: i < n, known_indices),
        )
    )
    known_s = list(
        map(
            lambda i: (i - n, current_step.guessed_key[i]),
            filter(lambda i: i >= n, known_indices),
        )
    )
    assert all((propagation_data.key[i] == known_e_i for i, known_e_i in known_e))
    assert all((propagation_data.key[i + n] == known_s_i for i, known_s_i in known_s))
    ordered_key_indices_s = list(
        map(lambda i: i - n, filter(lambda i: i >= n, ordered_key_indices))
    )
    ordered_key_indices_e = list(
        map(lambda i: i, filter(lambda i: i < n, ordered_key_indices))
    )
    print_v("Substituting known coefficients of s..")
    lwe_instance = substitute_s(propagation_data.lwe_instance, known_s)
    assert lwe_instance.is_solution(key)
    print_v("Eliminating coefficients using known coefficients of e..")
    (
        lwe_instance,
        substituted_equations,
        removed_e,
        solved_s_indices,
        mapping_e,
        key_rem,
    ) = substitute_e(
        lwe_instance,
        known_e,
        ordered_key_indices_s[len(known_s):],
        propagation_data.key.copy(),
    )
    assert lwe_instance.is_solution(key_rem)
    assert all((mapping_e[i] is None for i in removed_e))
    assert all((mapping_e[i] is not None for i in range(n) if i not in removed_e))
    assert len(removed_e) == len(set(removed_e))
    solved_s_indices = list(map(lambda x: x[0], known_s)) + solved_s_indices
    print_v(f"Recovered {len(solved_s_indices)} indices.")
    if len(solved_s_indices) < n:
        print_v(
            f"Preparing new lwe instance with dimension {(len(lwe_instance.a), len(lwe_instance.a[0]))}.."
        )
        print_v("Removing solved s..")
        lwe, mapping_s, key_rem = remove_solved_s(
            lwe_instance, solved_s_indices, key_rem
        )
        print_v(f"New lwe instance has dim(a) = {(len(lwe.a), len(lwe.a[0]))}.")
        assert lwe.is_solution(key_rem)
        assert len(removed_e) == len(set(removed_e))
        print_v(f"New lwe instance has dim(a) = {(len(lwe.a), len(lwe.a[0]))}.")

        remaining_indices = [i for i in range(n) if i not in removed_e] + [
            i + n for i in range(n) if i not in solved_s_indices
        ]
        dist_key_remaining = [
            dist
            for i, dist in enumerate(current_step.results)
            if i in remaining_indices
        ]
        (
            success,
            res,
            usvp_basis,
            bikz,
            guess_rank,
            key_rank,
            distance,
        ) = estimate_hinted_lwe(
            lwe,
            propagation_data,
            dist_key_remaining,
            key_rem,
            max_beta,
            perform,
            block_size,
            add_fplll,
            max_enum,
        )
        propagation_data.set_lattice_data(
            usvp_basis,
            bikz,
            guess_rank,
            step,
            step_rank,
            key_rank,
            distance,
        )
        if not success:
            return False
        if not perform:
            return True

        for idx, coeff in zip(remaining_indices, res):
            assert key[idx] == coeff or key_rem[idx] == -coeff
            if idx >= n:
                solved_s_indices.append(idx)
                known_s.append((idx - n, coeff))

        assert len(solved_s_indices) == n or not success
    else:
        propagation_data.set_lattice_data(
            None, (0, 0), 0, step, step_rank, 0, 0
        )
        if not perform:
            return True

    if len(solved_s_indices) >= n:
        for i, si in known_s:
            assert key[i + n] == si
        if perform:
            key = solve_from_substituted(
                known_s,
                substituted_equations,
                propagation_data.lwe_instance,
                propagation_data.key,
            )
            assert key == propagation_data.key
        return True
    return False


# More general function for tests, not needed here
def ln_lattice_volume(basis):
    basis = np.array(basis)
    sq = basis.transpose() @ basis
    sign, log_det = np.linalg.slogdet(sq)
    return log_det / 2


def delta_beta(beta):
    delta = pow(math.pi * beta, 1 / beta)
    delta *= beta / (2 * math.pi * math.e)
    delta = pow(delta, 1 / (2 * beta - 2))
    return delta


def is_solveable(beta, dim, log_volume, norm_s):
    left = math.sqrt(beta / dim) * norm_s
    delta = delta_beta(beta)
    right = pow(delta, 2 * beta - dim - 1) * math.exp(log_volume / dim)
    return left <= right


# Inefficient but works
def get_bikz(dim, ln_volume, short, step=1, max_b=5000):
    norm_s = norm(short)
    for beta in range(50, max_b, step):
        if is_solveable(beta, dim, ln_volume, norm_s):
            if beta > 50:
                return beta, beta - step
            else:
                return beta, 0
    return None, max_b


def usvp_basis_from_lwe(lwe, guess=None):
    n = len(lwe.a)
    m = len(lwe.a[0])
    guess = guess if guess else [0 for _ in range(n + m)]
    new_b = (
        [bi - ki for bi, ki in zip(lwe.b, guess[: len(lwe.b)])]
        + [-si for si in guess[len(lwe.b):]]
        + [1]
    )
    usvp_basis = [[0 for _ in range(n + m + 1)] for _ in range(n + m)]
    for i in range(n):
        usvp_basis[i][i] = lwe.q
        for j in range(m):
            usvp_basis[j + n][i] = lwe.a[i][j]
    for i in range(m):
        usvp_basis[i + n][i + n] = -1
    usvp_basis.append(new_b)
    return usvp_basis


def sanitize_dists(dists):
    dists_new = []
    for d in dists:
        dn = {}
        for v, p in d.items():
            if p > 0.01:
                dn[v] = p
        if len(dn) == 0:
            dn = d.copy()
        dists_new.append(dn)
    return dists_new


def get_best_key(dists, max_enum, key, num_bins=50, num_merged=20):
    from histo_guesser import HistogramGuesser
    print_v("Enumerating..")
    dists = sanitize_dists(dists)
    assert len(dists) == len(key)
    guesser = HistogramGuesser.build(dists, num_bins, num_merged)
    rk = guesser.key_estimation(key)
    if rk:
        print_v(f"Approximate key rank: 2^{math.log2(rk)}")
    else:
        print_v("Approximate key rank greater than 2^128")
    best = guesser.next_key()
    assert best is not None
    best_dist = euclidean_dist(best, key)
    best_guess_idx = 0
    print_v(f"Dist of first guess (sanity check): {best_dist}")
    for i in range(max_enum):
        guess = guesser.next_key()
        if guess is None:
            break
        dist = euclidean_dist(key, guess)
        if dist < best_dist:
            best = guess
            best_dist = dist
            best_guess_idx = i
    print_v(f"Best key at position {best_guess_idx}.")
    return best, best_guess_idx, rk


def estimate_hinted_lwe(
    lwe,
    propagation_data,
    dist_key_remaining,
    key_rem,
    max_beta,
    perform,
    block_size,
    add_fplll,
    max_enum,
):
    n = len(lwe.a)
    m = len(lwe.a[0])
    guess = list(most_likely_list(dist_key_remaining))
    dist = euclidean_dist(guess, key_rem)
    print_v(f"Distance between naive hint and key: {dist}")
    if max_enum > 1:
        guess, guess_rank, key_rank = get_best_key(
            dist_key_remaining, max_enum, key_rem
        )
        dist = euclidean_dist(guess, key_rem)
        print_v(f"Distance between best hint and key: {dist}")
    key_rem = list(key_rem)
    guess_rank, key_rank = 0, None
    if dist == 0:
        print_v("Solved.")
    assert len(guess) == len(key_rem)
    assert lwe.is_solution(key_rem)
    usvp_basis = usvp_basis_from_lwe(lwe, guess)
    print_v(f"uSVP basis has dimensions {(len(usvp_basis), len(usvp_basis[0]))}")
    lat_elem = list(sub_vec(key_rem, guess))
    dim = len(usvp_basis)
    assert dim == n + m + 1
    assert all((len(b) == dim for b in usvp_basis))
    volume = n * math.log(lwe.q)
    print_v(f"Lattice volume: {volume}")
    print_v(f"Smallest vector norm: {norm(lat_elem)}")
    bikz_upper, bikz_lower = get_bikz(dim, volume, lat_elem)
    propagation_data.lattice_data = usvp_basis.copy()  # TODO: Deep copy
    propagation_data.bikz = (bikz_lower, bikz_upper)
    print_v(f"{bikz_lower} <= BIKZ <= {bikz_upper}")
    if bikz_upper <= max_beta or bikz_upper <= 50:
        if perform:
            res = bkz_fplll(
                usvp_basis,
                block_size if block_size else bikz_upper,
                propagation_data.get_dir(),
                add_fplll,
            )
            recovered_partial_key = list(add_vec(res, guess))
            return (
                True,
                recovered_partial_key,
                usvp_basis,
                (bikz_lower, bikz_upper),
                guess_rank,
                key_rank,
                dist,
            )
        return (
            True,
            None,
            usvp_basis,
            (bikz_lower, bikz_upper),
            guess_rank,
            key_rank,
            dist,
        )
    return (
        False,
        None,
        usvp_basis,
        (bikz_lower, bikz_upper),
        guess_rank,
        key_rank,
        dist,
    )


def bkz_fplll(basis, block_size, path, add_fplll):
    import subprocess
    import os
    # Strange hack because fpylll is difficult to install on some systems
    basis = str(basis).replace(",", "")
    os.makedirs("data", exist_ok=True)
    filename = f"{path}/fplll_mat.txt"
    with open(filename, "w") as f:
        f.write(basis)
    cmd = f"fplll {add_fplll} -a bkz -b {block_size} {filename}"
    # cmd = f"fplll -v -a lll {filename}"
    print_v(f"Running '{cmd}'")
    output = subprocess.check_output(cmd, shell=True)
    output = output.decode("ascii").replace("\n", ",").replace(" ", ", ")
    mat = eval(output)[0]
    return mat[0][:-1]
