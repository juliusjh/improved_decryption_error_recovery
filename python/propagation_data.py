import os
import math
import importlib
from importlib import import_module
from datetime import datetime

from python.inequalities import Inequality

import python.version as version
from python.helpers import (
    IneqType,
    mlwe_to_lwe,
    var,
    most_likely,
    print_v,
)

python_kyber = import_module(f"python_kyber{version.KYBER_VERSION}")


def sort_key_indices(probs_dict):
    sorted_props = sorted(probs_dict, key=lambda v: probs_dict[v][1])
    return sorted_props


def count_correct(probs_dict, key):
    sorted_props = sort_key_indices(probs_dict)
    i = 0
    for d in sorted_props:
        most_likely = max(probs_dict[d][0], key=lambda v: probs_dict[d][0][v])
        if most_likely != key[d]:
            break
        else:
            i += 1
    return i


def total_correct(most_likely, key):
    correct = 0
    for l, k in zip(most_likely, key):
        if l == k:
            correct += 1
    return correct


class PropagationDataStep:
    def __init__(
        self,
        guessed_key,
        correct_coefficients,
        recovered_coefficients,
        step,
        entropies,
        results,
        avg_entropy,
        max_entropy,
        variances,
        ordered_key_indices,
        distance_to_correct,
    ):
        self.guessed_key = guessed_key
        self.correct_coefficients = correct_coefficients
        self.recovered_coefficients = recovered_coefficients
        self.step = step
        self.entropies = entropies
        self.results = results
        self.avg_entropy = avg_entropy
        self.max_entropy = max_entropy
        self.variances = variances
        self.ordered_key_indices = ordered_key_indices
        self.distance_to_correct = distance_to_correct

    @classmethod
    def new(cls, results, key, step):
        guessed_key = most_likely(results)
        correct_coefficients = total_correct(guessed_key, key)
        recovered_coefficients = count_correct(results, key)
        entropies_list = [None for _ in range(len(results))]
        results_list = [None for _ in range(len(results))]
        for coeff_index in results:
            results_list[coeff_index] = results[coeff_index][0]
            entropies_list[coeff_index] = results[coeff_index][1]
        avg_entropy = sum(entropies_list) / len(entropies_list)
        max_entropy = max(entropies_list)
        variances = [var(d) for d in results_list]
        ordered_key_indices = sort_key_indices(results)
        distance_to_correct = math.sqrt(
            sum(((ki - gi) ** 2 for ki, gi in zip(key, guessed_key)))
        )
        return cls(
            guessed_key,
            correct_coefficients,
            recovered_coefficients,
            step,
            entropies_list,
            results_list,
            avg_entropy,
            max_entropy,
            variances,
            ordered_key_indices,
            distance_to_correct,
        )

    @classmethod
    def from_file(cls, file, key, step):
        module = importlib.import_module(file)
        dists = getattr(module, f"dists_{step}")
        entropies = getattr(module, f"entropies_{step}")
        assert len(dists) == len(entropies)
        result = {i: (dists[i], entropies[i]) for i in range(len(dists))}
        return cls.new(result, key, step)

    def plot_variances(self, dir, kyber_ver, plot_type, f="variances", bins=50):
        import matplotlib.pyplot as plt
        f = f"{f}.{plot_type}"
        plt.hist(self.variances[:kyber_ver], bins)
        plt.savefig(f"{dir}/e_{f}")
        plt.clf()
        plt.hist(self.variances[kyber_ver:], bins)
        plt.savefig(f"{dir}/s_{f}")
        plt.clf()

    def plot_std_devs(self, dir, kyber_ver, plot_type, f="std_devs", bins=50):
        import matplotlib.pyplot as plt
        f = f"{f}.{plot_type}"
        plt.hist(
            list(map(lambda x: math.sqrt(max(0, x)), self.variances[:kyber_ver])),
            bins,
        )
        plt.savefig(f"{dir}/e_{f}")
        plt.clf()
        plt.hist(
            list(map(lambda x: math.sqrt(max(0, x)), self.variances[kyber_ver:])),
            bins,
        )
        plt.savefig(f"{dir}/s_{f}")
        plt.clf()

    def plot_distributions(self, dir, key, kyber_ver, plot_type, plot_max):
        import matplotlib.pyplot as plt
        if plot_max is None:
            plot_max = kyber_ver
        plot_max = min(plot_max, kyber_ver)
        for i, (dist, coeff) in enumerate(
            zip(self.results[:kyber_ver], key[:plot_max])
        ):
            plot_dist(dist, coeff, f"{dir}/e_{i}.{plot_type}")
        for i, (dist, coeff) in enumerate(
            zip(self.results[kyber_ver:], key[kyber_ver : kyber_ver + plot_max])
        ):
            plot_dist(dist, coeff, f"{dir}/s_{i}.{plot_type}")

    def plot_entropies(self, dir, kyber_ver, plot_type, f="entropies", bins=50):
        import matplotlib.pyplot as plt
        f = f"{f}.{plot_type}"
        plt.hist(self.entropies[:kyber_ver], bins)
        plt.savefig(f"{dir}/e_{f}")
        plt.clf()
        plt.hist(self.entropies[kyber_ver:], bins)
        plt.savefig(f"{dir}/s_{f}")
        plt.clf()

    def plot(self, dir, key, kyber_ver, plot_type, bins, plot_max):
        import matplotlib.pyplot as plt
        self.plot_distributions(
            dir, key, kyber_ver, plot_type=plot_type, plot_max=plot_max
        )
        self.plot_entropies(dir, kyber_ver, bins=bins, plot_type=plot_type)
        self.plot_std_devs(dir, kyber_ver, bins=bins, plot_type=plot_type)
        self.plot_variances(dir, kyber_ver, bins=bins, plot_type=plot_type)

    def save_distributions(self, f, kyber_ver):
        with open(f, "a") as file:
            file.write(f"dists_e_{self.step} = [\n")
            for dist in self.results[:kyber_ver]:
                file.write(f"{dist},\n")
            file.write("]\n")
            file.write(f"dists_s_{self.step} = [\n")
            for dist in self.results[kyber_ver:]:
                file.write(f"{dist},\n")
            file.write("]\n")
            file.write(
                f"dists_{self.step} = dists_e_{self.step} + dists_s_{self.step}\n"
            )

    def save_variances(self, f, kyber_ver):
        with open(f, "a") as file:
            file.write(f"variances_e_{self.step} = {self.variances[:kyber_ver]}\n")
            file.write(f"variances_s_{self.step} = {self.variances[kyber_ver:]}\n")
            file.write(
                f"variances_{self.step} = variances_e_{self.step} + variances_s_{self.step}\n"
            )

    def save_guessed_key(self, f, kyber_ver):
        with open(f, "a") as file:
            file.write(f"guessed_e_{self.step} = {self.guessed_key[:kyber_ver]}\n")
            file.write(f"guessed_s_{self.step} = {self.guessed_key[kyber_ver:]}\n")
            file.write(
                f"guessed_{self.step} = guessed_e_{self.step} + guessed_s_{self.step}\n"
            )
            file.write(
                f"correct_coefficients_total_{self.step} = {self.correct_coefficients}\n"
            )
            file.write(
                f"correct_coefficients_chain_{self.step} = {self.recovered_coefficients}\n"
            )
            file.write(
                f"ordered_key_indices_{self.step} = {self.ordered_key_indices}\n"
            )
            file.write(
                f"norm_to_correct_key_{self.step} = {self.distance_to_correct}\n"
            )

    def save_entropies(self, f, kyber_ver):
        with open(f, "a") as file:
            file.write(f"entropies_e_{self.step} = {self.entropies[:kyber_ver]}\n")
            file.write(f"entropies_s_{self.step} = {self.entropies[kyber_ver:]}\n")
            file.write(
                f"entropies_{self.step} = entropies_e_{self.step} + entropies_s_{self.step}\n"
            )


def plot_dist(dist, correct, f):
    import matplotlib.pyplot as plt
    x, y = zip(*sorted(dist.items()))
    plt.plot(x, y)
    plt.title(str(correct))
    plt.savefig(f)
    plt.clf()


class LWEInstance:
    def __init__(self, a, b, q=3329):
        self.a = a
        self.b = b
        self.q = q
        assert len(a) == len(b)

    @classmethod
    def from_mlwe(cls, sample, q=3329):
        k = len(sample.pk.a)
        a_map = lambda p: p.intt().montgomery_reduce().to_list()
        a = [[a_map(sample.pk.a[i].to_list()[j]) for j in range(k)] for i in range(k)]
        b = sample.pk.pk.intt().montgomery_reduce().to_lists()
        ############# Sanity checks regarding NTT and Montgomery domain #############
        # intt: *R^-1
        # montgomery_reduce: *R^-1
        # to_mont: *R
        # mul: *R
        a_s = (sample.sk.sk.apply_matrix_left_ntt(sample.pk.a)).intt()
        a_s_e = a_s + sample.e
        assert (
            a_s_e.reduce().to_lists()
            == sample.pk.pk.intt().montgomery_reduce().to_lists()
        )
        #############
        a_mat = list(
            map(
                lambda ai: ai.intt().montgomery_reduce().to_lists(),
                sample.pk.a,
            )
        )
        b_vec = sample.pk.pk.intt().montgomery_reduce().to_lists()
        a, b = mlwe_to_lwe(a_mat, b_vec)

        return cls(a, b)

    def is_solution(self, key):
        n = len(self.a)
        k = len(self.a[0])
        assert len(key) == n + k
        assert all((len(ai) == k for ai in self.a))
        e = key[:n]
        s = key[n:]
        a_times_s = (sum(aij * sj for aij, sj in zip(ai, s)) for ai in self.a)
        expr = (ats_i + ei - bi for ats_i, ei, bi in zip(a_times_s, e, self.b))
        expr_mod_q = list((ri % self.q for ri in expr))
        res = all((x_i == 0 for x_i in expr_mod_q))
        return res

    def copy(self):
        a_new = list(map(lambda a_i: a_i.copy(), self.a.copy()))
        return LWEInstance(a_new, self.b.copy(), self.q)


class LatticeData:
    def __init__(
        self,
        usvp_basis,
        bikz,
        enumeration_rank,
        step_used,
        step_rank,
        key_rank,
        distance,
    ):
        self.usvp_basis = usvp_basis
        self.bikz = bikz
        self.enumeration_rank = enumeration_rank
        self.step_used = step_used
        self.step_rank = step_rank
        self.key_rank = key_rank
        self.distance = distance


class PropagationData:
    def __init__(
        self,
        steps,
        key,
        inequalities,
        filtered_cts,
        max_delta_v,
        start,
        end,
        dir,
        version,
        num_corrects,
        lwe_instance,
        settings,
        lattice_data=None,
        bikz=None,
        dir_prefix="data",
    ):
        self.key = key
        self.inequalities = inequalities
        self.filtered_cts = filtered_cts
        self.steps = steps
        self.start = start
        self.max_delta_v = max_delta_v
        self.end = end
        self.dir = dir
        self.kyber_ver = version
        self.num_corrects = num_corrects
        self.lwe_instance = lwe_instance
        self.lattice_data = lattice_data
        self.bikz = bikz
        self.dir_prefix = dir_prefix
        self.settings = settings
        self.p_correct = num_corrects / len(inequalities)

    @classmethod
    def new(cls, key, inequalities, filtered_cts, max_delta_v, lwe_instance):
        assert lwe_instance.is_solution(key)
        return cls(
            {},
            key,
            inequalities,
            filtered_cts,
            max_delta_v,
            None,
            None,
            None,
            int(version.KYBER_VERSION),
            sum([1 if ineq.is_correct else 0 for ineq in inequalities])
            if inequalities
            else None,
            lwe_instance,
            settings=None,
        )

    def set_settings(self, settings):
        self.settings = settings

    def set_dir_prefix(self, dir_prefix):
        self.dir_prefix = dir_prefix

    def set_start(self, date):
        self.start = date

    def set_end(self, date):
        self.end = date

    def add_step(self, step, data):
        assert not self.steps.get(step)
        self.steps[step] = data

    def get_dir(self):
        if not self.dir:
            self.set_dir()
        return self.dir

    def set_dir(self, dir=None):
        if not dir:
            datestr = self.start.strftime("%d%m%y%H%M%S")
            self.dir = f"{self.dir_prefix}/run_{len(self.inequalities) if self.inequalities else None}_{self.max_delta_v}_{str(self.num_corrects).replace('.', '_')}_{datestr}"
        else:
            self.dir = f"{self.dir_prefix}/{dir}"
        os.makedirs(self.dir, exist_ok=False)

    def set_lattice_data(
        self,
        usvp_basis,
        bikz,
        enumeration_rank,
        step_used,
        step_rank,
        key_rank,
        distance,
    ):
        self.lattice_data = LatticeData(
            usvp_basis,
            bikz,
            enumeration_rank,
            step_used,
            step_rank,
            key_rank,
            distance,
        )

    def has_incorrects(self):
        return self.num_corrects < len(self.inequalities)

    @classmethod
    def load_data(cls, path, load_steps=False, load_ineqs=False, load_last_step=False):
        module_path = path.replace("/", ".")
        module = importlib.import_module(f"{module_path}.run_data")
        lwe_module = importlib.import_module(f"{module_path}.lwe_instance")
        key = module.key
        assert key == lwe_module.key
        filtered_cts = module.filtered_cts
        max_delta_v = module.max_delta_v
        lwe_instance = LWEInstance(lwe_module.a, lwe_module.b)
        assert lwe_instance.is_solution(key)
        inequalities = None
        if load_ineqs or (not load_steps and not load_last_step):
            print_v("Loading inequalities..")
            ineq_module = importlib.import_module(f"{module_path}.inequalities")
            coeffs = ineq_module.coeffs
            bs = ineq_module.bs
            signs = ineq_module.signs
            is_corrects = ineq_module.is_corrects
            p_corrects = ineq_module.p_corrects
            assert (
                len(coeffs) == len(bs)
                and len(bs) == len(signs)
                and len(signs) == len(is_corrects)
            )
            inequalities = [
                Inequality(c, IneqType.LE if t == "<=" else IneqType.GE, b, ic)
                for c, t, b, ic in zip(coeffs, signs, bs, is_corrects, p_corrects)
            ]
            print_v(f"Loaded {len(inequalities)} inequalities.")
        data = cls.new(key, inequalities, filtered_cts, max_delta_v, lwe_instance)
        if load_steps:
            print_v("Loading steps..")
            steps = {}
            for subdir in (
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ):
                sp = subdir.split("_")
                if sp[0] == "step":
                    step = int(sp[1])
                    step_data = PropagationDataStep.from_file(
                        f"{module_path}.{subdir}.step_data", key, step
                    )
                    steps[step] = step_data
            data.steps = steps
            data.set_start(datetime.now())
            print_v(f"Loaded {len(steps)} steps.")
        elif load_last_step:
            print_v("Loading last step..")
            steps = {}
            subdirs = (
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
                and name.split("_")[0] == "step"
            )
            subsdirs = list(
                map(
                    lambda dirname: (int(dirname.split("_")[-1]), dirname),
                    subdirs,
                )
            )
            step, subdir = sorted(subsdirs, key=lambda x: x[0])[-1]
            step_data = PropagationDataStep.from_file(
                f"{module_path}.{subdir}.step_data", key, step
            )
            steps[step] = step_data
            data.steps = steps
            data.set_start(datetime.now())
            print_v(f"Loaded step {step}.")

        return data

    def save_data(self):
        if not self.dir:
            self.set_dir()
        print_v(f"Saving to {self.dir}")
        print_v("Saving base data..")
        with open(self.dir + "/run_data.py", "a") as f:
            f.write(f"key_e = {self.key[:self.kyber_ver]}\n")
            f.write(f"key_s = {self.key[self.kyber_ver:]}\n")
            f.write("key = key_e + key_s\n")
            f.write(f"max_delta_v = {self.max_delta_v}\n")
            f.write(f"filtered_cts = {self.filtered_cts}\n")
            f.write(f"ineqs = {len(self.inequalities)}\n")
            f.write(f"correct_ineqs = {self.num_corrects}\n")
            f.write(f"recovered_coefficients = {self.recovered_coefficients}")
        with open(self.dir + "/lwe_instance.py", "a") as f:
            f.write(f"a = {self.lwe_instance.a}\n")
            f.write(f"b = {self.lwe_instance.b}\n")
            f.write(f"e = {self.key[:self.kyber_ver]}\n")
            f.write(f"s = {self.key[self.kyber_ver:]}\n")
            f.write("key = e + s\n")
        print_v("Saving inequalities..")
        # Everthing but elegant
        if self.inequalities:
            with open(self.dir + "/inequalities.py", "a") as f:
                f.write("coeffs = [\n")
                for ineq in self.inequalities:
                    f.write(f"{ineq.coefficients},\n")
                f.write("]\n")
                f.write("signs = [\n")
                for ineq in self.inequalities:
                    f.write(('"<="' if ineq.sign == IneqType.LE else '">="') + ",\n")
                f.write("]\n")
                f.write("bs = [\n")
                for ineq in self.inequalities:
                    f.write(f"{ineq.b},\n")
                f.write("]\n")
                f.write("is_corrects = [\n")
                for ineq in self.inequalities:
                    f.write(f"{ineq.is_correct},\n")
                f.write("]\n")
                f.write("p_corrects = [\n")
                for ineq in self.inequalities:
                    f.write(f"{ineq.p_correct},\n")
                f.write("]\n")

        if self.steps:
            print_v("Saving step data..")
            for step in self.steps:
                step_dir = f"{self.dir}/step_{step}"
                os.makedirs(step_dir, exist_ok=True)
                step_file = step_dir + "/step_data.py"
                self.steps[step].save_guessed_key(step_file, self.kyber_ver)
                self.steps[step].save_variances(step_file, self.kyber_ver)
                self.steps[step].save_entropies(step_file, self.kyber_ver)
                self.steps[step].save_distributions(step_file, self.kyber_ver)
        if self.lattice_data:
            print_v("Saving lattice data..")
            with open(self.dir + "/lattice.py", "a") as f:
                f.write(f"usvp_basis = {self.lattice_data.usvp_basis}\n")
            with open(self.dir + "/lattice_params.py", "a") as f:
                f.write(f"bikz_lower = {self.lattice_data.bikz[0]}\n")
                f.write(f"bikz_upper = {self.lattice_data.bikz[1]}\n")
                f.write(f"enumeration_rank = {self.lattice_data.enumeration_rank}\n")
                f.write(f"step_used = {self.lattice_data.step_used}\n")
                f.write(f"step_rank = {self.lattice_data.step_rank}\n")
                f.write(f"key_rank = {self.lattice_data.key_rank}\n")
                f.write(f"distance = {self.lattice_data.distance}\n")
        with open(self.dir + "/settings.py", "a") as f:
            f.write(f"settings = {self.settings}")

    def plot(self, plot_type, bins, plot_max):
        if not self.dir:
            self.set_dir()
        print_v(f"Plotting into {self.dir}")
        for step in self.steps:
            print_v(f"Plotting step {step}..")
            step_dir = f"{self.dir}/step_{step}/plots"
            os.makedirs(step_dir, exist_ok=True)
            self.steps[step].plot(
                step_dir, self.key, self.kyber_ver, plot_type, bins, plot_max
            )
