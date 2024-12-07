import networkx as nx
from pysat.solvers import Minicard
from pysat.formula import CNF, CNFPlus, IDPool
from pysat.card import CardEnc, EncType
from utils import INFINITY

vpool = IDPool(start_from=1)

X_ID = 1
B_ID = 2
T_ID = 3
Y_ID = 4

b_id = lambda t: vpool.id((B_ID, t))
x_id = lambda i, t: vpool.id((X_ID, i, t))
t_id = lambda i, t: vpool.id((T_ID, i, t))
y_id = lambda i, t, p: vpool.id((Y_ID, i, t, p))


def goal_constraint(G: nx.Graph, cnf):
    n = len(G)
    t = 2 * n + 1
    for i in range(n):
        cnf.append([x_id(i, t)])
    cnf.append([b_id(t)])


def start_constraint(G: nx.Graph, cnf):
    n = len(G)
    for i in range(n):
        cnf.append([-x_id(i, 0)])
    cnf.append([-b_id(0)])


def max_allowed_transitions_constraint(G: nx.Graph, cnf, k):
    n = len(G)
    T_max = 2 * n + 1
    # At each time step, we have a set of t_id(i,t). We want at-most-k true.
    # Also, let's enforce at least one move per step to avoid trivialities.
    for t in range(T_max + 1):
        lits = [t_id(i, t) for i in range(n)]
        # Enforce at-most-k:
        enc = CardEnc.atmost(lits, bound=k, vpool=vpool, encoding=EncType.seqcounter)
        for c in enc.clauses:
            cnf.append(c)
        # Also enforce at least one person moves each step except maybe the last step (t=2*n+1):
        if t < T_max:
            # at-least-one:
            cnf.append(lits)


def individuals_transition_constraint(G: nx.Graph, cnf):
    n = len(G)
    T_max = 2 * n + 1
    for i in range(n):
        for t in range(T_max + 1):
            # Original shepherd-individual move constraints (difference)
            cnf.append([-t_id(i, t), -x_id(i, t), b_id(t)])
            cnf.append([-t_id(i, t), x_id(i, t), -b_id(t)])

            # Transition rules
            cnf.append([-x_id(i, t), -t_id(i, t), -x_id(i, t + 1)])
            cnf.append([-x_id(i, t), t_id(i, t), x_id(i, t + 1)])
            cnf.append([x_id(i, t), -t_id(i, t), x_id(i, t + 1)])
            cnf.append([x_id(i, t), t_id(i, t), -x_id(i, t + 1)])


#! To remove this in the rapport
def individual_cannot_be_on_both_sides_constraint(G: nx.Graph, cnf):
    pass


def individuals_conflict_not_allowed_without_shepard(G: nx.Graph, cnf):
    n = len(G)
    T_max = 2 * n + 1
    for t in range(T_max + 1):
        for v1, v2 in G.edges():
            cnf.append([-x_id(v1, t), -x_id(v2, t), b_id(t)])
            cnf.append([x_id(v1, t), x_id(v2, t), -b_id(t)])


def shepard_alternation_constraint(G: nx.Graph, cnf, k):
    # Shepherd must alternate sides each step
    n = len(G)
    T_max = 2 * n + 2
    for t in range(T_max):
        if t % 2 == 0:
            cnf.append([-b_id(t)])
        else:
            cnf.append([b_id(t)])


def gen_solution(G: nx.Graph, k: int):
    n = len(G)
    cnf = CNFPlus()

    start_constraint(G, cnf)
    goal_constraint(G, cnf)
    max_allowed_transitions_constraint(G, cnf, k)
    individuals_conflict_not_allowed_without_shepard(G, cnf)
    individuals_transition_constraint(G, cnf)
    shepard_alternation_constraint(G, cnf, k)

    s = Minicard()
    s.append_formula(cnf.clauses)
    sat = s.solve()
    if not sat:
        return None

    model = s.get_model()
    T_max = 2 * n + 2
    solution = []
    for t in range(T_max):
        left_side = set()
        right_side = set()
        for i in range(n):
            pos = model[x_id(i, t) - 1] > 0
            if pos:
                right_side.add(i)
            else:
                left_side.add(i)
        shepherd_pos = model[b_id(t) - 1] > 0
        solution.append((shepherd_pos, left_side, right_side))
    return solution


def find_alcuin_number(G: nx.Graph) -> int:
    n = len(G)
    for K in range(0, n + 1):
        sol = gen_solution(G, K)
        if sol is not None:
            return K
    return INFINITY


def gen_solution_cvalid(G: nx.Graph, k: int, c: int):
    n = len(G)
    cnf = CNFPlus()

    start_constraint(G, cnf)
    goal_constraint(G, cnf)
    max_allowed_transitions_constraint(G, cnf, k)
    individual_cannot_be_on_both_sides_constraint(G, cnf)
    individuals_conflict_not_allowed_without_shepard(G, cnf)
    individuals_transition_constraint(G, cnf)
    shepard_alternation_constraint(G, cnf, k)

    # c-valid compartments constraints:
    T_max = 2 * n + 1
    for t in range(T_max + 1):
        for i in range(n):
            clause_at_least_one = [-t_id(i, t)]
            for p in range(1, c + 1):
                cnf.append([-t_id(i, t), -y_id(i, t, p)])
                clause_at_least_one.append(y_id(i, t, p))
            cnf.append(clause_at_least_one)
            for p in range(1, c + 1):
                for q in range(p + 1, c + 1):
                    cnf.append([-y_id(i, t, p), -y_id(i, t, q)])
        for u, v in G.edges():
            for p in range(1, c + 1):
                cnf.append([-y_id(u, t, p), -y_id(v, t, p)])

    s = Minicard()
    s.append_formula(cnf.clauses)
    sat = s.solve()
    if not sat:
        return None
    model = s.get_model()

    T_sol = 2 * n + 2
    solution = []
    for t in range(T_sol):
        b = model[b_id(t) - 1] > 0
        left_side = set()
        right_side = set()
        for i in range(n):
            pos = model[x_id(i, t) - 1] > 0
            if pos:
                right_side.add(i)
            else:
                left_side.add(i)
        compartments = [[] for _ in range(c)]
        for i in range(n):
            if model[t_id(i, t) - 1] > 0:
                for p in range(1, c + 1):
                    if model[y_id(i, t, p) - 1] > 0:
                        compartments[p - 1].append(i)
                        break
        non_empty_compartments = tuple(set(comp) for comp in compartments if comp)
        solution.append((b, left_side, right_side, non_empty_compartments))
    return solution


def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    n = len(G)
    for K in range(0, n + 1):
        sol = gen_solution_cvalid(G, K, c)
        if sol is not None:
            return K
    return INFINITY
