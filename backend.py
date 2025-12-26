from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyomo.environ import *
from pyomo.opt import TerminationCondition
from typing import List, Dict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Request Model (Inputs same as Excel feed) --------

class OptimizeRequest(BaseModel):
    F: List[str]                     # Functions
    R: List[str]                     # Roles
    W: Dict[str, float]              # Workload per function
    C: float                         # Productive hours per person per day
    N_current: Dict[str, int]        # Current headcount
    alpha: Dict[str, float] = {}     # Minimum role share weights
    penalty: Dict[str, float]        # Removal penalty weights


# -------------------- OPTIMIZATION MODEL --------------------

@app.post("/optimize")
def optimize(req: OptimizeRequest):

    m = ConcreteModel()

    # Sets
    m.F = Set(initialize=req.F)
    m.R = Set(initialize=req.R)

    # Parameters
    m.W = Param(m.F, initialize=req.W)
    m.C = Param(initialize=req.C)

    def _N_init(m, f, r):
        return req.N_current.get(f"{f}|{r}", 0)

    m.N = Param(m.F, m.R, initialize=_N_init)

    m.alpha = Param(m.R, initialize=lambda m, r: req.alpha.get(r, 0.0))
    m.pen = Param(m.R, initialize=lambda m, r: req.penalty.get(r, 1.0))

    # Decision variables
    m.x = Var(m.F, m.R, domain=NonNegativeIntegers)

    # Shortage slack (should be 0 if feasible)
    m.short = Var(m.F, domain=NonNegativeReals)

    def total_hc(m, f):
        return sum(m.x[f, r] for r in m.R)

    # -------- Constraint 1: Workload Coverage --------
    def workload_cover(m, f):
        return total_hc(m, f) * m.C + m.short[f] >= m.W[f]
    m.workload_cover = Constraint(m.F, rule=workload_cover)

    # -------- Constraint 2: Headcount Upper Bound --------
    def upper_bound(m, f, r):
        return m.x[f, r] <= m.N[f, r]
    m.upper_bound = Constraint(m.F, m.R, rule=upper_bound)

    # -------- Constraint 3: Minimum Role Share (Hierarchy) --------
    def role_share(m, f, r):
        if value(m.alpha[r]) <= 0:
            return Constraint.Skip
        return m.x[f, r] >= m.alpha[r] * total_hc(m, f)
    m.role_share = Constraint(m.F, m.R, rule=role_share)

    # -------- Objective: Minimize Removals + Big Penalty on Shortage --------
    BIG_M = 10_000.0

    def obj_rule(m):
        removed_cost = sum(
            m.pen[r] * (m.N[f, r] - m.x[f, r])
            for f in m.F for r in m.R
        )
        shortage_cost = BIG_M * sum(m.short[f] for f in m.F)
        return removed_cost + shortage_cost

    m.obj = Objective(rule=obj_rule, sense=minimize)

    # -------- Solve --------
    solver = SolverFactory("glpk")
    results = solver.solve(m)

    if results.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible
    ):
        return {"status": "error", "message": "Model infeasible"}

    # -------- Output --------
    rows = []

    for f in m.F:
        cur_total = sum(value(m.N[f, r]) for r in m.R)
        opt_total = sum(value(m.x[f, r]) for r in m.R)

        workload = value(m.W[f])
        cap = opt_total * value(m.C)
        shortage = value(m.short[f])

        for r in m.R:
            rows.append({
                "Function": f,
                "Role": r,
                "Current": int(value(m.N[f, r])),
                "Optimal": int(value(m.x[f, r])),
                "Removed": int(value(m.N[f, r]) - value(m.x[f, r])),
                "Workload": round(workload, 2),
                "Capacity": round(cap, 2),
                "Shortage": round(shortage, 2)
            })

    return {
        "status": "ok",
        "rows": rows
    }
