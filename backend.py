from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from pyomo.environ import *
from pyomo.opt import TerminationCondition

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# STORAGE — filled by /upload_sheet from frontend
# --------------------------------------------------

EMPLOYEES = []          # individual employee records
FUNCTIONS = {}          # workload aggregation
CURRENT_HC = {}         # function-role headcount


# --------------------------------------------------
# FRONTEND MODELS
# --------------------------------------------------

class EmployeeSheet(BaseModel):
    name: str | None = ""
    code: str | None = ""
    func: str
    min: float
    max: float
    avg: float
    role: str | None = "Executive"   # designation → mapped in frontend


class OptimizeRequest(BaseModel):
    N_current: Dict[str, int]    # comes directly from Excel (auto HC)


# --------------------------------------------------
# API — store uploaded employee sheet
# --------------------------------------------------

@app.post("/upload_sheet")
def upload_sheet(emp: EmployeeSheet):

    EMPLOYEES.append(emp.dict())

    f = emp.func.strip()

    # ----------- workload aggregation ----------
    if f not in FUNCTIONS:
        FUNCTIONS[f] = dict(employees=0, min=0.0, max=0.0, avg=0.0)

    FUNCTIONS[f]["employees"] += 1
    FUNCTIONS[f]["min"] += emp.min
    FUNCTIONS[f]["max"] += emp.max
    FUNCTIONS[f]["avg"] += emp.avg

    return {"status": "stored", "function": f}


# --------------------------------------------------
# API — workload summary (used by frontend)
# --------------------------------------------------

@app.get("/workload")
def workload_summary():
    return {
        "employees": EMPLOYEES,
        "functions": FUNCTIONS,
        "current_hc": CURRENT_HC
    }


# --------------------------------------------------
# INTERNAL — Build model inputs (teacher framework)
# --------------------------------------------------

ROLES = ["Manager", "AsstManager", "Officer", "Executive"]

# HQ productivity norm
CAPACITY_HOURS = 6.5

# mild hierarchy weights (teacher suggestion)
ALPHA = {
    "Manager": 0.10,
    "AsstManager": 0.20,
    "Officer": 0.0,
    "Executive": 0.0,
}

# redeployment-friendly objective weights
PENALTY_REMOVE = {
    "Manager": 3.0,
    "AsstManager": 2.0,
    "Officer": 1.0,
    "Executive": 1.0,
}

BIG_M = 10_000.0  # avoids infeasibility


def build_model_inputs(N_current_req: Dict[str, int]):

    F = list(FUNCTIONS.keys())

    # total workload = SUM of employee avg hours in function
    W = {f: FUNCTIONS[f]["avg"] for f in F}

    # convert frontend HC dict → role matrix
    def build_matrix():
        out = {}
        for f in F:
            for r in ROLES:
                key = f"{f}|{r}"
                out[(f, r)] = int(N_current_req.get(key, 0))
        return out

    N_current = build_matrix()

    return F, ROLES, W, CAPACITY_HOURS, N_current


# --------------------------------------------------
# OPTIMIZATION ENGINE
# --------------------------------------------------

@app.post("/optimize")
def optimize(req: OptimizeRequest):

    F, R, W, C, N_current = build_model_inputs(req.N_current)

    m = ConcreteModel()

    m.F = Set(initialize=F)
    m.R = Set(initialize=R)

    m.W = Param(m.F, initialize=W)
    m.C = Param(initialize=C)

    def _N_init(m, f, r):
        return N_current.get((f, r), 0)

    m.N = Param(m.F, m.R, initialize=_N_init)

    m.alpha = Param(m.R, initialize=lambda m, r: ALPHA.get(r, 0.0))
    m.pen = Param(m.R, initialize=lambda m, r: PENALTY_REMOVE.get(r, 1.0))

    # decision variables
    m.x = Var(m.F, m.R, domain=NonNegativeIntegers)

    # shortage slack (hours/day)
    m.short = Var(m.F, domain=NonNegativeReals)

    def total_hc(m, f):
        return sum(m.x[f, r] for r in m.R)

    # -------- Constraint 1: Workload Coverage --------
    def workload_rule(m, f):
        return total_hc(m, f) * m.C + m.short[f] >= m.W[f]
    m.workload = Constraint(m.F, rule=workload_rule)

    # -------- Constraint 2: Upper Bound (no expansion) --------
    def upper_rule(m, f, r):
        return m.x[f, r] <= m.N[f, r]
    m.upper = Constraint(m.F, m.R, rule=upper_rule)

    # -------- Constraint 3: Mild hierarchy role share --------
    def role_share(m, f, r):
        if value(m.alpha[r]) <= 0:
            return Constraint.Skip
        return m.x[f, r] >= m.alpha[r] * total_hc(m, f)
    m.role_share = Constraint(m.F, m.R, rule=role_share)

    # -------- Objective: minimize redeployment --------
    def obj(m):
        removed = sum(
            m.pen[r] * (m.N[f, r] - m.x[f, r])
            for f in m.F for r in m.R
        )
        shortage = BIG_M * sum(m.short[f] for f in m.F)
        return removed + shortage

    m.obj = Objective(rule=obj, sense=minimize)

    solver = SolverFactory("glpk")
    results = solver.solve(m)

    if results.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        return {"status": "error", "message": "Model infeasible"}

    # -------- Build Output Table --------
    rows = []

    for f in m.F:

        capacity = sum(value(m.x[f, r]) for r in m.R) * value(m.C)

        for r in m.R:

            rows.append({
                "Function": f,
                "Role": r,

                # current manpower → from Excel
                "Current": int(value(m.N[f, r])),

                # optimal manpower → solver result
                "Optimal": int(value(m.x[f, r])),

                # redeployable manpower
                "Removed": int(value(m.N[f, r]) - value(m.x[f, r])),

                "Workload": round(value(m.W[f]), 2),
                "Capacity": round(capacity, 2),
                "Shortage": round(value(m.short[f]), 2),
            })

    return {"status": "ok", "rows": rows}
