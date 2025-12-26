from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from pyomo.environ import *
from pyomo.opt import TerminationCondition

app = FastAPI()

# Allow frontend calls (Vercel / Local / Railway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# STORAGE — EMPLOYEE WORKLOAD DATA
# --------------------------------------------------

# Each upload_sheet() call stores one employee entry
EMPLOYEES = []

# Derived function-wise workload aggregation
FUNCTIONS = {}



# --------------------------------------------------
# MODELS / SCHEMAS
# --------------------------------------------------

class EmployeeSheet(BaseModel):
    name: str = ""
    code: str = ""
    func: str
    min: float
    max: float
    avg: float


class OptimizeRequest(BaseModel):
    N_current: Dict[str, int]



# --------------------------------------------------
# API — Upload Parsed Employee Sheet
# Called by frontend after Excel processing
# --------------------------------------------------

@app.post("/upload_sheet")
def upload_sheet(emp: EmployeeSheet):

    EMPLOYEES.append(emp.dict())

    func = emp.func.strip()

    if func not in FUNCTIONS:
        FUNCTIONS[func] = {
            "employees": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0
        }

    FUNCTIONS[func]["employees"] += 1
    FUNCTIONS[func]["min"] += emp.min
    FUNCTIONS[func]["max"] += emp.max
    FUNCTIONS[func]["avg"] += emp.avg

    return {"status": "stored", "function": func}



# --------------------------------------------------
# API — Return Function-wise Workload Summary
# Used to populate workload summary table
# --------------------------------------------------

@app.get("/workload")
def workload_summary():
    return {
        "employees": EMPLOYEES,
        "functions": FUNCTIONS
    }



# --------------------------------------------------
# INTERNAL — Build Optimization Payload
# --------------------------------------------------

def build_optimizer_inputs():

    F = list(FUNCTIONS.keys())

    # Workload = average hours/day (teacher rule)
    W = {f: FUNCTIONS[f]["avg"] for f in F}

    # Default roles (UI & model aligned)
    R = ["Manager", "AsstManager", "Officer", "Executive"]

    # Capacity (hrs/day per employee) — fixed assumption
    C = 6.5

    return F, R, W, C



# --------------------------------------------------
# API — OPTIMIZATION ENGINE
# Pyomo + GLPK
# --------------------------------------------------

@app.post("/optimize")
def optimize(req: OptimizeRequest):

    F, R, W, C = build_optimizer_inputs()

    m = ConcreteModel()

    m.F = Set(initialize=F)
    m.R = Set(initialize=R)

    m.W = Param(m.F, initialize=W)
    m.C = Param(initialize=C)

    # Current Headcount Matrix
    def _N_init(m, f, r):
        return req.N_current.get(f"{f}|{r}", 0)

    m.N = Param(m.F, m.R, initialize=_N_init)

    # Decision Variables
    m.x = Var(m.F, m.R, domain=NonNegativeIntegers)
    m.short = Var(m.F, domain=NonNegativeReals)

    # Total HC per function
    def total_hc(m, f):
        return sum(m.x[f, r] for r in m.R)

    # Workload Satisfaction
    def workload_rule(m, f):
        return total_hc(m, f) * m.C + m.short[f] >= m.W[f]

    m.workload = Constraint(m.F, rule=workload_rule)

    # Upper bound — cannot exceed current staff
    def upper_bound(m, f, r):
        return m.x[f, r] <= m.N[f, r]

    m.upper = Constraint(m.F, m.R, rule=upper_bound)

    # Objective: minimize removals + heavy penalty on shortage
    BIG_M = 10000

    def obj(m):
        removed = sum(
            (m.N[f, r] - m.x[f, r])
            for f in m.F for r in m.R
        )
        shortage = BIG_M * sum(m.short[f] for f in m.F)
        return removed + shortage

    m.obj = Objective(rule=obj, sense=minimize)

    # Solve Model
    solver = SolverFactory("glpk")
    results = solver.solve(m)

    if results.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        return {"status": "error", "message": "Model infeasible"}


    # Build Response Table
    rows = []

    for f in m.F:

        capacity = sum(value(m.x[f, r]) for r in m.R) * value(m.C)

        for r in m.R:

            rows.append({
                "Function": f,
                "Role": r,
                "Current": int(value(m.N[f, r])),
                "Optimal": int(value(m.x[f, r])),
                "Removed": int(value(m.N[f, r]) - value(m.x[f, r])),
                "Workload": round(value(m.W[f]), 2),
                "Capacity": round(capacity, 2),
                "Shortage": round(value(m.short[f]), 2)
            })

    return {"status": "ok", "rows": rows}
