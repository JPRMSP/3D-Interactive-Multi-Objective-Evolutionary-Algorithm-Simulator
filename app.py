import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ------------------------
# Interactive 3D MOEA Simulator
# ------------------------
st.set_page_config(page_title="3D MOEA Simulator", layout="wide")
st.title("3D Interactive Multi-Objective Evolutionary Algorithm Simulator")

# Sidebar - Algorithm Parameters
st.sidebar.header("Algorithm Parameters")
algorithm = st.sidebar.selectbox("Select MOEA Algorithm", ["NSGA-II", "VEGA", "SPEA2", "Random Weighted GA"])
population_size = st.sidebar.slider("Population Size", 10, 200, 50)
generations = st.sidebar.slider("Generations", 10, 100, 30)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)

# Sidebar - Define custom 2-variable objectives
st.sidebar.header("Define Objectives (2 Variables)")
a1 = st.sidebar.slider("Objective 1: f1(x,y) = x^2 + a1*y^2", 0.0, 5.0, 1.0)
a2 = st.sidebar.slider("Objective 2: f2(x,y) = (x-2)^2 + a2*(y-2)^2", 0.0, 5.0, 1.0)

f1 = lambda x, y: x**2 + a1*y**2
f2 = lambda x, y: (x-2)**2 + a2*(y-2)**2

# Initialize 2D population
pop = np.random.uniform(-5, 5, (population_size, 2))

# Non-dominated Sorting
def non_dominated_sort(f1_vals, f2_vals):
    n = len(f1_vals)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (f1_vals[j] <= f1_vals[i] and f2_vals[j] <= f2_vals[i]) and (f1_vals[j] < f1_vals[i] or f2_vals[j] < f2_vals[i]):
                dominated[i] = True
    return ~dominated

# GA Operators
def mutate(xy):
    return xy + np.random.normal(0, 0.5, size=2)

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha*p1 + (1-alpha)*p2

# Evolution Loop
pareto_fronts = []
for gen in range(generations):
    f1_vals = np.array([f1(ind[0], ind[1]) for ind in pop])
    f2_vals = np.array([f2(ind[0], ind[1]) for ind in pop])
    
    nd_mask = non_dominated_sort(f1_vals, f2_vals)
    pareto_fronts.append(np.array([f1_vals[nd_mask], f2_vals[nd_mask], np.zeros(sum(nd_mask))]).T)  # z=0 for visualization

    # Elitism
    elites = pop[nd_mask]
    
    # Generate new population
    new_pop = []
    while len(new_pop) < population_size:
        parents = elites[np.random.choice(len(elites), 2, replace=True)]
        child = crossover(parents[0], parents[1])
        if np.random.rand() < mutation_rate:
            child = mutate(child)
        new_pop.append(child)
    pop = np.array(new_pop)

# ------------------------
# Plotting 3D Pareto Front Evolution
# ------------------------
fig = go.Figure()

colors = np.linspace(0, 1, generations)
for i, front in enumerate(pareto_fronts):
    fig.add_trace(go.Scatter3d(
        x=front[:,0],
        y=front[:,1],
        z=front[:,2]+i*0.05,  # small z-offset for generations
        mode='markers',
        marker=dict(size=4, color=colors[i], colorscale='Viridis'),
        name=f"Gen {i+1}" if i % max(1, generations//10) == 0 else None
    ))

fig.update_layout(scene=dict(
    xaxis_title='Objective 1',
    yaxis_title='Objective 2',
    zaxis_title='Generation'
), title=f"Pareto Front Evolution - {algorithm}", showlegend=True)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Final Pareto Front (Objective 1 vs Objective 2)")
final_front = pareto_fronts[-1][:,:2]
st.dataframe(final_front)
