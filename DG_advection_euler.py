from firedrake import *
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

T = 2*math.pi
mesh_dx = 40
dt = T/300
errs = []
dts = []

for i in range(4):
    mesh = UnitSquareMesh(mesh_dx, mesh_dx, quadrilateral=True)

    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)

    x, y = SpatialCoordinate(mesh)

    velocity = as_vector((0.5 - y, x - 0.5))
    u = Function(W).interpolate(velocity)

    bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
    cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
    cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
    slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

    bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
    slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                0.0, 1.0), 0.0)

    q = Function(V).interpolate(1.0 + bell + cone + slot_cyl)
    q_init = Function(V).assign(q)


    qs = []

    dtc = Constant(dt)
    q_in = Constant(1.0)

    dq_trial = TrialFunction(V)
    n = FacetNormal(mesh)
    un = 0.5*(dot(u, n) + abs(dot(u, n)))
    phi = TestFunction(V)
    a = phi*dq_trial*dx + dtc * (-dq_trial*div(phi*u)*dx 
                                + conditional(dot(u, n) > 0, phi*dot(u, n)*dq_trial, 0.0)*ds
                                + (phi('+') - phi('-'))*(un('+')*dq_trial('+') - un('-')*dq_trial('-'))*dS)

    L1 = phi*(q)*dx - dtc * (conditional(dot(u,n) < 0, phi*dot(u, n)*q_in, 0.0)*ds)

    dq = Function(V)
    params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    prob1 = LinearVariationalProblem(a, L1, dq)
    solv1 = LinearVariationalSolver(prob1)


    t = 0.0
    step = 0 
    output_freq = 20

    while t < T - 0.5*dt:
        solv1.solve()
        q.assign(dq)

        step += 1
        t += dt

        if step % output_freq == 0:
            qs.append(q.copy(deepcopy=True))
            print(f"t= {t}")

    L2_err = sqrt(assemble((q-q_init)*(q - q_init) * dx))
    L2_init = sqrt(assemble(q_init*q_init*dx))
    errs.append(L2_err/L2_init)
    dts.append(dt)
    dt = dt/2
    dtc.assign(dt)
    mesh_dx *= 2
    t = 0.0
    step = 0
    q.assign(q_init)

    """
    nsp = 16
    fn_plotter = FunctionPlotter(mesh, num_sample_points=nsp)

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(q_init, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
    fig.colorbar(colors)

    def animate(q):
        colors.set_array(fn_plotter(q))

    interval = 1e3 * output_freq * dt
    animation = FuncAnimation(fig, animate, frames=qs, interval=interval)
    try:
        animation.save("DG_advection_euler.mp4", writer="ffmpeg")
    except:
        print("Failed to write movie! Try installing 'ffmpeg'")
    """
plt.loglog(dts, errs, marker='o')
plt.xlabel("Time step size dt")
plt.ylabel("Normalized L2 error")
plt.title("Convergence of backward Euler DG advection")
plt.grid(True)
plt.show()