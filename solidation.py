# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
from tkinter import N
from matplotlib.ft2font import KERNING_DEFAULT
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

maximum_step = 1024
n_particles = 64
square_size = 0.05
x_offset = 0.3
y_offset = 0.9

printer_x = ti.field(float, ())
printer_v = 0.005
printer_size = 0.05
printer_particles = 256
printer = ti.Vector.field(2, float, printer_particles)


n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())
bound = 3
E = 400

cooldown = 0.015
t = ti.field(float, ())
pretime = ti.field(float, ())

inv_dx = float(n_grid)
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

solid_particles = ti.field(int, ())
solid_x = ti.Vector.field(2, float, n_particles*maximum_step)
time_stamp = ti.field(int, ())
window_size = 256
start_time_window = ti.field(float, window_size)

affine_window = ti.Matrix.field(2, 2, float)
position_window = ti.Vector.field(2, float)
v_window = ti.Vector.field(2, float)
F_window = ti.Matrix.field(2, 2, float)
Jp_window = ti.field(float)
C_window = ti.Matrix.field(2, 2, float)
window = ti.root.pointer(ti.ij,(window_size,n_particles))
# window = ti.root.dynamic(ti.i,window_size).dynamic(ti.j,n_particles)
window.place(position_window, affine_window, v_window, F_window, Jp_window, C_window)


boundary_length = 1000
boundary = ti.field(float, boundary_length)


# Implement with new data structure
@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # STEP 1: P2G
    for index, p in position_window:
        Xp = position_window[index,p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F_window[index, p] = (ti.Matrix.identity(float, 2) + dt * C_window[index,p]) @ F_window[index, p]
        mu = 500.0
        U, sig, V = ti.svd(F_window[index, p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            Jp_window[index, p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        stress0 = 2 * mu * (F_window[index, p] - U @ V.transpose()) @ F_window[index, p].transpose() + ti.Matrix.identity(float, 2) * lambda_0 * J * (J - 1)
        stress0 = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress0
        affine0 = stress0 + p_mass * C_window[index,p]

        mu = 0.01
        stress1 = 2 * mu * (F_window[index, p] - U @ V.transpose()) @ F_window[index, p].transpose() + ti.Matrix.identity(float, 2) * lambda_0 * J * (J - 1)
        stress1 = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress1
        affine1 = stress1 + p_mass * C_window[index,p]
        duration = ti.min(t[None]-start_time_window[index],15)
        cur_ratio = 1.0/(1.0+ti.exp(-duration+10.0))
        # cur_ratio = 0.5

        affine_window[index,p] = affine0*cur_ratio + affine1*(1.0-cur_ratio)

        for i, j in ti.static(ti.ndrange(3,3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            v_f = p_mass * v_window[index,p] + affine_window[index,p]@dpos
            grid_v[base + offset] += weight * v_f
            grid_m[base + offset] += weight * p_mass

    # STEP 2: Normalize Grid    
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        dist = attractor_pos[None] - dx * ti.Vector([i, j])
        grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 10
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0

    # STEP 3: G2P
    for index, p in position_window:
        Xp = position_window[index,p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v_window[index,p] = new_v
        #! ADD Boundary conditions based on list
        xind = int(position_window[index,p].x * boundary_length - 0.5)
        if position_window[index,p].y < boundary[xind]:
            v_window[index,p].y = 0.0
            v_window[index,p].x = 0.0
        #!
        position_window[index,p] += dt * v_window[index,p]
        C_window[index,p] = new_C

# will only be called one time
@ti.kernel
def start():
    t[None] = 0.0
    solid_particles[None] = 0
    time_stamp[None] = 0
    pretime[None] = 0.0
    printer_x[None] = 0.0

# will be called n times
@ti.kernel
def init():
    q_ind = time_stamp[None] % window_size
    start_time_window[q_ind] = t[None]
    for i in range(n_particles):
        position_window[q_ind,i] = [ti.random() * square_size + printer_x[None], ti.random() * square_size + y_offset + printer_size - square_size]
        v_window[q_ind,i] = [0, 0]
        # J[i] = 1
        F_window[q_ind, i] = ti.Matrix([[1, 0], [0, 1]])
        # Jp_window[q_ind, i] = 1


# will be called n-1 times when append new fluid
@ti.kernel
def solid_accumulate():
    q_ind = time_stamp[None] % window_size
    for i in range(n_particles):
        solid_x[i+solid_particles[None]] = position_window[q_ind,i]
        xind = int(position_window[q_ind,i].x * boundary_length - 0.5)
        boundary[xind] = max(boundary[xind], position_window[q_ind,i].y)
    solid_particles[None] += n_particles

def accumulate():
    time_stamp[None] += 1
    if time_stamp[None] >= window_size:
        solid_accumulate()
    init()

@ti.kernel
def printer_render():
    for p in printer:
        printer[p] = [ti.random() * printer_size + printer_x[None], ti.random() * printer_size + y_offset + printer_size]


start()
init()
gui = ti.GUI('MPM88')
while gui.running:
    t[None] += 1e-2
    if ti.sin(t[None]*np.pi*printer_v/0.0095) > 0:
        printer_x[None] += printer_v
    else:
        printer_x[None] -= printer_v
    # ratio[None] = 1.0/(1.0+ti.exp(-t[None]+10.0))
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        # elif e.key == 'r':
        #     start()
        #     init()
        elif e.key in ('p'):
            print(t[None])
            print(printer_x[None])
        elif e.key == 'a':
            accumulate()

    mouse_pos = gui.get_cursor_pos()
    attractor_pos[None] = mouse_pos
    attractor_strength[None] = (gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB))*2.0

    if t[None] - pretime[None] > cooldown:
        accumulate()
        pretime[None] = t[None]

    for s in range(50):
        printer_render()
        substep()

    gui.clear(0x112F41)
    A = position_window.to_numpy().reshape(window_size*n_particles,2)
    A[np.isnan(A)] = 0
    gui.circles(A, radius=1.5, color=0x068587)
    gui.circles(solid_x.to_numpy(), radius=1.5, color=0xED553B)
    gui.circles(printer.to_numpy(), radius=1.5, color=0xFF94E6)
    gui.circle(mouse_pos, radius=15, color=0x336699)
    gui.show()