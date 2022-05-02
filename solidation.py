# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti

ti.init(arch=ti.gpu)

n_particles = 8192
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

t = ti.field(float, ())
ratio = ti.field(float, ())
inv_dx = float(n_grid)
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
F = ti.Matrix.field(2, 2, dtype=float,shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation


x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
# J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

all_particles = ti.field(int, ())
total_x = ti.Vector.field(2, float, n_particles*10)
total_affine = ti.Matrix.field(2, 2, float, n_particles*10)
cur_affine = ti.Matrix.field(2, 2, float, n_particles)



@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
# ? edit begin, solid algorithms
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] 
        # deformation gradient update
        # h = ti.exp(
        #     10 *
        #     (1.0 -
        #      Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        # mu, la = mu_0 * h, lambda_0 * h
        #! mu = 0.0
        mu = 500.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            # if material[p] == 2:  # Snow
            #     new_sig = min(max(sig[d, d], 1 - 2.5e-2),
            #                   1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        
        # F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        temp = []
        stress0 = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * lambda_0 * J * (J - 1)
        stress0 = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress0
        affine0 = stress0 + p_mass * C[p]
        temp.append(affine0)

        mu = 0.1
        stress1 = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * lambda_0 * J * (J - 1)
        stress1 = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress1
        affine1 = stress1 + p_mass * C[p]
        temp.append(affine1)

        affine = affine0*ratio[None] + affine1*(1.0-ratio[None])
        cur_affine[p] = affine
# ? edit end
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            # ratio = 0.9
            v0 = p_mass * v[p] + affine0@dpos
            v1 = p_mass * v[p] + affine1@dpos 
            v_f = v0*(ratio[None]) + v1*(1.0-ratio[None])
            v_f = p_mass * v[p] + affine@dpos
            # grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_v[base + offset] += weight * v_f 
            grid_m[base + offset] += weight * p_mass
    
    # Add boundary particles
    for p in range(all_particles[None]):
        Xp = total_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        affine = total_affine[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * affine@dpos
            grid_m[base + offset] += weight * p_mass

    # projection & gravity
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

    for p in x:
        Xp = x[p] / dx
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
        v[p] = new_v
        x[p] += dt * v[p]
        # J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

@ti.kernel
def start():
    t[None] = 0.0
    ratio[None] = 0.0
    all_particles[None] = 0


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.3 + 0.3, ti.random() * 0.3 + 0.65]
        v[i] = [0, -1]
        # J[i] = 1
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1


@ti.kernel
def accumulate():
    t[None] = 0.0
    for i in range(n_particles):
        total_x[i+all_particles[None]] = x[i]
        total_affine[i+all_particles[None]] = cur_affine[i]

    all_particles[None] += n_particles

start()
init()
gui = ti.GUI('MPM88')
while gui.running:
    t[None] += 1e-2
    ratio[None] = 1.0/(1.0+ti.exp(-t[None]+10.0))
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            start()
            init()
        elif e.key in ('p'):
            print(t[None],ratio[None])
        elif e.key == 'a':
            accumulate()
            init()
    mouse_pos = gui.get_cursor_pos()
    attractor_pos[None] = mouse_pos
    attractor_strength[None] = (gui.is_pressed(gui.LMB) - gui.is_pressed(gui.RMB))*2.0
    for s in range(50):
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.circles(total_x.to_numpy(), radius=1.5, color=0xED553B)
    gui.circle(mouse_pos, radius=15, color=0x336699)
    gui.show()