import os
import time
import numpy as np
import mujoco
from mujoco import viewer
from mujoco import mjtGeom
import pinocchio as pin
import example_robot_data

# kako ce se brojevi ispisivati na konzoli, sa 5 decimala i bez naucnog zapisa
np.set_printoptions(precision=5, suppress=True)


# =========================
# 1) MuJoCo model
# =========================
XML_PATH = os.path.expanduser(
    "~/mujoco_menagerie/franka_emika_panda/panda_torque.xml"
)
model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
data_mj = mujoco.MjData(model_mj)


# =========================
# 2) Pinocchio model
# =========================
robot = example_robot_data.load("panda")
model_pin = robot.model
data_pin = model_pin.createData()

# gravitacija
model_pin.gravity.linear = np.array([0.0, 0.0, -9.81])

# -------------------------
# TCP parent frame + offset
# -------------------------
EE_FRAME_NAME = "panda_leftfinger"
ee_frame_id = model_pin.getFrameId(EE_FRAME_NAME)
# stavio sam ofstet posto panda?leftfinger nije tacno na vrhu prsta
TCP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.054], dtype=float)


# =========================
# 3) Pomoćne funkcije
# =========================
def update_pin(model, data, q_full, dq_full=None):
    """
    Ažurira forward kinematics i frame placements.
    Ako je dq_full prosleđen, računa FK i sa brzinama.
    """
    if dq_full is None:
        pin.forwardKinematics(model, data, q_full)
    else:
        pin.forwardKinematics(model, data, q_full, dq_full)
    pin.updateFramePlacements(model, data)


def skew(v):
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=float)


def get_ee_homogeneous_transform(model, data, q_full, frame_id, tcp_offset_local=None):
    update_pin(model, data, q_full)

    T = data.oMf[frame_id].homogeneous.copy()

    if tcp_offset_local is not None:
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ tcp_offset_local

    return T


def get_ee_position(model, data, q_full, frame_id, tcp_offset_local=None):
    T = get_ee_homogeneous_transform(model, data, q_full, frame_id, tcp_offset_local)
    return T[:3, 3].copy()


def get_ee_rotation(model, data, q_full, frame_id, tcp_offset_local=None):
    T = get_ee_homogeneous_transform(model, data, q_full, frame_id, tcp_offset_local)
    return T[:3, :3].copy()


def get_frame_jacobian_6d(model, data, q_full, frame_id, tcp_offset_local=None):
    """
    6D geometrijski Jakobijan 6x7 u LOCAL_WORLD_ALIGNED okviru.

    Redovi:
    - J6[:3, :]  -> linearni deo
    - J6[3:, :]  -> ugaoni deo
    """
    update_pin(model, data, q_full)

    J6 = pin.computeFrameJacobian(
        model,
        data,
        q_full,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )[:, :7].copy()

    if tcp_offset_local is not None:
        R = data.oMf[frame_id].rotation.copy()
        r_world = R @ tcp_offset_local
        J6[:3, :] = J6[:3, :] - skew(r_world) @ J6[3:, :]

    return J6


def numerical_jdot_times_v_6d(model, data, q_full, v_full, frame_id,
                              tcp_offset_local=None, eps=1e-6):
    """
    Numerička aproksimacija:
        Jdot(q,v) * v ≈ (J(q + eps*v) - J(q)) / eps * v
    za 6D Jakobijan.
    """
    J_now = get_frame_jacobian_6d(model, data, q_full, frame_id, tcp_offset_local)
    q_next = pin.integrate(model, q_full, v_full * eps)
    J_next = get_frame_jacobian_6d(model, data, q_next, frame_id, tcp_offset_local)
    Jdot = (J_next - J_now) / eps
    return Jdot @ v_full[:7]


def damped_pseudoinverse(J, damping=1e-2):
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))


def quintic_phase(T, t):
    """
    Kvintična faza sigma(t) između 0 i 1, zajedno sa izvodima.
    sigma(0)=0, sigma(T)=1
    dsigma(0)=dsigma(T)=0
    ddsigma(0)=ddsigma(T)=0
    """
    t = np.clip(t, 0.0, T)
    s = t / T

    sigma = 10 * s**3 - 15 * s**4 + 6 * s**5
    dsigma = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    ddsigma = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)

    return sigma, dsigma, ddsigma


def quintic_scalar_trajectory(L_total, T, t):
    """
    5th order scalar trajectory:
        L(0)=0, L(T)=L_total
        dL(0)=dL(T)=0
        ddL(0)=ddL(T)=0
    """
    sigma, dsigma, ddsigma = quintic_phase(T, t)

    L = L_total * sigma
    dL = L_total * dsigma
    ddL = L_total * ddsigma
    return L, dL, ddL


def axis_angle_from_rotation_world_consistent(R_start, R_goal):
    """
    Relativna rotacija:
        R_rel = R_start.T @ R_goal

    log3(R_rel) daje rotacioni vektor izražen u START (lokalnom) okviru.
    Zato vraćamo:
    - axis_local : osa u start/local okviru
    - axis_world : ista osa prebačena u world
    - theta      : ukupni ugao
    """
    R_rel = R_start.T @ R_goal
    rotvec = pin.log3(R_rel)
    theta = np.linalg.norm(rotvec)

    if theta < 1e-10:
        axis_local = np.array([1.0, 0.0, 0.0], dtype=float)
    else:

    #probati da se prvo prebaci u nulu rot vec pa tek onda da se iyuce osa
        axis_local = rotvec / theta

    axis_world = R_start @ axis_local
    return axis_local, axis_world, theta


def draw_trajectory_points(viewer_handle, actual_pts, desired_pts,
                           radius_actual=0.004, radius_desired=0.003):
    """
    Crta TCP putanje kao kuglice:
    - crveno = stvarna putanja
    - plavo  = željena putanja
    """
    max_geom = viewer_handle.user_scn.maxgeom
    viewer_handle.user_scn.ngeom = 0

    identity_mat = np.eye(3).reshape(-1)

    # prvo stvarna putanja
    for p in actual_pts:
        if viewer_handle.user_scn.ngeom >= max_geom:
            break

        g = viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mjtGeom.mjGEOM_SPHERE,
            np.array([radius_actual, radius_actual, radius_actual]),
            p,
            identity_mat,
            np.array([1.0, 0.0, 0.0, 1.0])   # crveno
        )
        viewer_handle.user_scn.ngeom += 1

    # onda željena putanja
    for p in desired_pts:
        if viewer_handle.user_scn.ngeom >= max_geom:
            break

        g = viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mjtGeom.mjGEOM_SPHERE,
            np.array([radius_desired, radius_desired, radius_desired]),
            p,
            identity_mat,
            np.array([0.0, 0.0, 1.0, 1.0])   # plavo
        )
        viewer_handle.user_scn.ngeom += 1


def rotation_matrix_from_z_axis(direction):
    """
    Pravi matricu rotacije tako da lokalna z-osa pokazuje u pravcu 'direction'.
    """
    z = np.asarray(direction, dtype=float)
    norm_z = np.linalg.norm(z)

    if norm_z < 1e-12:
        return np.eye(3)

    z = z / norm_z

    if abs(z[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)

    x = np.cross(tmp, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-12:
        return np.eye(3)
    x = x / x_norm

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    R = np.column_stack([x, y, z])
    return R


def add_capsule_between_points(viewer_handle, p0, p1, radius, rgba):
    """
    Crta jednu kapsulu između dve tačke bez mjv_connector.
    """
    if viewer_handle.user_scn.ngeom >= viewer_handle.user_scn.maxgeom:
        return

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    center = 0.5 * (p0 + p1)
    direction = p1 - p0
    length = np.linalg.norm(direction)

    if length < 1e-12:
        return

    R = rotation_matrix_from_z_axis(direction)

    g = viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom]

    mujoco.mjv_initGeom(
        g,
        mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.5 * length, 0.0], dtype=float),
        center,
        R.reshape(-1),
        np.array(rgba, dtype=float)
    )

    viewer_handle.user_scn.ngeom += 1


def draw_frame_axes_capsules(viewer_handle, T, axis_length=0.08, axis_radius=0.0025,
                             origin_radius=0.004):
    """
    Crta koordinatni sistem:
    - bela kuglica u originu
    - X/Y/Z ose kao tanke kapsule
    """
    origin = T[:3, 3]
    R = T[:3, :3]

    x_end = origin + axis_length * R[:, 0]
    y_end = origin + axis_length * R[:, 1]
    z_end = origin + axis_length * R[:, 2]

    identity_mat = np.eye(3).reshape(-1)

    if viewer_handle.user_scn.ngeom < viewer_handle.user_scn.maxgeom:
        g = viewer_handle.user_scn.geoms[viewer_handle.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mjtGeom.mjGEOM_SPHERE,
            np.array([origin_radius, origin_radius, origin_radius], dtype=float),
            origin,
            identity_mat,
            np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        )
        viewer_handle.user_scn.ngeom += 1

    add_capsule_between_points(viewer_handle, origin, x_end, axis_radius, [1.0, 0.0, 0.0, 1.0])
    add_capsule_between_points(viewer_handle, origin, y_end, axis_radius, [0.0, 1.0, 0.0, 1.0])
    add_capsule_between_points(viewer_handle, origin, z_end, axis_radius, [0.0, 0.0, 1.0, 1.0])


# =========================
# 4) Početna konfiguracija
# =========================
q_start_arm = np.array([-0.5, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0], dtype=float)

q_init_full = np.zeros(9)
q_init_full[:7] = q_start_arm
q_init_full[7:] = np.array([0.0001, 0.0001], dtype=float)

data_mj.qpos[:] = q_init_full
data_mj.qvel[:] = 0.0
mujoco.mj_forward(model_mj, data_mj)


# =========================
# 5) Target poze
# =========================
T_A = get_ee_homogeneous_transform(
    model_pin, data_pin, q_init_full, ee_frame_id, TCP_OFFSET_LOCAL
)

T_B = T_A.copy()
T_B[:3, 3] += np.array([0.00, 0.20, 0.00], dtype=float)

# početna orijentacija
R_A_des = T_A[:3, :3].copy()

# krajnja orijentacija
axis_world_B = np.array([1.0, 0.0, 0.0], dtype=float)
theta_B = np.deg2rad(10.0)
#exp3 je Rodrigezova formula / pretvara vektor rotacije u matricu rotacije
R_B_des = pin.exp3(axis_world_B * theta_B) @ R_A_des

T_A[:3, :3] = R_A_des
T_B[:3, :3] = R_B_des

print("T_A =\n", T_A)
print("T_B =\n", T_B)
print("p_A =", T_A[:3, 3])
print("p_B =", T_B[:3, 3])
print("R_A_des =\n", R_A_des)
print("R_B_des =\n", R_B_des)


# =========================
# 6) IDC gainovi (joint space)
# =========================
#Kp = np.diag([100, 100, 80, 60, 40, 30, 20]).astype(float)
#Kd = np.diag([20, 20, 16, 12, 8, 6, 4]).astype(float)
Kp = np.diag([100, 100, 80, 60, 40, 30, 20]).astype(float)
Kd = np.diag([30, 30, 24, 18, 12, 10, 8]).astype(float)

# =========================
# 7) Task-space gainovi
# =========================
# translacija
Kx_p = np.diag([8.0, 8.0, 8.0])
Kx_d = np.diag([4.0, 4.0, 4.0])

# orijentacija
Kr_p = np.diag([1.0, 1.0, 1.0])
Kr_d = np.diag([0.4, 0.4, 0.4])


# Jacobian damping 
jac_damping = 1e-2


# =========================
# 8) Parametri trajektorije
# =========================
T_segment = 4.0
num_repeats = 5
total_segments = 2 * num_repeats
total_time = total_segments * T_segment


# =========================
# 9) Referentna joint trajektorija
# =========================
q_ref_full = q_init_full.copy()
dq_ref_full = np.zeros(9)


# =========================
# 10) Liste za crtanje putanje
# =========================
actual_traj = []
desired_traj = []

draw_every_n_steps = 10
max_traj_points = 300


# =========================
# 11) Simulacija
# =========================
dt = model_mj.opt.timestep
step_count = 0

with viewer.launch_passive(model_mj, data_mj) as v:
    while v.is_running():

        # koristi simulaciono vreme
        t = data_mj.time

        # -----------------------------------
        # (A) Stvarno stanje iz MuJoCo
        # -----------------------------------
        q = data_mj.qpos.copy()
        dq = data_mj.qvel.copy()

        q_arm = q[:7]
        dq_arm = dq[:7]

        # -----------------------------------
        # (B) Aktivni segment
        # -----------------------------------
        if t >= total_time:
            segment = total_segments - 1
            t_local = T_segment
        else:
            segment = int(t // T_segment)
            t_local = t - segment * T_segment

        if segment % 2 == 0:
            T_start = T_A
            T_goal = T_B
        else:
            T_start = T_B
            T_goal = T_A

        x_start = T_start[:3, 3].copy()
        x_goal = T_goal[:3, 3].copy()

        R_start = T_start[:3, :3].copy()
        R_goal = T_goal[:3, :3].copy()

        # -----------------------------------
        # (C1) Pozicija
        # -----------------------------------
        #formiram vektor između krajnje i početne pozicije
        direction_vec = x_goal - x_start
        #određujem dužinu tog vektora - pređeni put
        path_length = np.linalg.norm(direction_vec)

        if path_length < 1e-10:
            raise ValueError("Dužina putanje je praktično nula.")
        
        #određujem ort vektor
        ort_vec = direction_vec / path_length

        #interpolacija pređenog puta
        L, dL, ddL = quintic_scalar_trajectory(path_length, T_segment, t_local)

        x_d = x_start + ort_vec * L
        dx_d = ort_vec * dL
        ddx_d = ort_vec * ddL

        # -----------------------------------
        # (C2) Orijentacija: world-konzistentna verzija
        # -----------------------------------
        axis_local, axis_world, theta_total = axis_angle_from_rotation_world_consistent(
            R_start, R_goal
        )

        theta_t = 0.0
        dtheta_t = 0.0
        ddtheta_t = 0.0

        if theta_total > 1e-10:
            theta_t, dtheta_t, ddtheta_t = quintic_scalar_trajectory(
                theta_total, T_segment, t_local
            )

        
        R_d = R_start @ pin.exp3(axis_local * theta_t)

        omega_d = axis_world * dtheta_t
        domega_d = axis_world * ddtheta_t

        #spajam zeljenu poyiciju i orijentaciju u matricu homogene transformacije
        T_d = np.eye(4)
        T_d[:3, :3] = R_d
        T_d[:3, 3] = x_d

        # -----------------------------------
        # (D) Stvarno EE stanje + 6D Jakobijan
        # -----------------------------------
        update_pin(model_pin, data_pin, q, dq)

        x_act = get_ee_position(model_pin, data_pin, q, ee_frame_id, TCP_OFFSET_LOCAL)
        R_act = get_ee_rotation(model_pin, data_pin, q, ee_frame_id, TCP_OFFSET_LOCAL)

        T_act = np.eye(4)
        T_act[:3, :3] = R_act
        T_act[:3, 3] = x_act

        J6 = get_frame_jacobian_6d(
            model_pin, data_pin, q, ee_frame_id, TCP_OFFSET_LOCAL
        )
        J6_pinv = damped_pseudoinverse(J6, damping=jac_damping)

        #racunam linearnu i ugaonu brzinu TCP-a
        v_ee = J6 @ dq_arm
        dx_act = v_ee[:3]
        omega_act = v_ee[3:]

        # Jdot(q,dq) * dq
        jdot_qdot_6d = numerical_jdot_times_v_6d(
            model_pin, data_pin, q, dq, ee_frame_id, TCP_OFFSET_LOCAL
        )

        # -----------------------------------
        # (E) Task-space feedback
        # -----------------------------------
        # pozicija
        pos_err = x_d - x_act
        vel_err = dx_d - dx_act

        ddx_cmd = ddx_d + Kx_d @ vel_err + Kx_p @ pos_err

        # orijentacija
        rot_err_local = pin.log3(R_act.T @ R_d)
        rot_err_world = R_act @ rot_err_local

        omega_err = omega_d - omega_act

        domega_cmd = domega_d + Kr_d @ omega_err + Kr_p @ rot_err_world


        #spajamo ddx_cmd i domega_cmd u a_task_cmd
        # definišemo težinsku matricu 
        W_task = np.diag([1.0, 1.0, 1.0, 0.25, 0.25, 0.25])
        #a_task_cmd = np.hstack([ddx_cmd, domega_cmd])
        a_task_cmd = W_task @ np.hstack([ddx_cmd, domega_cmd])

    
        # -----------------------------------
        # (F) Željeno joint ubrzanje
        # -----------------------------------
        #ddqd_cmd = J6_pinv @ (a_task_cmd - jdot_qdot_6d)
        ddqd_cmd = J6_pinv @ (a_task_cmd - W_task @ jdot_qdot_6d)

        # Integracija željene joint reference da od ddq dobije i dq i q
        dq_ref_full[:7] += ddqd_cmd * dt
        #dq_ref_full[7:] = 0.0
        dq_ref_full[:7] = np.clip(dq_ref_full[:7], -0.4, 0.4)

        q_ref_full = pin.integrate(model_pin, q_ref_full, dq_ref_full * dt)

        # prsti ostaju zatvoreni
        q_ref_full[7:] = np.array([0.0001, 0.0001], dtype=float)
        dq_ref_full[7:] = 0.0

        qd_arm = q_ref_full[:7].copy()
        dqd_arm = dq_ref_full[:7].copy()

        # -----------------------------------
        # (G) Dinamika iz Pinocchio
        # -----------------------------------
        q_pin = q.copy()
        v_pin = dq.copy()

        M = pin.crba(model_pin, data_pin, q_pin)
        M = 0.5 * (M + M.T)

        h = pin.nonLinearEffects(model_pin, data_pin, q_pin, v_pin)

        M_arm = M[:7, :7]
        h_arm = h[:7]

        # -----------------------------------
        # (H) IDC zakon
        # -----------------------------------
        e_q = qd_arm - q_arm
        e_dq = dqd_arm - dq_arm

        ddq_cmd_joint = ddqd_cmd + Kd @ e_dq + Kp @ e_q
        tau = M_arm @ ddq_cmd_joint + h_arm

        # opciono: saturacija momenta radi stabilnosti
        tau = np.clip(tau, -80.0, 80.0)

        # -----------------------------------
        # Pošalji momente
        # -----------------------------------
        data_mj.ctrl[:7] = tau

        if model_mj.nu > 7:
            data_mj.ctrl[7] = 0.0

        # -----------------------------------
        # Pamćenje tačaka putanje
        # -----------------------------------
        if step_count % draw_every_n_steps == 0:
            actual_traj.append(x_act.copy())
            desired_traj.append(x_d.copy())

            if len(actual_traj) > max_traj_points:
                actual_traj.pop(0)

            if len(desired_traj) > max_traj_points:
                desired_traj.pop(0)

        # -----------------------------------
        # Debug ispis
        # -----------------------------------
        if step_count % 200 == 0:
            print("\n========================")
            print(f"t = {t:.3f} s")
            print("segment =", segment, "t_local =", round(t_local, 3))

            print("x_d   =", x_d)
            print("x_act =", x_act)
            print("pos_err =", pos_err)

            print("theta_total =", theta_total)
            print("theta_t =", theta_t)
            print("dtheta_t =", dtheta_t)
            print("ddtheta_t =", ddtheta_t)

            print("axis_local =", axis_local)
            print("axis_world =", axis_world)

            print("omega_d =", omega_d)
            print("omega_act =", omega_act)
            print("omega_err =", omega_err)

            print("rot_err_local =", rot_err_local)
            print("rot_err_world =", rot_err_world)
            print("||rot_err_world|| =", np.linalg.norm(rot_err_world))

            print("ddx_cmd =", ddx_cmd)
            print("domega_cmd =", domega_cmd)

            print("q_arm =", q_arm)
            print("qd_arm =", qd_arm)
            print("tau =", tau)
            print("TCP_OFFSET_LOCAL =", TCP_OFFSET_LOCAL)

        # -----------------------------------
        # Korak simulacije
        # -----------------------------------
        mujoco.mj_step(model_mj, data_mj)

        # -----------------------------------
        # Crtanje putanje u viewer-u
        # -----------------------------------
        draw_trajectory_points(v, actual_traj, desired_traj)

        draw_frame_axes_capsules(v, T_A, axis_length=0.08, axis_radius=0.0025)
        draw_frame_axes_capsules(v, T_B, axis_length=0.08, axis_radius=0.0025)
        #draw_frame_axes_capsules(v, T_d, axis_length=0.05, axis_radius=0.0010)
        draw_frame_axes_capsules(v, T_act, axis_length=0.10, axis_radius=0.0020)
        #draw_frame_axes_capsules(v, T_d, axis_length=0.10, axis_radius=0.0020)

        v.sync()
        time.sleep(dt)
        step_count += 1