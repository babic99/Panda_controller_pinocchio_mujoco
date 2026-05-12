import os
import time
import numpy as np
import mujoco
from mujoco import viewer
from mujoco import mjtGeom
import pinocchio as pin
import example_robot_data
import matplotlib.pyplot as plt

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
# 1b) Spoljašnja  sila
# =========================
FORCE_BODY_NAME = "hand"

force_body_id = mujoco.mj_name2id(
    model_mj,
    mujoco.mjtObj.mjOBJ_BODY,
    FORCE_BODY_NAME
)

if force_body_id == -1:
    print("\nDostupna MuJoCo tela:")
    for i in range(model_mj.nbody):
        print(i, mujoco.mj_id2name(model_mj, mujoco.mjtObj.mjOBJ_BODY, i))
    raise ValueError(f"Body '{FORCE_BODY_NAME}' nije pronađen.")

force_start_time = 2.5
force_end_time = 2.8

# sila u world koordinatnom sistemu [Fx, Fy, Fz]
external_force_world = np.array([10.0, 0.0, 0.0], dtype=float)


# =========================
# 2) Pinocchio model
# =========================
robot = example_robot_data.load("panda")
model_pin = robot.model
data_pin = model_pin.createData()

model_pin.gravity.linear = np.array([0.0, 0.0, -9.81])


# =========================
# 3) TCP podešavanje
# =========================
EE_FRAME_NAME = "panda_leftfinger"
ee_frame_id = model_pin.getFrameId(EE_FRAME_NAME)

TCP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.054], dtype=float)


# =========================
# 4) Pomoćne funkcije
# =========================
def update_pin(model, data, q_full, dq_full=None):
    if dq_full is None:
        pin.forwardKinematics(model, data, q_full)
    else:
        pin.forwardKinematics(model, data, q_full, dq_full)

    pin.updateFramePlacements(model, data)


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def get_ee_homogeneous_transform(model, data, q_full, frame_id, tcp_offset_local=None):
    update_pin(model, data, q_full)

    T = data.oMf[frame_id].homogeneous.copy()

    if tcp_offset_local is not None:
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ tcp_offset_local

    return T


def get_frame_jacobian_6d(model, data, q_full, frame_id, tcp_offset_local=None):
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


def damped_pseudoinverse(J, damping=1e-3):
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(m))


def quintic_phase(T, t):
    t = np.clip(t, 0.0, T)
    s = t / T

    sigma = 10 * s**3 - 15 * s**4 + 6 * s**5
    dsigma = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    ddsigma = (60 * s - 180 * s**2 + 120 * s**3) / T**2

    return sigma, dsigma, ddsigma


def quintic_scalar_trajectory(L_total, T, t):
    sigma, dsigma, ddsigma = quintic_phase(T, t)

    L = L_total * sigma
    dL = L_total * dsigma
    ddL = L_total * ddsigma

    return L, dL, ddL


def axis_angle_from_rotation_world_consistent(R_start, R_goal):
    R_rel = R_start.T @ R_goal
    rotvec = pin.log3(R_rel)

    theta = np.linalg.norm(rotvec)

    if theta < 1e-10:
        axis_local = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis_local = rotvec / theta

    axis_world = R_start @ axis_local

    return axis_local, axis_world, theta


def inverse_kinematics_pose(
    model,
    data,
    q_init,
    frame_id,
    T_des,
    tcp_offset_local=None,
    max_iters=80,
    eps=1e-5,
    alpha=0.4,
    damping=1e-3
):
    q_ik = q_init.copy()
    q_ik[7:] = np.array([0.0001, 0.0001], dtype=float)

    for i in range(max_iters):
        T_act = get_ee_homogeneous_transform(
            model,
            data,
            q_ik,
            frame_id,
            tcp_offset_local
        )

        x_act = T_act[:3, 3]
        R_act = T_act[:3, :3]

        x_des = T_des[:3, 3]
        R_des = T_des[:3, :3]

        e_pos = x_des - x_act

        e_rot_local = pin.log3(R_act.T @ R_des)
        e_rot_world = R_act @ e_rot_local

        e = np.hstack([e_pos, e_rot_world])

        if np.linalg.norm(e) < eps:
            break

        J6 = get_frame_jacobian_6d(
            model,
            data,
            q_ik,
            frame_id,
            tcp_offset_local
        )

        J6_pinv = damped_pseudoinverse(J6, damping=damping)

        dq_arm = alpha * (J6_pinv @ e)

        dq_full = np.zeros(model.nv)
        dq_full[:7] = dq_arm

        q_ik = pin.integrate(model, q_ik, dq_full)
        q_ik[7:] = np.array([0.0001, 0.0001], dtype=float)

    return q_ik


# =========================
# 5) Funkcije za crtanje
# =========================
def rotation_matrix_from_z_axis(direction):
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

    return np.column_stack([x, y, z])


def add_capsule_between_points(viewer_handle, p0, p1, radius, rgba):
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


def draw_frame_axes_capsules(
    viewer_handle,
    T,
    axis_length=0.08,
    axis_radius=0.0025,
    origin_radius=0.004
):
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
            np.array([1.0, 1.0, 1.0, 1.0])
        )

        viewer_handle.user_scn.ngeom += 1

    add_capsule_between_points(viewer_handle, origin, x_end, axis_radius, [1.0, 0.0, 0.0, 1.0])
    add_capsule_between_points(viewer_handle, origin, y_end, axis_radius, [0.0, 1.0, 0.0, 1.0])
    add_capsule_between_points(viewer_handle, origin, z_end, axis_radius, [0.0, 0.0, 1.0, 1.0])


def draw_trajectory_points(
    viewer_handle,
    actual_pts,
    desired_pts,
    radius_actual=0.004,
    radius_desired=0.003
):
    max_geom = viewer_handle.user_scn.maxgeom
    viewer_handle.user_scn.ngeom = 0

    identity_mat = np.eye(3).reshape(-1)

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
            np.array([1.0, 0.0, 0.0, 1.0])
        )

        viewer_handle.user_scn.ngeom += 1

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
            np.array([0.0, 0.0, 1.0, 1.0])
        )

        viewer_handle.user_scn.ngeom += 1


# =========================
# 6) Početna konfiguracija
# =========================
q_start_arm = np.array(
    [-0.5, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0],
    dtype=float
)

q_init_full = np.zeros(9)
q_init_full[:7] = q_start_arm
q_init_full[7:] = np.array([0.0001, 0.0001], dtype=float)

data_mj.qpos[:] = q_init_full
data_mj.qvel[:] = 0.0

mujoco.mj_forward(model_mj, data_mj)


# =========================
# 7) Target poze
# =========================
T_A = get_ee_homogeneous_transform(
    model_pin,
    data_pin,
    q_init_full,
    ee_frame_id,
    TCP_OFFSET_LOCAL
)

T_B = T_A.copy()
T_B[:3, 3] += np.array([0.00, 0.20, 0.00], dtype=float)

R_A_des = T_A[:3, :3].copy()

axis_world_B = np.array([1.0, 0.0, 0.0], dtype=float)
theta_B = np.deg2rad(40.0)

R_B_des = pin.exp3(axis_world_B * theta_B) @ R_A_des

T_A[:3, :3] = R_A_des
T_B[:3, :3] = R_B_des

print("T_A =\n", T_A)
print("T_B =\n", T_B)


# =========================
# 8) IDC gainovi
# =========================
#Kp = np.diag([100, 100, 80, 60, 40, 30, 20]).astype(float)
#Kd = np.diag([30, 30, 24, 18, 12, 10, 8]).astype(float)
Kp = np.diag([100, 100, 80, 60, 40, 30, 20]).astype(float)
Kd = np.diag([35, 35, 28, 22, 16, 12, 10]).astype(float)


# =========================
# 9) Parametri trajektorije
# =========================
T_motion = 5.0      # vreme kretanja A -> B ili B -> A
T_settle = 2.0      # vreme zadržavanja u targetu
T_segment = T_motion + T_settle

num_repeats = 5

total_segments = 2 * num_repeats
total_time = total_segments * T_segment


# =========================
# 10) Reference za IDC
# =========================
q_ref_full = q_init_full.copy()
dq_ref_full = np.zeros(9)
ddq_ref_arm = np.zeros(7)


# =========================
# 11) Putanje za crtanje
# =========================
actual_traj = []
desired_traj = []

draw_every_n_steps = 10
max_traj_points = 300

# =========================
# 11b) Logovi za grafike
# =========================
time_log = []

x_des_log = []
x_act_log = []

rot_des_log = []
rot_act_log = []

pos_err_norm_log = []
rot_err_norm_log = []

external_force_log = []


# =========================
# 12) Simulacija
# =========================
dt = model_mj.opt.timestep
step_count = 0

with viewer.launch_passive(model_mj, data_mj) as v:

    while v.is_running():

        t = data_mj.time

        # -----------------------------------
        # Stvarno stanje iz MuJoCo
        # -----------------------------------
        q = data_mj.qpos.copy()
        dq = data_mj.qvel.copy()

        q_arm = q[:7]
        dq_arm = dq[:7]

        # -----------------------------------
        # Aktivni segment:
        # segment traje T_motion + T_settle
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
        # Sinteza pozicije TCP-a
        # Prvih T_motion sekundi se kreće,
        # zatim T_settle sekundi drži target.
        # -----------------------------------
        direction_vec = x_goal - x_start
        path_length = np.linalg.norm(direction_vec)

        if path_length < 1e-10:
            raise ValueError("Dužina putanje je praktično nula.")

        ort_vec = direction_vec / path_length

        if t_local <= T_motion:
            L, dL, ddL = quintic_scalar_trajectory(
                path_length,
                T_motion,
                t_local
            )

            x_d = x_start + ort_vec * L

            phase_name = "motion"

        else:
            L = path_length
            dL = 0.0
            ddL = 0.0

            x_d = x_goal.copy()

            phase_name = "settle"

        # -----------------------------------
        # Sinteza orijentacije TCP-a
        # -----------------------------------
        axis_local, axis_world, theta_total = axis_angle_from_rotation_world_consistent(
            R_start,
            R_goal
        )

        theta_t = 0.0
        dtheta_t = 0.0
        ddtheta_t = 0.0

        if theta_total > 1e-10:
            if t_local <= T_motion:
                theta_t, dtheta_t, ddtheta_t = quintic_scalar_trajectory(
                    theta_total,
                    T_motion,
                    t_local
                )
            else:
                theta_t = theta_total
                dtheta_t = 0.0
                ddtheta_t = 0.0

        R_d = R_start @ pin.exp3(axis_local * theta_t)

        T_d = np.eye(4)
        T_d[:3, :3] = R_d
        T_d[:3, 3] = x_d

        # -----------------------------------
        # Inverzna kinematika:
        # T_d -> q_ref
        # -----------------------------------
        q_ref_prev = q_ref_full.copy()
        dq_ref_prev = dq_ref_full.copy()

        q_ref_full = inverse_kinematics_pose(
            model_pin,
            data_pin,
            q_ref_full,
            ee_frame_id,
            T_d,
            tcp_offset_local=TCP_OFFSET_LOCAL,
            max_iters=80,
            eps=1e-5,
            alpha=0.4,
            damping=1e-3
        )

        q_ref_full[7:] = np.array([0.0001, 0.0001], dtype=float)

        # -----------------------------------
        # Numerički izvodi q_ref:
        # q_ref -> dq_ref -> ddq_ref
        # -----------------------------------
        dq_ref_full_new = np.zeros(9)

        q_diff = pin.difference(
            model_pin,
            q_ref_prev,
            q_ref_full
        )

        dq_ref_full_new[:7] = q_diff[:7] / dt
        dq_ref_full_new[7:] = 0.0

        # opciono ograničenje referentne brzine
        dq_ref_full_new[:7] = np.clip(dq_ref_full_new[:7], -0.8, 0.8)

        ddq_ref_arm = (dq_ref_full_new[:7] - dq_ref_prev[:7]) / dt

        # opciono ograničenje referentnog ubrzanja
        ddq_ref_arm = np.clip(ddq_ref_arm, -8.0, 8.0)

        dq_ref_full = dq_ref_full_new.copy()

        qd_arm = q_ref_full[:7].copy()
        dqd_arm = dq_ref_full[:7].copy()
        ddqd_arm = ddq_ref_arm.copy()

        # -----------------------------------
        # Stvarna TCP poza za debug i crtanje
        # -----------------------------------
        T_act = get_ee_homogeneous_transform(
            model_pin,
            data_pin,
            q,
            ee_frame_id,
            TCP_OFFSET_LOCAL
        )

        x_act = T_act[:3, 3].copy()
        R_act = T_act[:3, :3].copy()

        pos_err = x_d - x_act

        rot_err_local = pin.log3(R_act.T @ R_d)
        rot_err_world = R_act @ rot_err_local

        # -----------------------------------
        # Logovanje za grafike
        # -----------------------------------
        time_log.append(t)

        x_des_log.append(x_d.copy())
        x_act_log.append(x_act.copy())

        # Orijentaciju zapisujemo kao rotacioni vektor
        # u odnosu na globalni/world koordinatni sistem.
        rot_des_log.append(pin.log3(R_d))
        rot_act_log.append(pin.log3(R_act))

        pos_err_norm_log.append(np.linalg.norm(pos_err))
        rot_err_norm_log.append(np.linalg.norm(rot_err_world))

        # -----------------------------------
        # Dinamika iz Pinocchio
        # -----------------------------------
        q_pin = q.copy()
        v_pin = dq.copy()

        M = pin.crba(model_pin, data_pin, q_pin)
        M = 0.5 * (M + M.T)

        h = pin.nonLinearEffects(
            model_pin,
            data_pin,
            q_pin,
            v_pin
        )

        M_arm = M[:7, :7]
        h_arm = h[:7]

        # -----------------------------------
        # IDC zakon — jedini kontroler
        # -----------------------------------
        e_q = qd_arm - q_arm
        e_dq = dqd_arm - dq_arm

        ddq_cmd_joint = ddqd_arm + Kd @ e_dq + Kp @ e_q

        tau = M_arm @ ddq_cmd_joint + h_arm

        tau = np.clip(tau, -80.0, 80.0)

        # -----------------------------------
        # Slanje momenta u MuJoCo
        # -----------------------------------
        data_mj.ctrl[:7] = tau

        if model_mj.nu > 7:
            data_mj.ctrl[7] = 0.0

        # -----------------------------------
        # Pamćenje putanje
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
            print("segment =", segment)
            print("phase =", phase_name)
            print("t_local =", round(t_local, 3))
            print("T_motion =", T_motion)
            print("T_settle =", T_settle)

            print("x_d   =", x_d)
            print("x_act =", x_act)
            print("pos_err =", pos_err)

            print("theta_total =", theta_total)
            print("theta_t =", theta_t)
            print("rot_err_world =", rot_err_world)
            print("||rot_err_world|| =", np.linalg.norm(rot_err_world))

            print("q_arm  =", q_arm)
            print("qd_arm =", qd_arm)
            print("dqd_arm =", dqd_arm)
            print("ddqd_arm =", ddqd_arm)
            print("tau =", tau)

        # -----------------------------------
        # Spoljašnja kratkotrajna sila
        # -----------------------------------
        data_mj.xfrc_applied[:, :] = 0.0

        force_is_active = force_start_time <= t <= force_end_time

        if force_is_active:
            data_mj.xfrc_applied[force_body_id, :3] = external_force_world

        external_force_log.append(np.linalg.norm(external_force_world) if force_is_active else 0.0)

        # -----------------------------------
        # Simulacioni korak
        # -----------------------------------
        mujoco.mj_step(model_mj, data_mj)

        # -----------------------------------
        # Crtanje
        # -----------------------------------
        draw_trajectory_points(v, actual_traj, desired_traj)

        draw_frame_axes_capsules(v, T_A, axis_length=0.08, axis_radius=0.0025)
        draw_frame_axes_capsules(v, T_B, axis_length=0.08, axis_radius=0.0025)
        draw_frame_axes_capsules(v, T_act, axis_length=0.10, axis_radius=0.0020)
        # draw_frame_axes_capsules(v, T_d, axis_length=0.06, axis_radius=0.0015)

        v.sync()
        time.sleep(dt)

        step_count += 1

# =========================
# 13) Grafici posle simulacije
# =========================
import matplotlib.pyplot as plt

time_log = np.array(time_log)

x_des_log = np.array(x_des_log)
x_act_log = np.array(x_act_log)

rot_des_log = np.array(rot_des_log)
rot_act_log = np.array(rot_act_log)

pos_err_norm_log = np.array(pos_err_norm_log)
rot_err_norm_log = np.array(rot_err_norm_log)

# rad -> stepeni
rot_des_deg = np.rad2deg(rot_des_log)
rot_act_deg = np.rad2deg(rot_act_log)
rot_err_norm_deg = np.rad2deg(rot_err_norm_log)

labels = ["x", "y", "z"]
colors = ["r", "g", "b"]


# ============================================
# GRAFIK 1: Pozicija TCP-a
# ============================================
plt.figure(figsize=(11, 6))

for i in range(3):
    plt.plot(
        time_log,
        x_des_log[:, i],
        linestyle="--",
        color=colors[i],
        label=f"{labels[i]} desired"
    )

    plt.plot(
        time_log,
        x_act_log[:, i],
        linestyle="-",
        color=colors[i],
        label=f"{labels[i]} actual"
    )

plt.title("TCP pozicija (desired vs actual)")
plt.xlabel("Vreme [s]")
plt.ylabel("Pozicija [m]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================
# GRAFIK 2: Orijentacija TCP-a (stepeni)
# ============================================
plt.figure(figsize=(11, 6))

for i in range(3):
    plt.plot(
        time_log,
        rot_des_deg[:, i],
        linestyle="--",
        color=colors[i],
        label=f"rot {labels[i]} desired [deg]"
    )

    plt.plot(
        time_log,
        rot_act_deg[:, i],
        linestyle="-",
        color=colors[i],
        label=f"rot {labels[i]} actual [deg]"
    )

plt.title("TCP orijentacija (rotation vector) [deg]")
plt.xlabel("Vreme [s]")
plt.ylabel("Ugao [deg]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================
# GRAFIK 3: Norme grešaka
# ============================================

# Poziciona greška
plt.figure(figsize=(11, 5))
plt.plot(time_log, pos_err_norm_log, label="||x_d - x|| [m]")
plt.title("Poziciona greška")
plt.xlabel("Vreme [s]")
plt.ylabel("Greška [m]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Orijentaciona greška u stepenima
plt.figure(figsize=(11, 5))
plt.plot(time_log, rot_err_norm_deg, label="||rot_err|| [deg]")
plt.title("Orijentaciona greška")
plt.xlabel("Vreme [s]")
plt.ylabel("Greška [deg]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()