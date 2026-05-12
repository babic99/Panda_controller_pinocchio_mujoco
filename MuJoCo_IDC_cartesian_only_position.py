import os
import time
import numpy as np
import mujoco
from mujoco import viewer
from mujoco import mjtGeom
import pinocchio as pin
import example_robot_data

# kako ce se brojevi ispisivati na konzoli, sa 5 deicimala i bez naucnog zapisa 
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

EE_FRAME_NAME = "panda_hand"
ee_frame_id = model_pin.getFrameId(EE_FRAME_NAME)


# =========================
# 3) Pomoćne funkcije
# =========================
def update_pin(model, data, q_full, dq_full=None):
    """
    Ažurira Forward kinematics i frame placements.
    Ako je dq_full prosleđen, računa FK i sa brzinama.
    """
    if dq_full is None:
        pin.forwardKinematics(model, data, q_full)
    else:
        pin.forwardKinematics(model, data, q_full, dq_full)
    pin.updateFramePlacements(model, data)


def get_ee_homogeneous_transform(model, data, q_full, frame_id):
    update_pin(model, data, q_full)
    return data.oMf[frame_id].homogeneous.copy()


def get_ee_position(model, data, q_full, frame_id):
    update_pin(model, data, q_full)
    return data.oMf[frame_id].translation.copy()


def get_linear_jacobian(model, data, q_full, frame_id):
    """
    Linearni Jakobijan 3x7 u LOCAL_WORLD_ALIGNED smislu.
    Po tvom testu za ovu postavku linearni deo uzimamo kao J6[:3, :7].
    """
    update_pin(model, data, q_full)
    J6 = pin.computeFrameJacobian(
        model,
        data,
        q_full,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return J6[:3, :7].copy()


def numerical_jdot_times_v(model, data, q_full, v_full, frame_id, eps=1e-6):
    """
    Numerička aproksimacija:
        Jdot(q,v) * v ≈ (J(q + eps*v) - J(q)) / eps * v
    """
    J_now = get_linear_jacobian(model, data, q_full, frame_id)
    q_next = pin.integrate(model, q_full, v_full * eps)
    J_next = get_linear_jacobian(model, data, q_next, frame_id)
    Jdot = (J_next - J_now) / eps
    return Jdot @ v_full[:7]


def damped_pseudoinverse(J, damping=1e-2):
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))


def quintic_scalar_trajectory(L_total, T, t):
    """
    5th order scalar trajectory:
        L(0)=0, L(T)=L_total
        dL(0)=dL(T)=0
        ddL(0)=ddL(T)=0
    """
    t = np.clip(t, 0.0, T)
    s = t / T

    alpha = 10 * s**3 - 15 * s**4 + 6 * s**5
    dalpha = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    ddalpha = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)

    L = L_total * alpha
    dL = L_total * dalpha
    ddL = L_total * ddalpha
    return L, dL, ddL


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


# =========================
# 4) Početna konfiguracija
# =========================
q_start_arm = np.array([-0.5, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0], dtype=float)

q_init_full = np.zeros(9)
q_init_full[:7] = q_start_arm
q_init_full[7:] = np.array([0.04, 0.04], dtype=float)

data_mj.qpos[:] = q_init_full
data_mj.qvel[:] = 0.0
mujoco.mj_forward(model_mj, data_mj)


# =========================
# 5) Target poze
# =========================
T_A = get_ee_homogeneous_transform(model_pin, data_pin, q_init_full, ee_frame_id)

T_B = T_A.copy()
T_B[:3, 3] += np.array([0.00, 0.50, 0.00], dtype=float)

print("T_A =\n", T_A)
print("T_B =\n", T_B)
print("p_A =", T_A[:3, 3])
print("p_B =", T_B[:3, 3])


# =========================
# 6) IDC gainovi (joint space)
# =========================
Kp = np.diag([100, 100, 80, 60, 40, 30, 20]).astype(float)
Kd = np.diag([20, 20, 16, 12, 8, 6, 4]).astype(float)
#Kp = np.diag([120, 120, 100, 80, 60, 40, 30]).astype(float)
#Kd = np.diag([25, 25, 20, 16, 10, 8, 6]).astype(float)
#Kp = np.zeros(7)
#Kd = np.zeros(7)

# =========================
# 7) Task-space gainovi
# =========================
Kx_p = np.diag([8.0, 8.0, 8.0])
Kx_d = np.diag([4.0, 4.0, 4.0])

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

draw_every_n_steps = 10      # pamti svaku 10-tu tačku
max_traj_points = 300        # po listi


# =========================
# 11) Simulacija
# =========================
dt = model_mj.opt.timestep
step_count = 0

with viewer.launch_passive(model_mj, data_mj) as v:
    while v.is_running():

        # koristi simulaciono vreme, ne wall-clock
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

        # -----------------------------------
        # (C) Kartezijanska trajektorija po pravoj
        # -----------------------------------
        direction_vec = x_goal - x_start
        path_length = np.linalg.norm(direction_vec)

        if path_length < 1e-10:
            raise ValueError("Dužina putanje je praktično nula.")

        e_hat = direction_vec / path_length

        L, dL, ddL = quintic_scalar_trajectory(path_length, T_segment, t_local)

        x_d = x_start + e_hat * L
        dx_d = e_hat * dL
        ddx_d = e_hat * ddL

        # -----------------------------------
        # (D) Stvarno EE stanje + Jakobijan
        # -----------------------------------
        x_act = get_ee_position(model_pin, data_pin, q, ee_frame_id)
        J = get_linear_jacobian(model_pin, data_pin, q, ee_frame_id)
        J_pinv = damped_pseudoinverse(J, damping=jac_damping)

        dx_act = J @ dq_arm

        # Jdot(q,dq) * dq  -- koristi stvarno stanje
        jdot_qdot = numerical_jdot_times_v(
            model_pin, data_pin, q, dq, ee_frame_id
        )

        # -----------------------------------
        # (E) Task-space feedback
        # -----------------------------------
        pos_err = x_d - x_act
        vel_err = dx_d - dx_act

        # željena task akceleracija sa PD korekcijom
        ddx_cmd = ddx_d + Kx_d @ vel_err + Kx_p @ pos_err

        # -----------------------------------
        # (F) Željeno joint ubrzanje
        # -----------------------------------
        ddqd_cmd = J_pinv @ (ddx_cmd - jdot_qdot)

        # Integracija željene joint reference
        dq_ref_full[:7] += ddqd_cmd * dt
        dq_ref_full[7:] = 0.0

        q_ref_full = pin.integrate(model_pin, q_ref_full, dq_ref_full * dt)

        # prsti ostaju otvoreni
        q_ref_full[7:] = np.array([0.04, 0.04], dtype=float)
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
            data_mj.ctrl[7] = 255.0

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
            print("dx_d   =", dx_d)
            print("dx_act =", dx_act)
            print("ddx_cmd =", ddx_cmd)
            print("q_arm =", q_arm)
            print("qd_arm =", qd_arm)
            print("tau =", tau)

        # -----------------------------------
        # Korak simulacije
        # -----------------------------------
        mujoco.mj_step(model_mj, data_mj)

        # -----------------------------------
        # Crtanje putanje u viewer-u
        # -----------------------------------
        draw_trajectory_points(v, actual_traj, desired_traj)

        v.sync()
        time.sleep(dt)
        step_count += 1