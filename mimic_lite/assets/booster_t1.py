from dataclasses import replace
from pathlib import Path

import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
    MjlabCollisionCfg,
)
from active_adaptation.registry import Registry

registry = Registry.instance()

BOOSTER_T1_DATA_DIR = Path(__file__).resolve().parent / "data" / "booster_t1"
BOOSTER_T1_MJCF_PATH = BOOSTER_T1_DATA_DIR / "booster_t1.xml"
# GMR ships matching URDFs; only used by the isaaclab backend.
BOOSTER_T1_URDF_PATH = Path("/scratch/cluster/zifan/GMR/assets/booster_t1/T1_serial.urdf")

BOOSTER_T1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

BOOSTER_T1_BODY_NAMES = [
    "Trunk",
    "H1",
    "H2",
    "AL1",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR1",
    "AR2",
    "AR3",
    "right_hand_link",
    "Waist",
    "Hip_Pitch_Left",
    "Hip_Roll_Left",
    "Hip_Yaw_Left",
    "Shank_Left",
    "Ankle_Cross_Left",
    "left_foot_link",
    "Hip_Pitch_Right",
    "Hip_Roll_Right",
    "Hip_Yaw_Right",
    "Shank_Right",
    "Ankle_Cross_Right",
    "right_foot_link",
]

# Actuator parameters from holosoma t1_23dof (verified on the physical robot):
# effort/velocity limits, PD gains, joint friction, and reflected armatures.
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_ANKLE_PITCH = 0.0407621
ARMATURE_ANKLE_ROLL = 0.0111713


def _motor_actuator(
    joint_names_expr: str,
    *,
    effort: float,
    velocity: float,
    stiffness: float,
    damping: float,
    friction: float,
    armature: float,
) -> ActuatorCfg:
    return ActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=effort,
        velocity_limit=velocity,
        stiffness=stiffness,
        damping=damping,
        friction=friction,
        armature=armature,
    )


BOOSTER_T1_INIT_STATE = InitialStateCfg(
    pos=(0.0, 0.0, 0.68),
    joint_pos={
        "Left_Shoulder_Pitch": 0.2,
        "Left_Shoulder_Roll": -1.35,
        "Left_Elbow_Yaw": -0.5,
        "Right_Shoulder_Pitch": 0.2,
        "Right_Shoulder_Roll": 1.35,
        "Right_Elbow_Yaw": 0.5,
        ".*_Hip_Pitch": -0.2,
        ".*_Knee_Pitch": 0.4,
        ".*_Ankle_Pitch": -0.25,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)


BOOSTER_T1_CFG = AssetCfg(
    mjcf_path=BOOSTER_T1_MJCF_PATH,
    usd_path=BOOSTER_T1_URDF_PATH,
    init_state=BOOSTER_T1_INIT_STATE,
    self_collisions=True,
    actuators={
        "head": _motor_actuator(
            "AAHead_yaw|Head_pitch",
            effort=7.0,
            velocity=12.56,
            stiffness=20.0,
            damping=1.0,
            friction=0.5,
            armature=0.01,
        ),
        "arms": _motor_actuator(
            ".*_Shoulder_Pitch|.*_Shoulder_Roll|.*_Elbow_Pitch|.*_Elbow_Yaw",
            effort=18.0,
            velocity=18.84,
            stiffness=20.0,
            damping=0.5,
            friction=0.5,
            armature=ARMATURE_4310,
        ),
        "waist": _motor_actuator(
            "Waist",
            effort=30.0,
            velocity=10.88,
            stiffness=200.0,
            damping=5.0,
            friction=0.459068,
            armature=ARMATURE_6408,
        ),
        "hip_pitch": _motor_actuator(
            ".*_Hip_Pitch",
            effort=45.0,
            velocity=12.5,
            stiffness=200.0,
            damping=5.0,
            friction=0.486176,
            armature=ARMATURE_8112,
        ),
        "hip_roll": _motor_actuator(
            ".*_Hip_Roll",
            effort=30.0,
            velocity=10.9,
            stiffness=200.0,
            damping=5.0,
            friction=0.880781,
            armature=ARMATURE_6408,
        ),
        "hip_yaw": _motor_actuator(
            ".*_Hip_Yaw",
            effort=30.0,
            velocity=10.9,
            stiffness=200.0,
            damping=5.0,
            friction=0.238063,
            armature=ARMATURE_6408,
        ),
        "knee": _motor_actuator(
            ".*_Knee_Pitch",
            effort=60.0,
            velocity=11.7,
            stiffness=200.0,
            damping=5.0,
            friction=0.998922,
            armature=ARMATURE_8116,
        ),
        "ankle_pitch": _motor_actuator(
            ".*_Ankle_Pitch",
            effort=24.0,
            velocity=18.8,
            stiffness=50.0,
            damping=2.0,
            friction=0.71953,
            armature=ARMATURE_ANKLE_PITCH,
        ),
        "ankle_roll": _motor_actuator(
            ".*_Ankle_Roll",
            effort=12.0,
            velocity=12.4,
            stiffness=50.0,
            damping=2.0,
            friction=0.209926,
            armature=ARMATURE_ANKLE_ROLL,
        ),
    },
    sensors_mjlab=[
        ContactSensorCfg(
            name="contact_forces",
            primary_contact_match_mode="subtree",
            primary_contact_match_pattern=r"^(left_foot_link|right_foot_link)$",
            primary_contact_match_entity="robot",
            secondary_contact_match_mode="body",
            secondary_contact_match_pattern="terrain",
            track_air_time=True,
            history_length=4,
            reduce="netforce",
        ),
        ContactSensorCfg(
            name="self_collision",
            primary_contact_match_mode="subtree",
            primary_contact_match_pattern="Trunk",
            primary_contact_match_entity="robot",
            secondary_contact_match_mode="subtree",
            secondary_contact_match_pattern="Trunk",
            secondary_contact_match_entity="robot",
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    ],
    mjlab_collisions=[
        MjlabCollisionCfg(
            geom_names_expr=(".*_collision",),
            disable_other_geoms=False,
        ),
    ],
    joint_names_simulation=BOOSTER_T1_JOINT_NAMES,
    body_names_simulation=BOOSTER_T1_BODY_NAMES,
    joint_symmetry_mapping=symmetry_utils.mirrored(
        {
            "AAHead_yaw": (-1, "AAHead_yaw"),
            "Head_pitch": (1, "Head_pitch"),
            "Left_Shoulder_Pitch": (1, "Right_Shoulder_Pitch"),
            "Left_Shoulder_Roll": (-1, "Right_Shoulder_Roll"),
            "Left_Elbow_Pitch": (1, "Right_Elbow_Pitch"),
            "Left_Elbow_Yaw": (-1, "Right_Elbow_Yaw"),
            "Waist": (-1, "Waist"),
            "Left_Hip_Pitch": (1, "Right_Hip_Pitch"),
            "Left_Hip_Roll": (-1, "Right_Hip_Roll"),
            "Left_Hip_Yaw": (-1, "Right_Hip_Yaw"),
            "Left_Knee_Pitch": (1, "Right_Knee_Pitch"),
            "Left_Ankle_Pitch": (1, "Right_Ankle_Pitch"),
            "Left_Ankle_Roll": (-1, "Right_Ankle_Roll"),
        }
    ),
    spatial_symmetry_mapping=symmetry_utils.mirrored(
        {
            "Trunk": "Trunk",
            "H1": "H1",
            "H2": "H2",
            "AL1": "AR1",
            "AL2": "AR2",
            "AL3": "AR3",
            "left_hand_link": "right_hand_link",
            "Waist": "Waist",
            "Hip_Pitch_Left": "Hip_Pitch_Right",
            "Hip_Roll_Left": "Hip_Roll_Right",
            "Hip_Yaw_Left": "Hip_Yaw_Right",
            "Shank_Left": "Shank_Right",
            "Ankle_Cross_Left": "Ankle_Cross_Right",
            "left_foot_link": "right_foot_link",
        }
    ),
)

registry.register("asset", "booster_t1", BOOSTER_T1_CFG)


# ---------------------------------------------------------------------------
# Impedance-control variant.
#
# The default `booster_t1` uses stiff position-control PD gains (legs Kp=200).
# This variant swaps in soft, compliant gains (G1-style, derived from a 10Hz
# natural frequency / damping-ratio-2 tuning) so the policy commands smaller
# residual position targets against a more back-drivable plant. Only the PD
# gains change; effort/velocity limits, joint friction and armatures are kept
# identical to the physical-robot values. Pair with a small action_scale
# (0.25) in the task config.
# ---------------------------------------------------------------------------
BOOSTER_T1_IMPEDANCE_ACTUATORS = {
    "head": _motor_actuator(
        "AAHead_yaw|Head_pitch",
        effort=7.0,
        velocity=12.56,
        stiffness=5.0,
        damping=0.2,
        friction=0.5,
        armature=0.01,
    ),
    "arms": _motor_actuator(
        ".*_Shoulder_Pitch|.*_Shoulder_Roll|.*_Elbow_Pitch|.*_Elbow_Yaw",
        effort=18.0,
        velocity=18.84,
        stiffness=14.25062309787429,
        damping=0.907222843292423,
        friction=0.5,
        armature=ARMATURE_4310,
    ),
    "waist": _motor_actuator(
        "Waist",
        effort=30.0,
        velocity=10.88,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        friction=0.459068,
        armature=ARMATURE_6408,
    ),
    "hip_pitch": _motor_actuator(
        ".*_Hip_Pitch",
        effort=45.0,
        velocity=12.5,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        friction=0.486176,
        armature=ARMATURE_8112,
    ),
    "hip_roll": _motor_actuator(
        ".*_Hip_Roll",
        effort=30.0,
        velocity=10.9,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        friction=0.880781,
        armature=ARMATURE_6408,
    ),
    "hip_yaw": _motor_actuator(
        ".*_Hip_Yaw",
        effort=30.0,
        velocity=10.9,
        stiffness=40.17923847137318,
        damping=2.5578897650279457,
        friction=0.238063,
        armature=ARMATURE_6408,
    ),
    "knee": _motor_actuator(
        ".*_Knee_Pitch",
        effort=60.0,
        velocity=11.7,
        stiffness=99.09842777666113,
        damping=6.3088018534966395,
        friction=0.998922,
        armature=ARMATURE_8116,
    ),
    "ankle_pitch": _motor_actuator(
        ".*_Ankle_Pitch",
        effort=24.0,
        velocity=18.8,
        stiffness=14.25062309787429,
        damping=1.0,
        friction=0.71953,
        armature=ARMATURE_ANKLE_PITCH,
    ),
    "ankle_roll": _motor_actuator(
        ".*_Ankle_Roll",
        effort=12.0,
        velocity=12.4,
        stiffness=14.25062309787429,
        damping=1.0,
        friction=0.209926,
        armature=ARMATURE_ANKLE_ROLL,
    ),
}

BOOSTER_T1_IMPEDANCE_CFG = replace(
    BOOSTER_T1_CFG, actuators=BOOSTER_T1_IMPEDANCE_ACTUATORS
)
registry.register("asset", "booster_t1_impedance", BOOSTER_T1_IMPEDANCE_CFG)
