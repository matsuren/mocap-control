import numpy as np
from scipy.spatial.transform import Rotation as R
import glfw
from enum import Enum
import mujoco
import mujoco_viewer


def rotation(theta_x=0, theta_y=0, theta_z=0):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )
    R = rot_x.dot(rot_y).dot(rot_z)

    return R


def quat2euler(quat):
    # transfer quat to euler
    r = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))
    return r.as_euler("XYZ")


class Direction(Enum):
    POS: int = 1
    NEG: int = -1


class Controller:
    # The max speed.
    MAX_SPEED = 1.0

    # The minimum speed.
    MIN_SPEED = 0.0
    SPEED_CHANGE_PERCENT = 0.2

    def __init__(self, data) -> None:
        super().__init__()
        self._speeds = np.array([0.01, 0.1])
        self.data = data

    @property
    def pos_speed(self):
        """
        The speed that arm moves.
        """
        return self._speeds[0]

    @property
    def rot_speed(self):
        """
        The speed that wrist rotates.
        """
        return self._speeds[1]

    def speed_up(self):
        """
        Increase gripper moving speed.
        """
        self._speeds = np.minimum(
            self._speeds * (1 + self.SPEED_CHANGE_PERCENT), self.MAX_SPEED
        )

    def speed_down(self):
        """
        Decrease gripper moving speed.
        """
        self._speeds = np.maximum(
            self._speeds * (1 - self.SPEED_CHANGE_PERCENT), self.MIN_SPEED
        )

    def move_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._move(0, direction)

    def move_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._move(1, direction)

    def move_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._move(2, direction)

    def rot_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._rot(0, direction)

    def rot_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._rot(1, direction)

    def rot_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._rot(2, direction)

    def _rot(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        e = quat2euler(self.data.mocap_quat[0])
        if axis == 2:
            r = R.from_matrix(
                rotation(e[0], e[1], e[2] + self.rot_speed * direction.value)
            )
            self.data.mocap_quat[0] = np.array(
                [r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]]
            )
        elif axis == 1:
            r = R.from_matrix(
                rotation(e[0], e[1] + self.rot_speed * direction.value, e[2])
            )
            self.data.mocap_quat[0] = np.array(
                [r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]]
            )
        elif axis == 0:
            r = R.from_matrix(
                rotation(e[0] + self.rot_speed * direction.value, e[1], e[2])
            )
            self.data.mocap_quat[0] = np.array(
                [r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]]
            )
        else:
            pass

    def _move(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        if axis == 2:
            self.data.mocap_pos[0] = self.data.mocap_pos[0] + np.array(
                [0, 0, self.pos_speed * direction.value]
            )
        elif axis == 1:
            self.data.mocap_pos[0] = self.data.mocap_pos[0] + np.array(
                [0, self.pos_speed * direction.value, 0]
            )
        elif axis == 0:
            self.data.mocap_pos[0] = self.data.mocap_pos[0] + np.array(
                [self.pos_speed * direction.value, 0, 0]
            )
        else:
            pass


class Viewer(mujoco_viewer.MujocoViewer):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.controller = Controller(data)

    def _key_callback(self, window, key, scancode, action, mods):
        # Trigger on keyup only:
        if key == glfw.KEY_UP:
            self.controller.move_z(Direction.POS)
        elif key == glfw.KEY_DOWN:
            self.controller.move_z(Direction.NEG)
        elif key == glfw.KEY_RIGHT:
            self.controller.move_y(Direction.POS)
        elif key == glfw.KEY_LEFT:
            self.controller.move_y(Direction.NEG)
        elif key == glfw.KEY_B:
            self.controller.move_x(Direction.NEG)
        elif key == glfw.KEY_F:
            self.controller.move_x(Direction.POS)
        elif key == glfw.KEY_A:
            self.controller.rot_y(Direction.POS)
        elif key == glfw.KEY_S:
            self.controller.rot_y(Direction.NEG)
        elif key == glfw.KEY_Q:
            self.controller.rot_x(Direction.POS)
        elif key == glfw.KEY_W:
            self.controller.rot_x(Direction.NEG)
        elif key == glfw.KEY_Z:
            self.controller.rot_z(Direction.POS)
        elif key == glfw.KEY_X:
            self.controller.rot_z(Direction.NEG)
        elif key == glfw.KEY_MINUS:
            self.controller.speed_down()
        elif key == glfw.KEY_EQUAL:
            self.controller.speed_up()
        else:
            # If key is not overwrittten
            super()._key_callback(window, key, scancode, action, mods)

    def _create_overlay(self):
        super()._create_overlay()
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        add_overlay(
            topright,
            "Go up/down/left/right",
            "[up]/[down]/[left]/[right] arrow",
        )
        add_overlay(topright, "Go forwarf/backward", "[F]/[B]")
        add_overlay(topright, "ROT_X", "[Q]/[W]")
        add_overlay(topright, "ROT_Y", "[A]/[S]")
        add_overlay(topright, "ROT_Z", "[Z]/[X]")
        add_overlay(topright, "Slow down/Speed up", "[-]/[=]")


def main():
    # load model
    model = mujoco.MjModel.from_xml_path("./UR5_xhand.xml")
    data = mujoco.MjData(model)
    # viewer set up
    viewer = Viewer(model, data)
    # postion offset
    data.mocap_pos[0] = np.array([0.08229997, 0.10921554, 1.871059]) + np.array(
        [0.3, 0, -0.4]
    )

    # simulate and render
    while viewer.is_alive:
        mujoco.mj_step(model, data, nstep=10)
        viewer.render()
    # close
    viewer.close()


if __name__ == "__main__":
    main()
