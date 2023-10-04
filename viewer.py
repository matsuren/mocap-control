import numpy as np
from scipy.spatial.transform import Rotation as Rot
import glfw
import mujoco_viewer
import mujoco


class MocapObject:
    def __init__(
        self,
        data: mujoco._structs.MjData,
        quat_wxyz=[1, 0, 0, 0],
        translation=[0, 0, 0],
    ):
        # Mujoco data
        self.data = data
        self.mocap_id = 0

        # Initialize pose with given rotation (as quaternion) and translation
        # quaternion in [w, x, y, z] format from mujoco
        w, x, y, z = quat_wxyz
        # quaternion in [x, y, z, w] format to scipy
        self.rotation = Rot.from_quat([x, y, z, w])
        self.translation = np.array(translation)

        # Update based on initial value
        self.data.mocap_pos[self.mocap_id] = self.t
        self.data.mocap_quat[self.mocap_id] = self.quat_wxyz

    @property
    def R(self):
        # Returns the rotation as a 3x3 matrix
        return self.rotation.as_matrix()

    @property
    def quat_wxyz(self):
        # Returns the rotation as a 3x3 matrix
        x, y, z, w = self.rotation.as_quat(canonical=True)
        return w, x, y, z

    @property
    def t(self):
        # Returns the translation as a 3x1 vector
        return self.translation

    def move(self, dx, dy, dz):
        # Update translation
        self.translation += np.array([dx, dy, dz])
        # Update mocap data
        self.data.mocap_pos[self.mocap_id] = self.t

    def _rotate(self, diff_rotation: Rot):
        # Update rotation
        self.rotation = diff_rotation * self.rotation
        # Update mocap data
        self.data.mocap_quat[self.mocap_id] = self.quat_wxyz

    def rotate_x(self, diff_deg):
        # Rotate about x-axis by val degrees with respect to the fixed world coordinates
        diff_rotation = Rot.from_euler("x", diff_deg, degrees=True)
        self._rotate(diff_rotation)

    def rotate_y(self, diff_deg):
        # Rotate about y-axis by val degrees with respect to the fixed world coordinates
        diff_rotation = Rot.from_euler("y", diff_deg, degrees=True)
        self._rotate(diff_rotation)

    def rotate_z(self, diff_deg):
        # Rotate about z-axis by val degrees with respect to the fixed world coordinates
        diff_rotation = Rot.from_euler("z", diff_deg, degrees=True)
        self._rotate(diff_rotation)

    def __str__(self):
        # Returns a string representation of the mocap object's pose
        return f"""
Rotation (Quat): {self.rotation.as_quat(canonical=True)}
Translation: {self.translation}""".strip()

    def __repr__(self):
        return self.__str__()


class InteractiveViewer(mujoco_viewer.MujocoViewer):
    def __init__(
        self,
        model: mujoco._structs.MjModel,
        data: mujoco._structs.MjData,
        mocap: MocapObject,
    ):
        super().__init__(model, data)
        # Mujoco
        self.data = data
        self.mocap = mocap

        # Speed
        self.diff_t = 0.02
        self.diff_rot = 3  # Degree
        self.speedup_ratio = 0.1

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            return

        # Trigger on keyup only:
        if key == glfw.KEY_UP:
            self.mocap.move(0, 0, self.diff_t)
        elif key == glfw.KEY_DOWN:
            self.mocap.move(0, 0, -self.diff_t)
        elif key == glfw.KEY_W:
            self.mocap.move(self.diff_t, 0, 0)
        elif key == glfw.KEY_S:
            self.mocap.move(-self.diff_t, 0, 0)
        elif key == glfw.KEY_A:
            self.mocap.move(0, self.diff_t, 0)
        elif key == glfw.KEY_D:
            self.mocap.move(0, -self.diff_t, 0)
        elif key == glfw.KEY_RIGHT:
            self.mocap.rotate_x(self.diff_rot)
        elif key == glfw.KEY_LEFT:
            self.mocap.rotate_x(-self.diff_rot)
        elif key == glfw.KEY_R:
            self.mocap.rotate_y(self.diff_rot)
        elif key == glfw.KEY_F:
            self.mocap.rotate_y(-self.diff_rot)
        elif key == glfw.KEY_Q:
            self.mocap.rotate_z(self.diff_rot)
        elif key == glfw.KEY_E:
            self.mocap.rotate_z(-self.diff_rot)
        elif key == glfw.KEY_EQUAL:
            ratio = 1 + self.speedup_ratio
            self.diff_t *= ratio
            self.diff_rot *= ratio
        elif key == glfw.KEY_MINUS:
            ratio = 1 - self.speedup_ratio
            self.diff_t *= ratio
            self.diff_rot *= ratio
        elif key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.set_window_should_close(self.window, True)
        else:
            pass
            # # If key is not overwrittten
            # super()._key_callback(window, key, scancode, action, mods)

    def _create_overlay(self):
        super()._create_overlay()
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        # Remove topleft overlay because it's overwritten
        self._overlay[topleft] = ["", ""]

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        add_overlay(
            topright,
            "Go up/down",
            "[up]/[down] arrow",
        )
        add_overlay(topright, "Go forwarf/backward", "[W]/[S]")
        add_overlay(topright, "Go left/right", "[A]/[D]")
        add_overlay(topright, "ROT_X", "[left]/[right] arrow")
        add_overlay(topright, "ROT_Y", "[R]/[F]")
        add_overlay(topright, "ROT_Z", "[Q]/[E]")
        add_overlay(topright, "Slow down/Speed up", "[-]/[=]")
        add_overlay(
            topright,
            "Speed translate/rotate",
            f"{self.diff_t:.3f}, {self.diff_rot:.1f}[deg]",
        )


if __name__ == "__main__":
    print("InteractiveViewer")
    xml_string = """
<?xml version="1.0" ?>
<mujoco>
	<asset>
		<texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
		<material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
	</asset>
	<worldbody>
		<light name="light0" mode="targetbody" target="box" directional="true" pos="1 -1 3" />
		<geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>

		<body mocap="true" name="mocap" pos="0 0 0.5">
			<!-- <geom size="0.05 0.05 0.05" type="box" rgba="0 1 0 1" /> -->
		</body>		
		<body name="box" pos="0 0 0.5">
			<freejoint/>
			<geom size="0.15 0.15 0.15" type="box" rgba="0 0 1 1" />
		</body>
	</worldbody>
	<equality>
		<weld body1="mocap" body2="box"/>
	</equality>
</mujoco>
"""

    # Load model
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Setup mocap object
    quat = data.mocap_quat[0]
    t = data.mocap_pos[0]
    mocap = MocapObject(data=data, quat_wxyz=quat, translation=t)

    # Create interactive viewer
    viewer = InteractiveViewer(model, data, mocap)
    # simulate and render
    while viewer.is_alive:
        mujoco.mj_step(model, data, nstep=10)
        viewer.render()
    # close
    viewer.close()
