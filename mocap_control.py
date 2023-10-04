import numpy as np
import mujoco
from viewer import MocapObject, InteractiveViewer


def main():
    # load model
    model = mujoco.MjModel.from_xml_path("./UR5_xhand.xml")
    data = mujoco.MjData(model)

    # Setup mocap object
    t = np.array([0.08229997, 0.10921554, 1.871059]) + np.array([0.3, 0, -0.4])

    mocap = MocapObject(data=data, translation=t)

    # Create interactive viewer
    viewer = InteractiveViewer(model, data, mocap)

    # simulate and render
    while viewer.is_alive:
        mujoco.mj_step(model, data, nstep=10)
        viewer.render()
    # close
    viewer.close()


if __name__ == "__main__":
    main()
