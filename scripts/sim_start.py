"""Start Intel's MuJoCo plugin with a longer robot-connect budget.

The runtime gives the robot owner 10 s to come up, which a cold start with software (OSMesa) camera
rendering can overrun. Raise that default in this process, then hand over to the plugin's own CLI.

    python scripts/sim_start.py start --profile franka_panda --model scenes/can_flip/scene.xml
"""

import os
import sys

import functools

from physicalai.robot.transport import _shared_robot

_init = _shared_robot.SharedRobot.__init__
_budget = float(os.environ.get("CELL_CONNECT_TIMEOUT", "60"))


@functools.wraps(_init)
def _init_with_budget(self, *args, **kwargs):
    kwargs["connect_timeout"] = max(float(kwargs.get("connect_timeout", 0.0) or 0.0), _budget)
    _init(self, *args, **kwargs)


_shared_robot.SharedRobot.__init__ = _init_with_budget

from physicalai_mujoco_plugin.__main__ import main  # noqa: E402

sys.argv = ["physicalai-mujoco", *sys.argv[1:]]
sys.exit(main())
