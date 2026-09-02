"""Robotiq boundary with no assumed protocol or command schema."""


class GripperInterface:
    def __init__(self, motion_enabled: bool = False) -> None:
        self.motion_enabled = bool(motion_enabled)

    def command(self, arm: str, request: object) -> None:
        del arm, request
        if not self.motion_enabled:
            raise PermissionError("motion enable guard rejected gripper command")
        raise NotImplementedError("Robotiq command interface REQUIRES_ONSITE_VALIDATION")
