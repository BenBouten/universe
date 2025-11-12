"""Camera utilities and projection helpers."""

from __future__ import annotations

import math
from typing import Sequence

import pygame

from core.physics import cross, normalize, vector_length
from game.constants import CAMERA_DISTANCE, SCREEN_HEIGHT, SCREEN_WIDTH


class Camera:
    """Simple orbital camera that can focus on a target point."""

    def __init__(self) -> None:
        self.distance = CAMERA_DISTANCE
        self.yaw = 0.0
        self.pitch = math.radians(12)
        self.focus_mode = True
        self.focus_target = (0.0, 0.0, 0.0)
        self.mouse_sensitivity = 0.005
        self.position = (0.0, 0.0, -self.distance)
        self.forward = (0.0, 0.0, 1.0)
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, 1.0, 0.0)

    def toggle_focus(self) -> None:
        self.focus_mode = not self.focus_mode

    def set_focus_target(self, target: tuple[float, float, float]) -> None:
        self.focus_target = target

    def _calculate_orbit_position(self) -> tuple[float, float, float]:
        cos_pitch = math.cos(self.pitch)
        return (
            self.focus_target[0] + math.sin(self.yaw) * cos_pitch * self.distance,
            self.focus_target[1] + math.sin(self.pitch) * self.distance,
            self.focus_target[2] - math.cos(self.yaw) * cos_pitch * self.distance,
        )

    def _update_basis(self) -> None:
        tx, ty, tz = self.focus_target
        cx, cy, cz = self.position
        forward_vec = (tx - cx, ty - cy, tz - cz)
        forward = normalize(forward_vec)
        up_reference = (0.0, 1.0, 0.0)
        right = cross(forward, up_reference)
        if vector_length(right) < 1e-5:
            up_reference = (0.0, 0.0, 1.0)
            right = cross(forward, up_reference)
        right = normalize(right)
        up = cross(right, forward)
        self.forward = forward
        self.right = right
        self.up = up

    def update(self) -> None:
        if self.focus_mode:
            self.position = self._calculate_orbit_position()
        self._update_basis()

    def handle_mouse_motion(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        pitch_limit = math.radians(85)
        self.pitch = max(-pitch_limit, min(pitch_limit, self.pitch))

    def depth_of(self, point: tuple[float, float, float]) -> float:
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        dz = point[2] - self.position[2]
        return dx * self.forward[0] + dy * self.forward[1] + dz * self.forward[2]


camera = Camera()


def project_point(x: float, y: float, z: float) -> tuple[float, float, float] | None:
    """Project a 3D point into screen space."""
    dx = x - camera.position[0]
    dy = y - camera.position[1]
    dz = z - camera.position[2]
    camera_x = dx * camera.right[0] + dy * camera.right[1] + dz * camera.right[2]
    camera_y = dx * camera.up[0] + dy * camera.up[1] + dz * camera.up[2]
    camera_z = dx * camera.forward[0] + dy * camera.forward[1] + dz * camera.forward[2]
    if camera_z <= 1.0:
        return None
    scale = CAMERA_DISTANCE / camera_z
    screen_x = SCREEN_WIDTH / 2 + camera_x * scale
    screen_y = SCREEN_HEIGHT / 2 - camera_y * scale
    return screen_x, screen_y, scale


def draw_future_path(screen: pygame.Surface, path: Sequence[tuple[float, float, float]]) -> None:
    """Draw a projected polyline representing the predicted path."""
    if not path:
        return
    segments: list[tuple[int, int]] = []
    color = (100, 220, 255)
    for x, y, z in path:
        projected = project_point(x, y, z)
        if projected is None:
            if len(segments) >= 2:
                pygame.draw.lines(screen, color, False, segments, 1)
            segments = []
            continue
        px, py, _ = projected
        segments.append((int(px), int(py)))
        if len(segments) >= 32:
            pygame.draw.lines(screen, color, False, segments, 1)
            segments = [segments[-1]]
    if len(segments) >= 2:
        pygame.draw.lines(screen, color, False, segments, 1)
