"""Player-controlled ship logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from core.physics import cross, normalize, vector_length, wrap_coordinate, wrapped_delta
from game.camera import project_point
from game.constants import WHITE, WORLD_DEPTH, WORLD_HEIGHT, WORLD_WIDTH, G

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from game.scene import Universe
    from world.bodies import Star


class PlayerShip:
    """A controllable ship that gathers energy from planets."""

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = math.pi / 2
        self.pitch = 0.0
        self.mass = 120.0
        self.turn_speed = math.radians(140)
        self.pitch_speed = math.radians(100)
        self.thrust_force = 22000.0
        self.ship_radius = 10
        self.collection_radius = 26
        self.sector_progress = 0
        self.total_energy = 0
        self.control_mode = "rotation"
        self.main_throttle = 0.0
        self.translation_thrust = self.thrust_force * 0.32

    def handle_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        if self.control_mode == "rotation":
            yaw_input = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                yaw_input -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                yaw_input += 1.0
            self.yaw += yaw_input * self.turn_speed * dt

            pitch_input = 0.0
            if keys[pygame.K_DOWN]:
                pitch_input -= 1.0
            if keys[pygame.K_UP]:
                pitch_input += 1.0
            self.pitch += pitch_input * self.pitch_speed * dt
            self.pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.pitch))

            throttle_input = 0.0
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                throttle_input += 1.0
            if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                throttle_input -= 1.0
            self.main_throttle += throttle_input * dt
            self.main_throttle = max(0.0, min(1.0, self.main_throttle))

            forward = (
                math.cos(self.yaw) * math.cos(self.pitch),
                math.sin(self.pitch),
                math.sin(self.yaw) * math.cos(self.pitch),
            )
            acceleration = self.thrust_force * self.main_throttle / self.mass
            self.vx += forward[0] * acceleration * dt
            self.vy += forward[1] * acceleration * dt
            self.vz += forward[2] * acceleration * dt
        else:
            thrust_x = thrust_y = thrust_z = 0.0
            if keys[pygame.K_w]:
                thrust_x += 1.0
            if keys[pygame.K_s]:
                thrust_x -= 1.0
            if keys[pygame.K_d]:
                thrust_z += 1.0
            if keys[pygame.K_a]:
                thrust_z -= 1.0
            if keys[pygame.K_UP]:
                thrust_y += 1.0
            if keys[pygame.K_DOWN]:
                thrust_y -= 1.0
            if keys[pygame.K_RIGHT]:
                thrust_z += 0.6
            if keys[pygame.K_LEFT]:
                thrust_z -= 0.6

            length = math.sqrt(thrust_x * thrust_x + thrust_y * thrust_y + thrust_z * thrust_z)
            if length > 0.0:
                thrust_x /= length
                thrust_y /= length
                thrust_z /= length
            acceleration = self.translation_thrust / self.mass
            self.vx += thrust_x * acceleration * dt
            self.vy += thrust_y * acceleration * dt
            self.vz += thrust_z * acceleration * dt

    def toggle_control_mode(self) -> None:
        self.control_mode = "translation" if self.control_mode == "rotation" else "rotation"

    def cut_throttle(self) -> None:
        self.main_throttle = 0.0

    def get_position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def gravity_acceleration(
        self, universe: "Universe", x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        ax = ay = az = 0.0
        for solar_system in universe.solar_systems:
            bodies = [solar_system.star, *solar_system.planets]
            for body in bodies:
                dx = wrapped_delta(x, body.x, WORLD_WIDTH)
                dy = wrapped_delta(y, body.y, WORLD_HEIGHT)
                dz = wrapped_delta(z, body.z, WORLD_DEPTH)
                r_sq = dx * dx + dy * dy + dz * dz
                if r_sq < 1.0:
                    continue
                r = math.sqrt(r_sq)
                acceleration = G * body.mass / r_sq
                ax += acceleration * dx / r
                ay += acceleration * dy / r
                az += acceleration * dz / r
        return ax, ay, az

    def predict_future_path(
        self, universe: "Universe", duration: float = 120.0, step: float = 1.0
    ) -> list[tuple[float, float, float]]:
        positions: list[tuple[float, float, float]] = []
        px, py, pz = self.x, self.y, self.z
        vx, vy, vz = self.vx, self.vy, self.vz
        steps = max(1, int(duration / step))
        for _ in range(steps):
            ax, ay, az = self.gravity_acceleration(universe, px, py, pz)
            vx += ax * step
            vy += ay * step
            vz += az * step
            px = wrap_coordinate(px + vx * step, WORLD_WIDTH)
            py = wrap_coordinate(py + vy * step, WORLD_HEIGHT)
            pz = wrap_coordinate(pz + vz * step, WORLD_DEPTH)
            positions.append((px, py, pz))
        return positions

    def apply_gravity(self, universe: "Universe", dt: float) -> None:
        ax, ay, az = self.gravity_acceleration(universe, self.x, self.y, self.z)
        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt

    def update(self, dt: float) -> None:
        self.x = wrap_coordinate(self.x + self.vx * dt, WORLD_WIDTH)
        self.y = wrap_coordinate(self.y + self.vy * dt, WORLD_HEIGHT)
        self.z = wrap_coordinate(self.z + self.vz * dt, WORLD_DEPTH)
        self.vx *= 0.998
        self.vy *= 0.998
        self.vz *= 0.998

    def collect_energy(self, universe: "Universe") -> int:
        collected = 0
        for solar_system in universe.solar_systems:
            remaining_planets = []
            for planet in solar_system.planets:
                dx = wrapped_delta(self.x, planet.x, WORLD_WIDTH)
                dy = wrapped_delta(self.y, planet.y, WORLD_HEIGHT)
                dz = wrapped_delta(self.z, planet.z, WORLD_DEPTH)
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                if distance <= planet.radius + self.collection_radius:
                    energy_value = max(1, int(planet.mass))
                    collected += energy_value
                else:
                    remaining_planets.append(planet)
            solar_system.planets = remaining_planets
        if collected:
            self.sector_progress += collected
            self.total_energy += collected
        return collected

    def check_star_collision(self, stars: list["Star"]) -> bool:
        for star in stars:
            dx = wrapped_delta(self.x, star.x, WORLD_WIDTH)
            dy = wrapped_delta(self.y, star.y, WORLD_HEIGHT)
            dz = wrapped_delta(self.z, star.z, WORLD_DEPTH)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= star.radius + self.ship_radius - 2:
                return True
        return False

    def respawn(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = WORLD_DEPTH * 0.25
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = math.pi / 2
        self.pitch = 0.0
        self.main_throttle = 0.0
        self.sector_progress = 0

    def draw(self, screen: pygame.Surface) -> None:
        cos_pitch = math.cos(self.pitch)
        forward = (
            math.cos(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            math.sin(self.yaw) * cos_pitch,
        )
        up_reference = (0.0, 1.0, 0.0)
        right = cross(forward, up_reference)
        if vector_length(right) < 1e-5:
            right = (1.0, 0.0, 0.0)
        right = normalize(right)
        up_local = normalize(cross(right, forward))

        nose = (
            self.x + forward[0] * self.ship_radius * 3.0,
            self.y + forward[1] * self.ship_radius * 3.0,
            self.z + forward[2] * self.ship_radius * 3.0,
        )
        tail = (
            self.x - forward[0] * self.ship_radius * 1.8,
            self.y - forward[1] * self.ship_radius * 1.8,
            self.z - forward[2] * self.ship_radius * 1.8,
        )
        left = (
            self.x - right[0] * self.ship_radius + up_local[0] * self.ship_radius * 0.6,
            self.y - right[1] * self.ship_radius + up_local[1] * self.ship_radius * 0.6,
            self.z - right[2] * self.ship_radius + up_local[2] * self.ship_radius * 0.6,
        )
        right_pt = (
            self.x + right[0] * self.ship_radius + up_local[0] * self.ship_radius * 0.6,
            self.y + right[1] * self.ship_radius + up_local[1] * self.ship_radius * 0.6,
            self.z + right[2] * self.ship_radius + up_local[2] * self.ship_radius * 0.6,
        )

        points_3d = [nose, left, tail, right_pt]
        projected_points: list[tuple[int, int]] = []
        for point in points_3d:
            projected = project_point(*point)
            if projected is None:
                continue
            px, py, _ = projected
            projected_points.append((int(px), int(py)))
        if len(projected_points) >= 3:
            pygame.draw.polygon(screen, WHITE, projected_points, 0)

        collection_projection = project_point(self.x, self.y, self.z)
        if collection_projection:
            px, py, scale = collection_projection
            radius = int(self.collection_radius * scale)
            if radius > 1:
                pygame.draw.circle(screen, (90, 150, 255), (int(px), int(py)), radius, 1)
