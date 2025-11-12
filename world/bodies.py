"""Celestial bodies used in the Newtonian universe simulation."""

from __future__ import annotations

import math
from typing import List

import pygame

from core.physics import wrap_coordinate, wrapped_delta
from game.camera import camera, project_point
from game.constants import WHITE, WORLD_DEPTH, WORLD_HEIGHT, WORLD_WIDTH, G


class Planet:
    """A class that represents a planet orbiting a star."""

    def __init__(
        self,
        mass: float,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        color: tuple[int, int, int],
    ):
        self.mass = mass
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.color = color
        self.radius = max(2, int(self.mass ** (1.0 / 3.0)))
        self.orbit_radius: float | None = None

    def draw(self, screen: pygame.Surface) -> None:
        projected = project_point(self.x, self.y, self.z)
        if projected is None:
            return
        px, py, scale = projected
        radius = max(1, int(self.radius * scale))
        pygame.draw.circle(screen, self.color, (int(px), int(py)), radius)

    def apply_force(self, fx: float, fy: float, fz: float, dt: float) -> None:
        self.vx += (fx / self.mass) * dt
        self.vy += (fy / self.mass) * dt
        self.vz += (fz / self.mass) * dt

    def update(self, dt: float) -> None:
        self.x = wrap_coordinate(self.x + self.vx * dt, WORLD_WIDTH)
        self.y = wrap_coordinate(self.y + self.vy * dt, WORLD_HEIGHT)
        self.z = wrap_coordinate(self.z + self.vz * dt, WORLD_DEPTH)

    def attract(self, other: "Planet") -> tuple[float, float, float]:
        dx = wrapped_delta(self.x, other.x, WORLD_WIDTH)
        dy = wrapped_delta(self.y, other.y, WORLD_HEIGHT)
        dz = wrapped_delta(self.z, other.z, WORLD_DEPTH)
        r_sq = dx * dx + dy * dy + dz * dz
        if r_sq < 1e-4:
            return 0.0, 0.0, 0.0
        r = math.sqrt(r_sq)
        force = G * self.mass * other.mass / r_sq
        fx = force * dx / r
        fy = force * dy / r
        fz = force * dz / r
        return fx, fy, fz


class Star(Planet):
    """A subclass of Planet that represents a stationary star."""

    def __init__(self, mass: float, x: float, y: float, z: float, color: tuple[int, int, int]):
        super().__init__(mass, x, y, z, 0.0, 0.0, 0.0, color)
        self.radius = max(12, int(self.mass ** (1.0 / 3.0) * 1.4))

    def draw(self, screen: pygame.Surface) -> None:
        projected = project_point(self.x, self.y, self.z)
        if projected is None:
            return
        px, py, scale = projected
        radius = max(6, int(self.radius * scale))
        glow_radius = int(radius * 1.8)
        glow_color = (
            min(255, self.color[0] + 80),
            min(255, self.color[1] + 80),
            min(255, self.color[2] + 80),
        )
        pygame.draw.circle(screen, glow_color, (int(px), int(py)), glow_radius)
        pygame.draw.circle(screen, WHITE, (int(px), int(py)), int(radius * 1.1))
        pygame.draw.circle(screen, self.color, (int(px), int(py)), radius)

    def update(self, dt: float) -> None:  # noqa: ARG002
        # Stars stay fixed in this simulation
        return

    def attract(self, other: Planet) -> tuple[float, float]:
        return super().attract(other)


class SolarSystem:
    """A solar system consisting of a star and a collection of planets."""

    def __init__(self, star: Star, planets: List[Planet] | None = None):
        self.star = star
        self.planets: List[Planet] = planets or []

    def add_planet(self, planet: Planet) -> None:
        if planet.orbit_radius is None:
            dx = wrapped_delta(self.star.x, planet.x, WORLD_WIDTH)
            dy = wrapped_delta(self.star.y, planet.y, WORLD_HEIGHT)
            dz = wrapped_delta(self.star.z, planet.z, WORLD_DEPTH)
            planet.orbit_radius = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.planets.append(planet)

    def update(self, dt: float) -> None:
        for planet in self.planets:
            fx, fy, fz = self.star.attract(planet)
            planet.apply_force(fx, fy, fz, dt)
            planet.update(dt)

    def draw(self, screen: pygame.Surface) -> None:
        bodies: list[Planet] = [self.star, *self.planets]
        bodies.sort(key=lambda body: camera.depth_of((body.x, body.y, body.z)), reverse=True)
        for body in bodies:
            body.draw(screen)

    def is_empty(self) -> bool:
        return not self.planets
