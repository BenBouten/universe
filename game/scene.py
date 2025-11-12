"""Game setup and main update loop."""

from __future__ import annotations

import math
import random
from typing import List

import pygame

from core.physics import cross, normalize, random_unit_vector, vector_length, wrap_coordinate
from game.camera import camera, draw_future_path, project_point
from game.constants import (
    BLACK,
    BLUE,
    FPS,
    GREEN,
    LIGHT_BLUE,
    RED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
    WORLD_DEPTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    YELLOW,
    G,
)
from ui.hud import draw_hud
from world.bodies import Planet, SolarSystem, Star
from world.ship import PlayerShip


class Universe:
    """A universe containing multiple solar systems and decorative background stars."""

    def __init__(self) -> None:
        self.solar_systems: List[SolarSystem] = []
        self.background_color = BLACK
        self.backdrop: list[tuple[float, float, float, int, int, float]] = []
        self.refresh_space(initial=True)

    def _random_background(self) -> tuple[int, int, int]:
        base = random.randint(0, 30)
        return (
            base,
            base + random.randint(10, 40),
            base + random.randint(30, 70),
        )

    def _generate_backdrop(self) -> list[tuple[float, float, float, int, int, float]]:
        stars = []
        for _ in range(140):
            x = random.uniform(-WORLD_WIDTH / 2, WORLD_WIDTH / 2)
            y = random.uniform(-WORLD_HEIGHT / 2, WORLD_HEIGHT / 2)
            z = random.uniform(200.0, WORLD_DEPTH)
            radius = random.choice([1, 1, 2])
            brightness = random.randint(140, 255)
            phase = random.random() * math.tau
            stars.append((x, y, z, radius, brightness, phase))
        return stars

    def _create_random_system(self) -> SolarSystem:
        star_mass = random.uniform(8_000, 24_000)
        star_x = random.uniform(-WORLD_WIDTH / 3, WORLD_WIDTH / 3)
        star_y = random.uniform(-WORLD_HEIGHT / 3, WORLD_HEIGHT / 3)
        star_z = random.uniform(WORLD_DEPTH * 0.25, WORLD_DEPTH * 0.75)
        star_color = random.choice([YELLOW, RED, BLUE, (255, 220, 150)])
        star = Star(star_mass, star_x, star_y, star_z, star_color)

        solar_system = SolarSystem(star)
        for _ in range(random.randint(2, 6)):
            planet_mass = random.uniform(40, 160)
            distance = random.uniform(star.radius + 70, star.radius + 300)
            normal = random_unit_vector()
            if abs(normal[0]) < 0.8:
                helper = (1.0, 0.0, 0.0)
            else:
                helper = (0.0, 1.0, 0.0)
            u = normalize(cross(normal, helper))
            if vector_length(u) < 1e-5:
                helper = (0.0, 0.0, 1.0)
                u = normalize(cross(normal, helper))
            v = normalize(cross(normal, u))
            angle = random.uniform(0.0, math.tau)
            offset = (
                u[0] * math.cos(angle) + v[0] * math.sin(angle),
                u[1] * math.cos(angle) + v[1] * math.sin(angle),
                u[2] * math.cos(angle) + v[2] * math.sin(angle),
            )
            px = wrap_coordinate(star.x + offset[0] * distance, WORLD_WIDTH)
            py = wrap_coordinate(star.y + offset[1] * distance, WORLD_HEIGHT)
            pz = wrap_coordinate(star.z + offset[2] * distance, WORLD_DEPTH)
            orbit_speed = math.sqrt(G * star.mass / max(distance, 1.0))
            tangent = cross(normal, offset)
            tangent = normalize(tangent)
            direction = random.choice([-1.0, 1.0])
            vx = direction * tangent[0] * orbit_speed
            vy = direction * tangent[1] * orbit_speed
            vz = direction * tangent[2] * orbit_speed
            planet_color = random.choice([WHITE, GREEN, LIGHT_BLUE])
            planet = Planet(planet_mass, px, py, pz, vx, vy, vz, planet_color)
            planet.orbit_radius = distance
            solar_system.add_planet(planet)
        return solar_system

    def refresh_space(self, initial: bool = False) -> None:
        self.solar_systems = [self._create_random_system() for _ in range(random.randint(3, 6))]
        self.background_color = self._random_background()
        self.backdrop = self._generate_backdrop()
        if not initial:
            pygame.display.set_caption("Newtonian Universe - Nieuwe sector")

    def update(self, dt: float) -> None:
        for solar_system in self.solar_systems:
            solar_system.update(dt)
        self.solar_systems = [s for s in self.solar_systems if not s.is_empty()]

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(self.background_color)
        current_time = pygame.time.get_ticks() / 1000.0
        for x, y, z, radius, brightness, phase in self.backdrop:
            projected = project_point(x, y, z)
            if projected is None:
                continue
            px, py, scale = projected
            value = int(max(60, min(255, brightness + 40 * math.sin(current_time + phase))))
            color = (value, value, value)
            size = max(1, int(radius * max(scale, 0.2)))
            pygame.draw.circle(screen, color, (int(px), int(py)), size)
        for solar_system in self.solar_systems:
            solar_system.draw(screen)

    def get_stars(self) -> List[Star]:
        return [solar_system.star for solar_system in self.solar_systems]

    def is_empty(self) -> bool:
        return all(solar_system.is_empty() for solar_system in self.solar_systems)


def run_game() -> None:
    """Start the Newtonian universe simulation."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Newtonian Universe")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    info_font = pygame.font.SysFont("consolas", 20)

    universe = Universe()
    ship = PlayerShip(0.0, 0.0, WORLD_DEPTH * 0.25)
    camera.set_focus_target(ship.get_position())
    camera.update()
    sector = 1
    energy_goal = 320

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    ship.toggle_control_mode()
                elif event.key == pygame.K_f:
                    camera.toggle_focus()
                elif event.key == pygame.K_x:
                    ship.cut_throttle()
            elif event.type == pygame.MOUSEMOTION and camera.focus_mode:
                camera.handle_mouse_motion(event.rel[0], event.rel[1])

        keys = pygame.key.get_pressed()
        ship.handle_input(keys, dt)
        ship.apply_gravity(universe, dt)
        universe.update(dt)
        ship.update(dt)
        ship.collect_energy(universe)

        if ship.check_star_collision(universe.get_stars()):
            ship.respawn()

        if ship.sector_progress >= energy_goal or universe.is_empty():
            sector += 1
            energy_goal = int(energy_goal * 1.2)
            ship.sector_progress = 0
            universe.refresh_space()

        camera.set_focus_target(ship.get_position())
        camera.update()
        future_path = ship.predict_future_path(universe, duration=180.0, step=1.0)

        universe.draw(screen)
        draw_future_path(screen, future_path)
        ship.draw(screen)
        draw_hud(screen, font, info_font, ship, sector, energy_goal)

        pygame.display.flip()

    pygame.quit()
