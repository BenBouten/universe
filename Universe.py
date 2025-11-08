import math
import random
from typing import List, Optional

import pygame

# Screen and simulation constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
G = 0.08  # Gravitational constant scaled for the simulation
CAMERA_DISTANCE = 900.0
SPACE_BOUND = 2200.0
Z_BOUND = 1600.0
STAR_MIN_SPACING = 700.0

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 80, 60)
GREEN = (0, 255, 160)
BLUE = (100, 150, 255)
LIGHT_BLUE = (160, 200, 255)
ORANGE = (255, 200, 140)


def random_unit_vector() -> tuple[float, float, float]:
    theta = random.uniform(0.0, math.tau)
    z = random.uniform(-1.0, 1.0)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(theta), r * math.sin(theta), z)


def normalize_vector(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vec
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        return 0.0, 0.0, 0.0
    return x / length, y / length, z / length


def cross_product(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def wrap_range(value: float, minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span == 0:
        return minimum
    return ((value - minimum) % span) + minimum


def project_point(
    x: float,
    y: float,
    z: float,
    camera_position: tuple[float, float, float],
) -> Optional[tuple[float, float, float]]:
    cx, cy, cz = camera_position
    rel_x = x - cx
    rel_y = y - cy
    rel_z = z - cz
    if rel_z <= 50.0:
        return None
    scale = CAMERA_DISTANCE / rel_z
    screen_x = SCREEN_WIDTH / 2 + rel_x * scale
    screen_y = SCREEN_HEIGHT / 2 - rel_y * scale
    return screen_x, screen_y, scale


class Planet:
    """A class that represents a planet orbiting a star in three dimensions."""

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

    def draw(self, screen: pygame.Surface, camera_position: tuple[float, float, float]) -> None:
        projection = project_point(self.x, self.y, self.z, camera_position)
        if projection is None:
            return
        screen_x, screen_y, scale = projection
        radius = max(1, min(60, int(self.radius * scale)))
        pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), radius)

    def apply_force(self, fx: float, fy: float, fz: float, dt: float) -> None:
        self.vx += (fx / self.mass) * dt
        self.vy += (fy / self.mass) * dt
        self.vz += (fz / self.mass) * dt

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

    def attract(self, other: "Planet") -> tuple[float, float, float]:
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
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
        self.radius = max(16, int(self.mass ** (1.0 / 3.0) * 1.6))

    def draw(self, screen: pygame.Surface, camera_position: tuple[float, float, float]) -> None:
        projection = project_point(self.x, self.y, self.z, camera_position)
        if projection is None:
            return
        screen_x, screen_y, scale = projection
        base_radius = max(12, min(90, int(self.radius * scale)))
        glow_radius = int(base_radius * 1.9)
        glow_color = (
            min(255, self.color[0] + 80),
            min(255, self.color[1] + 80),
            min(255, self.color[2] + 80),
        )
        pygame.draw.circle(screen, glow_color, (int(screen_x), int(screen_y)), glow_radius)
        pygame.draw.circle(screen, WHITE, (int(screen_x), int(screen_y)), int(base_radius * 1.15))
        pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), base_radius)

    def update(self, dt: float) -> None:  # noqa: ARG002
        return

    def attract(self, other: Planet) -> tuple[float, float, float]:
        return super().attract(other)


class SolarSystem:
    """A solar system consisting of a star and a collection of planets."""

    def __init__(self, star: Star, planets: Optional[List[Planet]] = None):
        self.star = star
        self.planets: List[Planet] = planets or []

    def add_planet(self, planet: Planet) -> None:
        if planet.orbit_radius is None:
            dx = planet.x - self.star.x
            dy = planet.y - self.star.y
            dz = planet.z - self.star.z
            planet.orbit_radius = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.planets.append(planet)

    def update(self, dt: float) -> None:
        for planet in self.planets:
            fx, fy, fz = self.star.attract(planet)
            planet.apply_force(fx, fy, fz, dt)
            planet.update(dt)

    def draw(self, screen: pygame.Surface, camera_position: tuple[float, float, float]) -> None:
        drawables: list[tuple[float, Planet]] = []
        drawables.append((self.star.z - camera_position[2], self.star))
        for planet in self.planets:
            drawables.append((planet.z - camera_position[2], planet))
        drawables.sort(key=lambda item: item[0], reverse=True)
        for _, body in drawables:
            body.draw(screen, camera_position)

    def is_empty(self) -> bool:
        return not self.planets


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
            x = random.uniform(-SPACE_BOUND * 1.2, SPACE_BOUND * 1.2)
            y = random.uniform(-SPACE_BOUND * 0.8, SPACE_BOUND * 0.8)
            z = random.uniform(-SPACE_BOUND * 0.5, SPACE_BOUND * 1.2)
            radius = random.choice([1, 1, 1, 2])
            brightness = random.randint(150, 255)
            phase = random.random() * math.tau
            stars.append((x, y, z, radius, brightness, phase))
        return stars

    def _create_random_system(self, existing_stars: List[Star]) -> SolarSystem:
        star_mass = random.uniform(75_000, 160_000)
        star_color = random.choice([YELLOW, RED, BLUE, ORANGE])
        star_position = self._pick_star_position(existing_stars)
        star = Star(star_mass, *star_position, star_color)

        solar_system = SolarSystem(star)
        planet_count = random.randint(3, 6)
        for _ in range(planet_count):
            planet_mass = random.uniform(600, 2_400)
            distance = random.uniform(star.radius + 160, star.radius + 420)
            orbit_speed = math.sqrt(G * star.mass / max(distance, 1.0))
            normal = random_unit_vector()
            radial_seed = random_unit_vector()
            radial_direction = normalize_vector(cross_product(normal, radial_seed))
            if radial_direction == (0.0, 0.0, 0.0):
                radial_direction = normalize_vector(random_unit_vector())
            tangential_direction = cross_product(normal, radial_direction)
            tangential_direction = normalize_vector(tangential_direction)
            direction = random.choice([-1, 1])
            px = star.x + radial_direction[0] * distance
            py = star.y + radial_direction[1] * distance
            pz = star.z + radial_direction[2] * distance
            vx = direction * tangential_direction[0] * orbit_speed
            vy = direction * tangential_direction[1] * orbit_speed
            vz = direction * tangential_direction[2] * orbit_speed
            planet_color = random.choice([WHITE, GREEN, LIGHT_BLUE])
            planet = Planet(planet_mass, px, py, pz, vx, vy, vz, planet_color)
            planet.orbit_radius = distance
            solar_system.add_planet(planet)
        return solar_system

    def _pick_star_position(self, existing_stars: List[Star]) -> tuple[float, float, float]:
        for _ in range(80):
            candidate = (
                random.uniform(-SPACE_BOUND * 0.7, SPACE_BOUND * 0.7),
                random.uniform(-SPACE_BOUND * 0.5, SPACE_BOUND * 0.5),
                random.uniform(-Z_BOUND * 0.6, Z_BOUND * 0.8),
            )
            if all(
                math.dist(candidate, (star.x, star.y, star.z)) >= STAR_MIN_SPACING
                for star in existing_stars
            ):
                return candidate
        return (
            random.uniform(-SPACE_BOUND * 0.7, SPACE_BOUND * 0.7),
            random.uniform(-SPACE_BOUND * 0.5, SPACE_BOUND * 0.5),
            random.uniform(-Z_BOUND * 0.6, Z_BOUND * 0.8),
        )

    def refresh_space(self, initial: bool = False) -> None:
        system_target = random.randint(3, 5)
        systems: List[SolarSystem] = []
        for _ in range(system_target):
            system = self._create_random_system([existing.star for existing in systems])
            systems.append(system)
        self.solar_systems = systems
        self.background_color = self._random_background()
        self.backdrop = self._generate_backdrop()
        if not initial:
            pygame.display.set_caption("Newtonian Universe - Nieuwe sector")

    def update(self, dt: float) -> None:
        for solar_system in self.solar_systems:
            solar_system.update(dt)
        self.solar_systems = [s for s in self.solar_systems if not s.is_empty()]

    def draw(self, screen: pygame.Surface, camera_position: tuple[float, float, float]) -> None:
        screen.fill(self.background_color)
        current_time = pygame.time.get_ticks() / 1000.0
        sorted_backdrop = sorted(self.backdrop, key=lambda item: item[2] - camera_position[2], reverse=True)
        for x, y, z, radius, brightness, phase in sorted_backdrop:
            projection = project_point(x, y, z, camera_position)
            if projection is None:
                continue
            screen_x, screen_y, scale = projection
            sparkle = int(25 * math.sin(current_time + phase))
            value = int(max(90, min(255, brightness + sparkle)))
            color = (value, value, value)
            draw_radius = max(1, min(6, int(radius * max(scale, 0.1))))
            pygame.draw.circle(screen, color, (int(screen_x), int(screen_y)), draw_radius)
        for solar_system in self.solar_systems:
            solar_system.draw(screen, camera_position)

    def get_stars(self) -> List[Star]:
        return [solar_system.star for solar_system in self.solar_systems]

    def is_empty(self) -> bool:
        return all(solar_system.is_empty() for solar_system in self.solar_systems)


class PlayerShip:
    """A controllable ship that gathers energy from planets."""

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.angle = -math.pi / 2
        self.mass = 120.0
        self.turn_speed = math.radians(180)
        self.thrust_force = 18000.0
        self.vertical_force = 14000.0
        self.ship_radius = 10
        self.collection_radius = 22
        self.sector_progress = 0
        self.total_energy = 0

    def handle_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angle -= self.turn_speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += self.turn_speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            thrust_ax = math.cos(self.angle) * (self.thrust_force / self.mass)
            thrust_ay = math.sin(self.angle) * (self.thrust_force / self.mass)
            self.vx += thrust_ax * dt
            self.vy += thrust_ay * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            speed = math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)
            if speed > 1e-6:
                retro = self.thrust_force / self.mass
                self.vx -= (self.vx / speed) * retro * dt
                self.vy -= (self.vy / speed) * retro * dt
                self.vz -= (self.vz / speed) * retro * dt
        if keys[pygame.K_r] or keys[pygame.K_SPACE]:
            self.vz += (self.vertical_force / self.mass) * dt
        if keys[pygame.K_f] or keys[pygame.K_LCTRL]:
            self.vz -= (self.vertical_force / self.mass) * dt

    def apply_gravity(self, universe: "Universe", dt: float) -> None:
        for solar_system in universe.solar_systems:
            bodies: list[Planet] = [solar_system.star, *solar_system.planets]
            for body in bodies:
                dx = body.x - self.x
                dy = body.y - self.y
                dz = body.z - self.z
                r_sq = dx * dx + dy * dy + dz * dz
                if r_sq < 4.0:
                    continue
                r = math.sqrt(r_sq)
                acceleration = G * body.mass / r_sq
                self.vx += (acceleration * dx / r) * dt
                self.vy += (acceleration * dy / r) * dt
                self.vz += (acceleration * dz / r) * dt

    def update(self, dt: float) -> None:
        self.x = wrap_range(self.x + self.vx * dt, -SPACE_BOUND, SPACE_BOUND)
        self.y = wrap_range(self.y + self.vy * dt, -SPACE_BOUND, SPACE_BOUND)
        self.z = wrap_range(self.z + self.vz * dt, -Z_BOUND, Z_BOUND)

    def collect_energy(self, universe: Universe) -> int:
        collected = 0
        for solar_system in universe.solar_systems:
            remaining_planets = []
            for planet in solar_system.planets:
                dx = planet.x - self.x
                dy = planet.y - self.y
                dz = planet.z - self.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                if distance <= planet.radius + self.collection_radius:
                    energy_value = max(5, int(planet.mass * 0.18))
                    collected += energy_value
                else:
                    remaining_planets.append(planet)
            solar_system.planets = remaining_planets
        if collected:
            self.sector_progress += collected
            self.total_energy += collected
        return collected

    def check_star_collision(self, stars: List[Star]) -> bool:
        for star in stars:
            dx = star.x - self.x
            dy = star.y - self.y
            dz = star.z - self.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= star.radius + self.ship_radius - 2:
                return True
        return False

    def respawn(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.angle = -math.pi / 2
        self.sector_progress = 0

    def draw(self, screen: pygame.Surface, camera_position: tuple[float, float, float]) -> None:
        projection = project_point(self.x, self.y, self.z, camera_position)
        if projection is None:
            return
        _, _, scale = projection
        forward = (math.cos(self.angle), math.sin(self.angle), 0.0)
        left = (-math.sin(self.angle), math.cos(self.angle), 0.0)
        up = (0.0, 0.0, 1.0)
        base_radius = self.ship_radius * (0.8 + scale * 0.4)
        nose_offset = 2.4 * base_radius
        wing_offset = 1.2 * base_radius
        nose = (
            self.x + forward[0] * nose_offset,
            self.y + forward[1] * nose_offset,
            self.z + forward[2] * nose_offset,
        )
        left_point = (
            self.x + left[0] * wing_offset + up[0] * base_radius * 0.3,
            self.y + left[1] * wing_offset + up[1] * base_radius * 0.3,
            self.z + left[2] * wing_offset + up[2] * base_radius * 0.3,
        )
        right_point = (
            self.x - left[0] * wing_offset + up[0] * base_radius * 0.3,
            self.y - left[1] * wing_offset + up[1] * base_radius * 0.3,
            self.z - left[2] * wing_offset + up[2] * base_radius * 0.3,
        )
        projected_points = []
        for px, py, pz in (nose, left_point, right_point):
            projected = project_point(px, py, pz, camera_position)
            if projected is None:
                return
            screen_x, screen_y, _ = projected
            projected_points.append((int(screen_x), int(screen_y)))
        pygame.draw.polygon(screen, WHITE, projected_points)
        circle_projection = project_point(self.x, self.y, self.z, camera_position)
        if circle_projection is not None:
            screen_x, screen_y, circle_scale = circle_projection
            radius = max(12, int(self.collection_radius * circle_scale))
            pygame.draw.circle(screen, (90, 150, 255), (int(screen_x), int(screen_y)), radius, 1)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Newtonian Universe 3D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    info_font = pygame.font.SysFont("consolas", 20)

    universe = Universe()
    ship = PlayerShip(0.0, 0.0, 0.0)
    sector = 1
    energy_goal = 360

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
            energy_goal = int(energy_goal * 1.22)
            ship.sector_progress = 0
            universe.refresh_space()

        camera_position = (ship.x, ship.y, ship.z - CAMERA_DISTANCE)
        universe.draw(screen, camera_position)
        ship.draw(screen, camera_position)

        hud_lines = [
            f"Sector: {sector}",
            f"Energie verzameld: {ship.sector_progress}/{energy_goal}",
            f"Totale energie: {ship.total_energy}",
            "Besturing: WASD of pijltjes, R/Spatie = stijg, F/Ctrl = daal",
            "Gebruik retrograde remmen (S) om in een baan te komen",
        ]
        for i, text in enumerate(hud_lines):
            surface = font.render(text, True, WHITE)
            screen.blit(surface, (12, 12 + i * 20))

        info_surface = info_font.render(
            "3D Newtoniaanse zwaartekracht houdt alles in beweging",
            True,
            WHITE,
        )
        screen.blit(info_surface, (12, SCREEN_HEIGHT - 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
