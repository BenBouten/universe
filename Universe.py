import math
import random
from typing import List, Sequence, Tuple

import pygame

# Screen and simulation constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
G = 0.08  # Gravitational constant scaled for the simulation (stronger pull)

WORLD_WIDTH = 1000.0
WORLD_HEIGHT = 800.0
WORLD_DEPTH = 1400.0

CAMERA_DISTANCE = 600.0
CAMERA_POS = (0.0, 0.0, -CAMERA_DISTANCE)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 80, 60)
GREEN = (0, 255, 160)
BLUE = (100, 150, 255)
LIGHT_BLUE = (160, 200, 255)


def wrap_coordinate(value: float, limit: float) -> float:
    half = limit / 2.0
    while value < -half:
        value += limit
    while value > half:
        value -= limit
    return value


def vector_length(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    length = vector_length(vector)
    if length == 0.0:
        return 0.0, 0.0, 0.0
    return tuple(component / length for component in vector)


def cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def random_unit_vector() -> Tuple[float, float, float]:
    z = random.uniform(-1.0, 1.0)
    theta = random.uniform(0.0, math.tau)
    radius = math.sqrt(1.0 - z * z)
    return radius * math.cos(theta), radius * math.sin(theta), z


def project_point(x: float, y: float, z: float) -> tuple[float, float, float] | None:
    dx = x - CAMERA_POS[0]
    dy = y - CAMERA_POS[1]
    dz = z - CAMERA_POS[2]
    if dz <= 1.0:
        return None
    scale = CAMERA_DISTANCE / dz
    screen_x = SCREEN_WIDTH / 2 + dx * scale
    screen_y = SCREEN_HEIGHT / 2 + dy * scale
    return screen_x, screen_y, scale


def wrapped_delta(origin: float, target: float, limit: float) -> float:
    diff = target - origin
    half = limit / 2.0
    if diff > half:
        diff -= limit
    elif diff < -half:
        diff += limit
    return diff


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
        glow_color = (min(255, self.color[0] + 80), min(255, self.color[1] + 80), min(255, self.color[2] + 80))
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
        bodies.sort(key=lambda body: body.z - CAMERA_POS[2], reverse=True)
        for body in bodies:
            body.draw(screen)

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
            # Create an orthonormal basis for the orbital plane
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

    def handle_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
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

        cos_pitch = math.cos(self.pitch)
        forward = (
            math.cos(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            math.sin(self.yaw) * cos_pitch,
        )

        thrust_input = 0.0
        if keys[pygame.K_w]:
            thrust_input += 1.0
        if keys[pygame.K_s]:
            thrust_input -= 0.7

        vertical_input = 0.0
        if keys[pygame.K_q]:
            vertical_input += 1.0
        if keys[pygame.K_e]:
            vertical_input -= 1.0

        thrust_acc = self.thrust_force / self.mass
        self.vx += forward[0] * thrust_acc * thrust_input * dt
        self.vy += forward[1] * thrust_acc * thrust_input * dt
        self.vz += forward[2] * thrust_acc * thrust_input * dt

        up_reference = (0.0, 1.0, 0.0)
        vertical_acc = thrust_acc * 0.6
        self.vx += up_reference[0] * vertical_input * vertical_acc * dt
        self.vy += up_reference[1] * vertical_input * vertical_acc * dt
        self.vz += up_reference[2] * vertical_input * vertical_acc * dt

    def apply_gravity(self, universe: "Universe", dt: float) -> None:
        for solar_system in universe.solar_systems:
            bodies: list[Planet] = [solar_system.star, *solar_system.planets]
            for body in bodies:
                dx = wrapped_delta(self.x, body.x, WORLD_WIDTH)
                dy = wrapped_delta(self.y, body.y, WORLD_HEIGHT)
                dz = wrapped_delta(self.z, body.z, WORLD_DEPTH)
                r_sq = dx * dx + dy * dy + dz * dz
                if r_sq < 1.0:
                    continue
                r = math.sqrt(r_sq)
                acceleration = G * body.mass / r_sq
                self.vx += (acceleration * dx / r) * dt
                self.vy += (acceleration * dy / r) * dt
                self.vz += (acceleration * dz / r) * dt

    def update(self, dt: float) -> None:
        self.x = wrap_coordinate(self.x + self.vx * dt, WORLD_WIDTH)
        self.y = wrap_coordinate(self.y + self.vy * dt, WORLD_HEIGHT)
        self.z = wrap_coordinate(self.z + self.vz * dt, WORLD_DEPTH)
        self.vx *= 0.998
        self.vy *= 0.998
        self.vz *= 0.998

    def collect_energy(self, universe: Universe) -> int:
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

    def check_star_collision(self, stars: List[Star]) -> bool:
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


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Newtonian Universe")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    info_font = pygame.font.SysFont("consolas", 20)

    universe = Universe()
    ship = PlayerShip(0.0, 0.0, WORLD_DEPTH * 0.25)
    sector = 1
    energy_goal = 320

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
            energy_goal = int(energy_goal * 1.2)
            ship.sector_progress = 0
            universe.refresh_space()

        universe.draw(screen)
        ship.draw(screen)

        hud_lines = [
            f"Sector: {sector}",
            f"Energie verzameld: {ship.sector_progress}/{energy_goal}",
            f"Totale energie: {ship.total_energy}",
            "Besturing: A/D draaien, W/S thrust",
            "Pijltjes op/af voor pitch, Q/E stijg/daal",
            "Vang planeten, ontwijk sterren",
        ]
        for i, text in enumerate(hud_lines):
            surface = font.render(text, True, WHITE)
            screen.blit(surface, (12, 12 + i * 20))

        info_surface = info_font.render("Newtoniaanse zwaartekracht houdt alles in beweging", True, WHITE)
        screen.blit(info_surface, (12, SCREEN_HEIGHT - 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
