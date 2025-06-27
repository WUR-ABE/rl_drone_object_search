from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING

from gymnasium.error import DependencyNotInstalled
import numpy as np

from drone_grid_env.envs.action import ActionSpace
from drone_grid_env.envs.utils import get_file

# Allow to use without pygame, give import error at rendering
try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from typing import Any, Literal
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from pygame import Color, Surface
    from pygame.font import Font
    from pygame.time import Clock

    from drone_grid_env.envs.drone import Drone
    from drone_grid_env.envs.world import World

    RGBAOutput = tuple[int, int, int, int]
    ColorValue = Color | int | str | tuple[int, int, int] | RGBAOutput | Sequence[int]


VIEWPORT_W = 600
VIEWPORT_H = 400


@dataclass
class RenderingConfig:
    world_image_padding: int = 20
    drone_image_size: NDArray[np.uint8] = field(default_factory=lambda: np.array((3, 3), dtype=np.uint8))
    drone_scale_factor: NDArray[np.uint8] = field(default_factory=lambda: np.array((5, 5), dtype=np.uint8))
    render_fps: int = 30

    # Colors
    fov_color: Color = field(default_factory=lambda: pygame.Color(0, 0, 255))
    flight_path_color: Color = field(default_factory=lambda: pygame.Color(0, 0, 255))
    action_color: Color = field(default_factory=lambda: pygame.Color(255, 0, 0))
    text_color: Color = field(default_factory=lambda: pygame.Color(0, 0, 0))
    classified_color: Color = field(default_factory=lambda: pygame.Color(50, 50, 50))
    object_color: Color = field(default_factory=lambda: pygame.Color(128, 128, 128))
    background_color: Color = field(default_factory=lambda: pygame.Color(180, 180, 180))
    start_landing_zone_color: Color = field(default_factory=lambda: pygame.Color(30, 144, 255, 128))


class Rendering:
    def __init__(
        self,
        world: World[Any],
        drone: Drone,
        config: RenderingConfig = RenderingConfig(),
    ) -> None:
        self._world = world
        self._drone = drone
        self.config = config

        self._world_scale = np.zeros(2, dtype=np.float32)

        self._window: Surface | None = None
        self._clock: Clock | None = None
        self._font: Font | None = None
        self._isopen = False

        self._drone_element: Surface | None = None
        self._frame_buffer: list[NDArray[np.uint8]] = []

    def render(
        self,
        last_state: dict[str, NDArray[np.uint8]] | None,
        last_action: int | None,
        info: dict[str, Any] | None,
        mode: Literal["human", "rgb_array", "rgb_array_headless", "rgb_array_list", "rgb_array_list_headless"],
    ) -> NDArray[np.uint8] | list[NDArray[np.uint8]] | None:
        if pygame is None:
            raise DependencyNotInstalled("pygame not installed. Run 'pip install pygame'.")

        screen_size = (VIEWPORT_W, VIEWPORT_H)

        if self._window is None:
            if mode.endswith("_headless"):
                # https://www.pygame.org/wiki/HeadlessNoWindowsNeeded
                os.environ["SDL_VIDEODRIVER"] = "dummy"

            pygame.init()
            pygame.display.init()
            pygame.font.init()

            if self._world.additional_visualisation is not None:
                self._window = pygame.display.set_mode((screen_size[0] + screen_size[1], screen_size[1]), pygame.RESIZABLE)
            else:
                self._window = pygame.display.set_mode(screen_size, pygame.RESIZABLE)

            pygame.display.set_caption("DroneGridEnv")

        if self._drone_element is None:
            self._drone_element = pygame.image.load(get_file("drone.png")).convert_alpha()

        if self._font is None:
            self._font = pygame.font.SysFont("Calibri", 15)

        if self._clock is None and mode == "human":
            self._clock = pygame.time.Clock()

        if mode == "human":
            assert self._window is not None
            screen_size = self._window.get_size()  # Check for updates in screen size

        canvas = pygame.Surface(screen_size)
        canvas.fill((255, 255, 255))

        # Add world with drone, flight path and fov
        world_canvas = canvas.subsurface(pygame.Rect(0, 0, 0.74 * screen_size[0], screen_size[1]))
        self.world_to_surface(world_canvas)
        self.classified_pixels_to_surface(world_canvas)
        if last_state is not None and last_state["local_map"].shape[0] == 4:  # When state contains start-landing map
            self.start_landing_zone_to_surface(world_canvas)
        self.drone_to_surface(world_canvas)
        self.flight_path_to_surface(world_canvas)
        self.fov_to_surface(world_canvas)

        if last_action is not None:
            self.action_to_surface(last_action, world_canvas)

        state_canvas = canvas.subsurface(pygame.Rect(0.76 * screen_size[0], 20, 0.24 * screen_size[0] - 20, screen_size[1] - 20))
        if last_state is not None:
            self.state_to_surface(last_state, state_canvas)

        # Add env info
        if info is not None:
            info_canvas = canvas.subsurface(pygame.Rect(0, screen_size[1] - 15, screen_size[0], 15))
            self.info_to_surface(info, info_canvas)

        if self._world.additional_visualisation is not None:
            main_canvas = pygame.Surface((screen_size[0] + screen_size[1], screen_size[1]))
            main_canvas.blit(canvas, pygame.Rect(0, 0, screen_size[0], screen_size[1]))

            additional_canvas = main_canvas.subsurface(pygame.Rect(screen_size[0], 0, screen_size[1], screen_size[1]))
            self.image_and_title_to_surface(self._world.additional_visualisation, "Drone image", additional_canvas)
        else:
            main_canvas = canvas

        if mode == "human":
            assert self._window is not None
            assert self._clock is not None

            self._window.blit(main_canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.config.render_fps)

            return None

        frame = pygame.surfarray.pixels3d(main_canvas).transpose(1, 0, 2)

        # Add frame to frame buffer
        self._frame_buffer.append(frame)

        if mode.startswith("rgb_array_list"):
            return self._frame_buffer

        return frame

    def reset(self) -> None:
        self._frame_buffer.clear()

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.font.quit()
            pygame.display.quit()
            pygame.quit()

            self._isopen = False

    def world_to_surface(self, surface: Surface) -> None:
        """
        Adds a numpy array to a surface and create a custom coordinate system on the image.

        :param surface: Surface to draw world on.
        """
        world = np.full((*self._world.size, 3), self.config.background_color[:3], dtype=np.uint8)
        world[self._world.object_map] = self.config.object_color[:3]

        world_shape = self.np2pygame_coordinates(world.shape)[:2]
        world_element = pygame.Surface(world_shape)
        pygame.surfarray.blit_array(world_element, np.transpose(world, (1, 0, 2)))

        # Aspect resize and add on surface
        new_dim = self._fit_in_size(world_shape, surface.get_size(), padding=self.config.world_image_padding)  # type: ignore[arg-type]
        world_element = pygame.transform.scale(world_element, new_dim)
        surface.blit(world_element, (self.config.world_image_padding, self.config.world_image_padding))

        # Update world scale
        self._world_scale = np.array([new_dim[0] / world_shape[0], new_dim[1] / world_shape[1]])

    def start_landing_zone_to_surface(self, surface: Surface) -> None:
        for zone in self._world.config.start_landing_zones:
            size = self.np2pygame_coordinates(zone.shape)[:2] * self._world_scale
            position = self.np2pygame_coordinates((zone.x, zone.y))
            world_position = self.world2pixel_coordinates(position, center_correction=False)  # type: ignore[arg-type]

            zone_surface = pygame.Surface(size)
            zone_surface.fill(self.config.start_landing_zone_color)
            zone_surface.set_alpha(self.config.start_landing_zone_color.a)

            surface.blit(zone_surface, world_position)

    def drone_to_surface(
        self,
        surface: Surface,
        scale_factor: int = 1,
        color: ColorValue | None = None,
        custom_position: NDArray[np.uint16] | None = None,
    ) -> None:
        """
        Adds the image of a drone on top of a world.

        :param drone_position: Position of the drone in world coordinates.
        :param surface: Surface to draw the drone on.
        :param scale_factor: Scale factor of the drone image.
        :param color: Overrides the RGB value of the drone image.
        :param custom_position: Overrides the drones center position.
        """
        assert self._drone_element is not None

        # Update drone size
        new_dim = np.maximum(
            self.config.drone_image_size * scale_factor * self._world_scale,
            self.config.drone_scale_factor * scale_factor,
        ).astype(np.uint16)
        drone_element = pygame.transform.scale(self._drone_element, new_dim.tolist())

        if color is not None:
            self.fill_element(color, drone_element)

        # Update drone position
        position = self.np2pygame_coordinates(self._drone.position if custom_position is None else custom_position)[:2]
        world_position = self.world2pixel_coordinates(position) - new_dim / 2  # type: ignore[arg-type]

        surface.blit(drone_element, world_position.tolist())

    def flight_path_to_surface(
        self,
        surface: Surface,
        color: ColorValue | None = None,
        width: int = 3,
        custom_path: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        Function to draw a flight path on the world.

        :param surface: The surface to draw the flight path on.
        :param color: Overwrites the default color.
        :param width: Width of the flight path.
        :param custom_path: Overrides the drones flight path.
        """
        flight_path = self._drone.flight_path if custom_path is None else custom_path

        if len(flight_path) < 2:
            return

        _color = self.config.flight_path_color if color is None else color

        path = []
        for p in flight_path:
            position = self.np2pygame_coordinates(p)
            path.append(self.world2pixel_coordinates(position, center_correction=True))  # type: ignore[arg-type]

        pygame.draw.lines(surface, _color, False, path, width=width)

    def fov_to_surface(
        self, surface: Surface, color: ColorValue | None = None, width: int = 2, custom_position: NDArray[np.uint16] | None = None
    ) -> None:
        """
        Function to draw a rectangle on the world to indicate the world-Of-View (FOV).

        :param surface: The surface to draw the FOV rectangle on.
        :param fov: The FOV to draw on the surface.
        :param color: Overwrites the default color.
        :param width: Width of the FOV line.
        :param custom_position: Overwrites the FOV center position.
        """
        _color = self.config.fov_color if color is None else color

        center_position = self._drone.position if custom_position is None else custom_position
        tl = self.np2pygame_coordinates(center_position.astype(np.int16) - np.array(self._drone.fov) // 2)

        tl_world = self.world2pixel_coordinates(tl, center_correction=False)  # type: ignore[arg-type]
        square_dim = self.np2pygame_coordinates(self._drone.fov) * self._world_scale

        rect = pygame.Rect(tl_world[0], tl_world[1], square_dim[0], square_dim[1])
        pygame.draw.rect(surface, _color, rect, width=width)

    def state_to_surface(self, state: dict[str, NDArray[np.uint8]], surface: Surface) -> None:
        def _state_to_image(_state: NDArray[np.uint8], local: bool = False) -> NDArray[np.uint8]:
            if local:
                state_image = np.zeros((*_state.shape[1:], 3), dtype=np.uint8)
                state_image.fill(100)
                state_image[_state[0, :, :] > 0] = (255, 255, 0)  # All weeds + prior knowledge
            else:
                _state_image = np.stack((_state[0, :, :],) * 3, axis=-1)
                state_image = ((_state_image * (155 / 255)) + 100).astype(np.uint8)
                # print(np.unique(_state_image.reshape(-1, 3), axis=0))
                # print("---")
                # print(np.unique(state_image.reshape(-1, 3), axis=0))
                # exit()

            state_image[_state[1, :, :] > 0] = (0, 0, 0)  # No-fly-zone
            state_image[_state[2, :, :] > 0] = (255, 0, 255)  # Discovered weeds

            if _state.shape[0] == 4:
                state_image[_state[3, :, :] > 0] = (0, 0, 255)  # Start-landing zone

            return state_image

        global_image = _state_to_image(state["global_map"], local=False)
        local_image = _state_to_image(state["local_map"], local=True)

        self._draw_image_and_title(global_image, "global map", surface)
        self._draw_image_and_title(
            local_image,
            "local map",
            surface.subsurface(0, 0.5 * surface.get_height(), surface.get_width(), 0.5 * surface.get_height()),
        )

    def action_to_surface(self, action: int, surface: Surface, step_size: int = 1, color: ColorValue | None = None) -> None:
        """
        Function to draw the last action to the surface. Only works if action_map is defined during initialisation of
        this class.

        :param action: The last action that was done.
        :param surface: The surface on which the drone is redered.
        :param step_size: How many steps the drone takes in a move action (probably the altitude).
        :param color: Overwrites the default color.
        """
        _color = self.config.action_color if color is None else color
        world_position = self.np2pygame_coordinates(self._drone.position)[:2]

        # Get previous drone position
        previous_world_position = None
        if action == ActionSpace.FLY_NORTH:
            previous_world_position = (world_position[0], world_position[1] + step_size)
        elif action == ActionSpace.FLY_SOUTH:
            previous_world_position = (world_position[0], world_position[1] - step_size)
        elif action == ActionSpace.FLY_EAST:
            previous_world_position = (world_position[0] - step_size, world_position[1])
        elif action == ActionSpace.FLY_WEST:
            previous_world_position = (world_position[0] + step_size, world_position[1])
        elif action == ActionSpace.FLY_NORTH_WEST:
            previous_world_position = (world_position[0] + step_size, world_position[1] + step_size)
        elif action == ActionSpace.FLY_NORTH_EAST:
            previous_world_position = (world_position[0] - step_size, world_position[1] + step_size)
        elif action == ActionSpace.FLY_SOUTH_WEST:
            previous_world_position = (world_position[0] + step_size, world_position[1] - step_size)
        elif action == ActionSpace.FLY_SOUTH_EAST:
            previous_world_position = (world_position[0] - step_size, world_position[1] - step_size)

        if previous_world_position is not None:
            self.draw_arrow(
                self.world2pixel_coordinates(previous_world_position),
                self.world2pixel_coordinates(world_position),  # type: ignore[arg-type]
                surface,
                line_color=_color,
                tri_color=_color,
            )

    def classified_pixels_to_surface(self, surface: Surface, color: ColorValue | None = None) -> None:
        _color = self.config.classified_color if color is None else color

        for position in self._drone.found_object_positions:
            image_position = self.world2pixel_coordinates(position, center_correction=False)
            rect = pygame.Rect(
                image_position[1],
                image_position[0],
                np.ceil(self._world_scale[1]) + 1,
                np.ceil(self._world_scale[0]) + 1,
            )
            pygame.draw.rect(surface, _color, rect)

    def image_and_title_to_surface(self, image: NDArray[np.uint8], title: str, surface: Surface, color: ColorValue | None = None) -> None:
        """
        Function to draw an image with a title on a surface.

        :param image: The image to draw.
        :param title: The title that should be above the image.
        :param surface: The surface to draw the image and title on.
        :param color: Overwrites the default color.
        """
        assert self._font is not None

        surface.fill(pygame.Color(255, 255, 255))

        _color = self.config.text_color if color is None else color
        text_element = self._font.render(title, False, _color)
        surface.blit(text_element, (0, 0))

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        capture_shape = self.np2pygame_coordinates(image.shape)[:2]
        capture_element = pygame.Surface(capture_shape)
        pygame.surfarray.blit_array(capture_element, np.transpose(image, (1, 0, 2)))

        max_size = surface.get_size()
        new_dim = self._fit_in_size(capture_shape, (max_size[0], max_size[1] - 15))  # type: ignore[arg-type]
        capture_element = pygame.transform.scale(capture_element, new_dim)
        surface.blit(capture_element, (0, 15))

    def info_to_surface(self, info: dict[str, Any], surface: Surface, color: ColorValue | None = None) -> None:
        """
        Function that renders a dictionary as key:  value on a surface.

        :param info: The information dict.
        :param surface: The surface to draw the text on.
        :param color: Overwrites the default color.
        """
        assert self._font is not None

        _color = self.config.text_color if color is None else color
        text = ""
        for key, value in info.items():
            if isinstance(value, float):
                text += f"{key.capitalize()}: {value:.2f}   "
            else:
                text += f"{key.capitalize()}: {str(value)}   "

        if text:
            text_element = self._font.render(text, False, _color)
            surface.blit(text_element, (0, 0))

    def _draw_image_and_title(
        self,
        image: NDArray[np.uint8],
        title: str,
        surface: Surface,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Function to draw an image with a title on a surface.

        :param image: The image to draw.
        :param title: The title that should be above the image.
        :param surface: The surface to draw the image and title on.
        :param color: Overwrites the default color.
        :param height: Custom start height.
        """
        _color = self.config.text_color if color is None else color
        text_element = self._font.render(title, False, _color)  # type: ignore[union-attr,arg-type]
        surface.blit(text_element, (0, 0))

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        capture_shape = self.np2pygame_coordinates(image.shape)[:2]
        capture_element = pygame.Surface(capture_shape)
        pygame.surfarray.blit_array(capture_element, np.transpose(image, (1, 0, 2)))

        max_size = surface.get_size()
        new_dim = self._fit_in_size(capture_shape, (max_size[0], max_size[1] - 15))  # type: ignore[arg-type]
        capture_element = pygame.transform.scale(capture_element, new_dim)
        surface.blit(capture_element, (0, 15))

    def world2pixel_coordinates(self, world_coordinate: tuple[int, int] | NDArray[Any], center_correction: bool = True) -> tuple[int, int]:
        """
        Function that converts world to pixel coordinates.

        :param world_coordinate: The world coordinate.
        :param center_correction: True if the pixel coordinate should be on the center of the
                                  world coordinate instead of the top-left corner, False otherwise.
        :returns: The pixel coordinates.
        """
        pixel_center_correction = 0.5 * self._world_scale if center_correction else np.zeros(2, dtype=np.uint8)
        new_x = round(self._world_scale[0] * world_coordinate[0] + self.config.world_image_padding + pixel_center_correction[0])
        new_y = round(self._world_scale[1] * world_coordinate[1] + self.config.world_image_padding + pixel_center_correction[1])
        return new_x, new_y

    @staticmethod
    def fill_element(color: ColorValue, surface: Surface) -> None:
        """Fill all pixels of the surface with color, preserve transparency."""
        w, h = surface.get_size()

        for x in range(w):
            for y in range(h):
                a = surface.get_at((x, y))[3]

                if not isinstance(color, pygame.Color):
                    color = pygame.Color(color)

                transparant_color = pygame.Color(color.r, color.g, color.b, a)
                surface.set_at((x, y), transparant_color)

    @staticmethod
    def _fit_in_size(
        size: tuple[int, int],
        target_size: tuple[int, int],
        padding: int = 0,
    ) -> tuple[int, int]:
        """
        Function that returns the best fitting size of an image while keeping it's aspect ratio.

        :param size: The current size of the image.
        :param target_size: The maximum size of the space where the image should be rendered.
        :param padding: Optional padding around the image.
        :returns: The size that fists best in the given target_size.
        """
        target_ratio = target_size[1] / target_size[0]
        ratio = size[1] / size[0]
        if target_ratio > ratio:
            resize_width = target_size[0] - 2 * padding
            resize_height = round(resize_width * ratio)
        else:
            resize_height = target_size[1] - 2 * padding
            resize_width = round(resize_height / ratio)
        return resize_width, resize_height

    @staticmethod
    def np2pygame_coordinates(p: tuple[int, ...] | NDArray[Any]) -> tuple[int, ...]:
        """
        Function that converts a numpy coordinate (H, W, ...) to a pygame coordinate (W, H, ...)

        :param p: The numpy coordinate.
        :returns: The pygame coordinate.
        """
        return (p[1], p[0], *p[2:])

    @staticmethod
    def draw_arrow(
        start: tuple[int, int],
        end: tuple[int, int],
        surface: Surface,
        line_color: ColorValue = pygame.Color(0, 0, 0),
        tri_color: ColorValue = pygame.Color(0, 0, 0),
        trirad: int = 4,
        line_width: int = 2,
    ) -> None:
        pygame.draw.line(surface, line_color, start, end, line_width)
        rotation = np.rad2deg(np.arctan2(start[1] - end[1], end[0] - start[0])) + 90
        pygame.draw.polygon(
            surface,
            tri_color,
            (
                (
                    end[0] + trirad * np.sin(np.deg2rad(rotation)),
                    end[1] + trirad * np.cos(np.deg2rad(rotation)),
                ),
                (
                    end[0] + trirad * np.sin(np.deg2rad(rotation - 120)),
                    end[1] + trirad * np.cos(np.deg2rad(rotation - 120)),
                ),
                (
                    end[0] + trirad * np.sin(np.deg2rad(rotation + 120)),
                    end[1] + trirad * np.cos(np.deg2rad(rotation + 120)),
                ),
            ),
        )
