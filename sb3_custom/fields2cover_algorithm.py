from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import fields2cover as f2c
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm

from drone_grid_env.envs.action import ActionSpace

if TYPE_CHECKING:
    from typing import Any

    from fields2cover import Field, Point, Robot
    from numpy.typing import NDArray

    from stable_baselines3.common.type_aliases import GymEnv

    from drone_grid_env.envs.drone import Drone
    from drone_grid_env.envs.stopping_criteria import StoppingCriteria
    from drone_grid_env.envs.world import World


log = getLogger("__name__")


class NotSupportedException(Exception):
    pass


@dataclass
class ExecutedFlightPath:
    path: NDArray[np.uint16] | None = None
    current_idx: int = 0
    object_map: NDArray[np.bool_] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.bool_))


class Fields2CoverAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        env: GymEnv | str | None,
        sorter: f2c.RP_Single_cell_order_base_class = f2c.RP_Boustrophedon,
        visualize_swaths: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, env, 0, support_multi_env=True, **kwargs)  # type: ignore[arg-type]

        assert self.env is not None

        self._sorter = sorter
        self._visualize_swaths = visualize_swaths
        self._verbose = verbose

        self.flight_paths: list[ExecutedFlightPath] = [ExecutedFlightPath() for _ in range(self.env.num_envs)]

    def learn(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        raise NotSupportedException("Fields2Cover cannot be learned!")

    def predict(self, *args: Any, **kwargs: Any) -> tuple[NDArray[np.int64], tuple[NDArray[np.float32], ...] | None]:
        assert self.env is not None

        actions = np.empty(self.env.num_envs, dtype=np.int64)

        stopping_criterias: list[StoppingCriteria] = self.env.get_attr("stopping_criteria")
        drones: list[Drone] = self.env.get_attr("drone")
        worlds: list[World[Any]] = self.env.get_attr("world")

        for i in range(self.env.num_envs):
            # Detect if there was a reset, generate new path if the environment was reset
            if self.flight_paths[i].path is None or not np.array_equal(self.flight_paths[i].object_map, worlds[i].object_map):
                self.flight_paths[i].path = self.generate_flight_path(drones[i], worlds[i])
                self.flight_paths[i].current_idx = 0
                self.flight_paths[i].object_map = worlds[i].object_map

            actions[i] = self.get_action(self.flight_paths[i], worlds[i], drones[i], stopping_criterias[i])
            assert actions[i] is not None

        return actions, None

    def _setup_model(self) -> None:
        return None

    @classmethod
    def load(cls, *args: Any, env: GymEnv | None = None, **kwargs: Any) -> Fields2CoverAlgorithm:
        return cls(env=env)

    def get_action(
        self, flight_path: ExecutedFlightPath, world: World[Any], drone: Drone, stopping_criteria: StoppingCriteria
    ) -> ActionSpace | int:
        assert flight_path.path is not None

        if len(flight_path.path) == flight_path.current_idx:
            action = self._do_stopping_action(world, drone, stopping_criteria)
            return action

        goal = flight_path.path[flight_path.current_idx, :]
        action = self._move_to_goal(goal, drone.position)

        if action is not None:
            return action

        flight_path.current_idx += 1
        return self.get_action(flight_path, world, drone, stopping_criteria)

    def generate_flight_path(self, drone: Drone, world: World[Any]) -> NDArray[np.uint16]:
        robot = self.robot_from_drone(drone)
        field = self.field_from_world(world)

        # Generate constant headland
        const_hl = f2c.HG_Const_gen()
        no_hl: f2c.Cells = const_hl.generateHeadlands(field.getField(), 0.0)

        # Generate swaths
        bf = f2c.SG_BruteForce()
        bf.setAllowOverlap(True)
        swaths: f2c.Swaths = bf.generateBestSwaths(f2c.OBJ_NSwathModified(), robot.getCovWidth(), no_hl.getGeometry(0))

        # Sort swaths
        sorter = f2c.RP_Boustrophedon()
        swaths = sorter.genSortedSwaths(swaths)

        # Because Fields2Cover works in continious space and we have a discretised space, the last
        # line is not covered. This only happens when the distance between the last swath and the
        # one before last swath is smaller than the robot width. In these cases, move the last swath
        # one to the right. TODO: this does not work in all situations!
        direction = "X" if swaths[swaths.size() - 1].startPoint().getX() == swaths[swaths.size() - 1].endPoint().getX() else "Y"

        def getSwathCoordById(id: int) -> float:
            return float(getattr(swaths[id].startPoint(), f"get{direction}")())

        if swaths.size() > 1 and abs(getSwathCoordById(swaths.size() - 1) - getSwathCoordById(swaths.size() - 2)) < robot.getCovWidth():
            start_point: Point = swaths[swaths.size() - 1].startPoint()
            end_point: Point = swaths[swaths.size() - 1].endPoint()
            getattr(start_point, f"set{direction}")(getattr(start_point, f"get{direction}")() + 1)
            getattr(end_point, f"set{direction}")(getattr(end_point, f"get{direction}")() + 1)

            line_string = f2c.LineString()
            line_string.addPoint(start_point)
            line_string.addPoint(end_point)

            new_swath = f2c.Swath(robot.getCovWidth())
            new_swath.setPath(line_string)

            swaths = self.replace_swath_by_id(swaths, new_swath, swaths.size() - 1)

        # Check whether the start point of the drone is closer to one one of the start/end points
        # and reverse the swaths when needed
        distances = [
            self._distance(drone.position[:2], self._f2c_to_numpy(swaths[0].startPoint())),
            self._distance(drone.position[:2], self._f2c_to_numpy(swaths[0].endPoint())),
            self._distance(drone.position[:2], self._f2c_to_numpy(swaths[swaths.size() - 1].startPoint())),
            self._distance(drone.position[:2], self._f2c_to_numpy(swaths[swaths.size() - 1].endPoint())),
        ]

        if np.argmin(distances) in (1, 3):
            for swath in swaths:
                swath.reverse()

        if np.argmin(distances) in (2, 3):
            swaths.reverse()

        path_planner = f2c.PP_PathPlanning()
        dubins = f2c.PP_DubinsCurves()
        path: f2c.Path = path_planner.planPath(robot, swaths, dubins)
        path = path.discretizeSwath(1.0)

        # Extract waypoints
        flight_path = []
        on_swath = False
        for state in path.getStates():
            if state.type != f2c.PathSectionType_TURN or on_swath:
                flight_path.append(self._f2c_to_numpy(state.point))
                on_swath = state.type != f2c.PathSectionType_TURN

        return np.floor(np.array(flight_path, dtype=np.float32)).astype(np.uint16)

    @staticmethod
    def robot_from_drone(drone: Drone) -> Robot:
        robot = f2c.Robot(float(drone.fov[1]), float(drone.fov[0]))
        robot.setMinTurningRadius(0.0)
        return robot

    @staticmethod
    def field_from_world(world: World[Any]) -> Field:
        ring = f2c.LinearRing()
        ring.addPoint(0, 0)
        ring.addPoint(0, world.size[1] - 1)
        ring.addPoint(world.size[0] - 1, world.size[1] - 1)
        ring.addPoint(world.size[0] - 1, 0)

        # Repeat first point to get a closed ring
        ring.addPoint(0, 0)

        cell = f2c.Cell()
        cell.addRing(ring)

        cells = f2c.Cells()
        cells.addGeometry(cell)

        field = f2c.Field(cells)
        return field

    @staticmethod
    def replace_swath_by_id(swaths: f2c.Swaths, swath: f2c.Swath, idx: int) -> f2c.Swaths:
        new_swaths = f2c.Swaths()
        for i in range(swaths.size()):
            new_swaths.emplace_back(swath if i == idx else swaths[i])

        return new_swaths

    @staticmethod
    def _move_to_goal(point: NDArray[np.uint16], position: NDArray[np.uint16]) -> int | None:
        if position[1] > np.ceil(point[0]):
            return ActionSpace.FLY_WEST

        if position[1] < np.ceil(point[0]):
            return ActionSpace.FLY_EAST

        if position[0] > np.ceil(point[1]):
            return ActionSpace.FLY_NORTH

        if position[0] < np.ceil(point[1]):
            return ActionSpace.FLY_SOUTH

        # If goal reached
        return None

    @staticmethod
    def _do_stopping_action(world: World[Any], drone: Drone, stopping_criteria: StoppingCriteria) -> int | None:
        action = None

        if stopping_criteria.config.method == "land":
            if stopping_criteria.config.only_land_in_zone:
                goal = Fields2CoverAlgorithm._get_closest_landing_spot_path(world, drone.position)
                action = Fields2CoverAlgorithm._move_to_goal(goal, drone.position)

            action = ActionSpace.LAND if action is None else action

        if action is None:
            action = ActionSpace.FLY_EAST  # Just to avoid indexerror

        return action

    @staticmethod
    def _get_closest_landing_spot_path(world: World[Any], position: NDArray[np.uint16]) -> NDArray[np.uint16]:
        land_distances = {}
        for c in zip(*world.start_landing_map.nonzero()):
            land_distances[c] = Fields2CoverAlgorithm._distance(c, position)

        return min(land_distances, key=land_distances.get)  # type: ignore[arg-type]

    @staticmethod
    def _distance(coordinate_1: NDArray[np.uint16], coordinate_2: NDArray[np.uint16]) -> float:
        return np.linalg.norm(coordinate_1.astype(np.int16) - coordinate_2.astype(np.int16)).item()

    @staticmethod
    def _f2c_to_numpy(point: Point) -> NDArray[np.uint16]:
        return np.array([point.getY(), point.getX()], dtype=np.float32)
