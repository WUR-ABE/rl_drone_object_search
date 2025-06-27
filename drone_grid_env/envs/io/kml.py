from __future__ import annotations

from ast import literal_eval
from collections import OrderedDict
from typing import TYPE_CHECKING, cast

import numpy as np

from lxml import etree as ET
from simplekml import AltitudeMode, Coordinates, Kml, LineString, Point, Style

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from collections.abc import Mapping

    from numpy.typing import NDArray


class Coordinates2D:
    def __init__(self, coordinates: Coordinates) -> None:
        self._coords = [(x, y) for x, y, _ in coordinates._coords]

    def __str__(self) -> str:
        if not self._coords:
            return "0.0, 0.0"
        return " ".join(f"{cd[0]},{cd[1]}" for cd in self._coords)


class KMLParseException(Exception):
    pass


def get_namespace(root: ET._Element) -> str:
    return root.tag.split("}")[0].strip("{")


def get_element_value(element: ET._Element, element_name: str, namespaces: dict[str, str] | None = None) -> str:
    if (tag := element.find(element_name, namespaces=namespaces)) is None or (text := tag.text) is None:
        raise KMLParseException(f"Invalid or empty element {element_name}")
    return text


def read_kml[T: np.generic](
    kml_file: Path, dtype: type[T] = np.uint16
) -> tuple[OrderedDict[str, NDArray[T]], list[dict[str, Any]], dict[str, Any]]:
    with kml_file.open("rb") as kml_file_buffer:
        kml_root = ET.fromstring(kml_file_buffer.read())

        namespace = {"kml": get_namespace(kml_root)}
        extended_data = _parse_extended_data(kml_root, namespaces=namespace)

        data = OrderedDict()
        extended_data_per_item = []
        for placemark in kml_root.findall(".//kml:Placemark", namespaces=namespace):
            if (name := placemark.find("kml:name", namespaces=namespace)) is None:
                raise KMLParseException(f"Cannot find 'name' in {placemark}!")

            extended_data_per_item.append(_parse_extended_data(placemark, namespaces=namespace))

            if (point := placemark.find("kml:Point", namespaces=namespace)) is not None:
                coordinates = get_element_value(point, "kml:coordinates", namespaces=namespace)
            elif (linestring := placemark.find("kml:LineString", namespaces=namespace)) is not None:
                coordinates = get_element_value(linestring, "kml:coordinates", namespaces=namespace)
            else:
                raise KMLParseException(f"Cannot find 'point' or 'linestring' in {placemark}!")

            enter_sepered = coordinates.count("\n") > 0
            is_3d = coordinates.strip().split("\n" if enter_sepered else " ")[0].count(",") > 2
            data[name.text] = np.fromstring(
                coordinates.strip().replace("\n", ",") if enter_sepered else coordinates.strip().replace(" ", ","), sep=",", dtype=dtype
            ).reshape(-1, 3 if is_3d else 2)

        return data, extended_data_per_item, extended_data


def write_kml_flightpath(
    output_file: Path,
    data: list[NDArray[np.generic]] | Mapping[str, list[NDArray[np.generic]]],
    extended_data_per_item: list[Mapping[str, Any]] | None = None,
    extended_data: dict[str, Any] = {},
) -> None:
    if extended_data_per_item is not None:
        assert len(data) == len(extended_data_per_item)

    kml = Kml()

    style = Style()
    style.linestyle.color = "ff0000ff"
    style.linestyle.width = 4

    for k, v in extended_data.items():
        v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        kml.document.extendeddata.newdata(k, v)

    for i, (name, flight_path) in enumerate(data.items() if isinstance(data, dict) else enumerate(data)):
        linestring = cast(LineString, kml.newlinestring())
        linestring.name = name if isinstance(data, dict) else f"Episode {i}"
        linestring.coords = flight_path.tolist()

        # Fix coordinates when they are 2D coordinates (skip third coordinate)
        if flight_path.shape[1] == 2:
            linestring._kml["coordinates"] = Coordinates2D(linestring._kml["coordinates"])
        else:
            linestring.altitudemode = AltitudeMode.relativetoground
            linestring.extrude = 0
            linestring.tessellate = 0

        linestring.style = style

        if extended_data_per_item is not None:
            for k, v in extended_data_per_item[i].items():
                v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
                linestring.extendeddata.newdata(k, v)

    kml.save(output_file)


def write_kml_object_locations(
    output_file: Path,
    data: NDArray[np.generic],
    point_extended_data: list[dict[str, Any]] | None = None,
    general_extended_data: dict[str, Any] = {},
) -> None:
    kml = Kml()

    if point_extended_data is not None:
        assert len(point_extended_data) == data.shape[0]

    for k, v in general_extended_data.items():
        v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
        kml.document.extendeddata.newdata(k, v)

    for i in range(data.shape[0]):
        point = cast(Point, kml.newpoint())
        point.name = f"Detection {i}"
        point.coords = [data[i, :].tolist()]

        if point_extended_data is not None:
            for k, v in point_extended_data[i].items():
                v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
                point.extendeddata.newdata(k, v)

        if data.shape[1] == 2:
            point._kml["coordinates"] = Coordinates2D(point._kml["coordinates"])

    kml.save(output_file)


def _parse_extended_data(element: ET._Element, namespaces: dict[str, str] | None = None) -> dict[str, Any]:
    extended_data = {}
    if (extended_data_tag := element.find(".//kml:ExtendedData", namespaces=namespaces)) is not None:
        for data in extended_data_tag.findall("kml:Data", namespaces=namespaces):
            if not data.attrib.has_key("name"):
                raise KMLParseException(f"Extended data element {extended_data_tag} has no name attribute!")

            key = str(data.attrib["name"])
            value = get_element_value(data, "kml:value", namespaces=namespaces)

            try:
                extended_data[key] = literal_eval(value)
            except (ValueError, SyntaxError):
                extended_data[key] = value

    return extended_data


def read_kml_coordinate_map[T: np.generic](
    kml_file: Path,
    map_size: tuple[int, int],
    dtype: type[T] = np.float32,
    ext_data_value_key: str = "confidence",
) -> NDArray[T]:
    coordinate_dict, ext_data, _ = read_kml(kml_file, dtype=np.uint16)
    coordinate_map = np.zeros(map_size, dtype=dtype)
    for i, coordinate in enumerate(coordinate_dict.values()):
        coordinate_map[tuple(coordinate[0])] = ext_data[i][ext_data_value_key]

    return coordinate_map


if __name__ == "__main__":
    from pathlib import Path

    kml_file = Path("evaluations/distributions/distribution_strong-rl.kml")
    print(read_kml(kml_file))
