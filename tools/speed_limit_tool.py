import json
import math
import re
from typing import Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADIUS_METERS = 40


class SpeedLimitToolInput(BaseModel):
    latitude: float = Field(ge=-90, le=90, description="Current GPS latitude.")
    longitude: float = Field(ge=-180, le=180, description="Current GPS longitude.")


class SpeedLimitTool(BaseTool):
    name: str = "lookup_speed_limit"
    description: str = (
        "Looks up the posted road speed limit near explicit GPS coordinates using "
        "OpenStreetMap Overpass data. Use only when latitude and longitude are "
        "available. The result is informational and may be unavailable."
    )
    args_schema: Type[BaseModel] = SpeedLimitToolInput

    @staticmethod
    def _distance_squared(
        latitude: float, longitude: float, point: dict[str, object]
    ) -> float:
        point_latitude = point.get("lat")
        point_longitude = point.get("lon")
        if not isinstance(point_latitude, (int, float)) or not isinstance(
            point_longitude, (int, float)
        ):
            return math.inf
        latitude_delta = point_latitude - latitude
        longitude_delta = (point_longitude - longitude) * math.cos(
            math.radians(latitude)
        )
        return latitude_delta**2 + longitude_delta**2

    @staticmethod
    def _parse_speed_limit(maxspeed: object) -> float | None:
        if not isinstance(maxspeed, str):
            return None
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(km/h|kph|mph)?\s*", maxspeed.lower()
        )
        if match is None:
            return None
        speed = float(match.group(1))
        return speed * 1.609344 if match.group(2) == "mph" else speed

    def _run(self, latitude: float, longitude: float) -> str:
        query = f"""
            [out:json][timeout:10];
            way(around:{SEARCH_RADIUS_METERS},{latitude},{longitude})[highway][maxspeed];
            out tags geom;
        """
        request_url = f"{OVERPASS_URL}?{urlencode({'data': query})}"
        try:
            with urlopen(request_url, timeout=15) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            return f"Speed limit not available: Overpass lookup failed ({error})."

        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        nearest_speed_limit: float | None = None
        nearest_distance = math.inf
        for element in elements:
            if not isinstance(element, dict):
                continue
            tags = element.get("tags")
            geometry = element.get("geometry")
            if not isinstance(tags, dict) or not isinstance(geometry, list):
                continue
            speed_limit = self._parse_speed_limit(tags.get("maxspeed"))
            if speed_limit is None:
                continue
            distance = min(
                (self._distance_squared(latitude, longitude, point)
                 for point in geometry if isinstance(point, dict)),
                default=math.inf,
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_speed_limit = speed_limit

        if nearest_speed_limit is None:
            return "Speed limit not available from OpenStreetMap near these coordinates."
        return f"Posted speed limit: {nearest_speed_limit:g} km/h (OpenStreetMap)."