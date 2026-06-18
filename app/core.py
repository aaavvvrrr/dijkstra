import math
import heapq
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0

@dataclass(frozen=True)
class Point:
    row: int
    col: int

@dataclass
class RouteResult:
    path_coords: List[Tuple[float, float]]  # [(lon, lat)]
    total_distance_km: float
    total_time_hours: float

    def to_geojson(self) -> Dict[str, Any]:
        """Конвертирует результат в GeoJSON Feature с очисткой от numpy типов."""
        
        # 1. Явно приводим каждую координату к стандартному Python float
        clean_coords = [[float(lon), float(lat)] for lon, lat in self.path_coords]
        
        # 2. Явно приводим скалярные значения к стандартному Python float
        dist = float(self.total_distance_km)
        time = float(self.total_time_hours)
        speed = float(dist / time) if time > 0 else 0.0

        return {
            "type": "Feature",
            "properties": {
                "distance_km": round(dist, 2),
                "time_hours": round(time, 2),
                "avg_speed_kmh": round(speed, 2)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": clean_coords
            }
        }

class SphericalRasterRouter:
    def __init__(self, speed_raster: np.ndarray, lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float]):
        self.speed_raster = speed_raster.astype(np.float32, copy=False)
        self.rows, self.cols = self.speed_raster.shape
        self.min_lat, self.max_lat = lat_bounds
        self.min_lon, self.max_lon = lon_bounds
        
        self.d_lat = (self.max_lat - self.min_lat) / self.rows
        self.d_lon = (self.max_lon - self.min_lon) / self.cols
        self.dy_km = (math.radians(self.d_lat) * EARTH_RADIUS_KM)
        
        lats = np.linspace(self.max_lat, self.min_lat, self.rows)
        self.cos_lats = np.cos(np.radians(lats)).astype(np.float32)

    def _coord_to_index(self, lon: float, lat: float) -> Point:
        row = int((self.max_lat - lat) / self.d_lat)
        col = int((lon - self.min_lon) / self.d_lon)
        return Point(
            row=max(0, min(self.rows - 1, row)),
            col=max(0, min(self.cols - 1, col))
        )

    def _index_to_coord(self, point: Point) -> Tuple[float, float]:
        lat = self.max_lat - (point.row * self.d_lat)
        lon = self.min_lon + (point.col * self.d_lon)
        return (lon, lat)

    def _haversine_distance(self, p1: Point, p2: Point) -> float:
        lon1, lat1 = self._index_to_coord(p1)
        lon2, lat2 = self._index_to_coord(p2)
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def find_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> Optional[RouteResult]:
        start = self._coord_to_index(*start_coords)
        goal = self._coord_to_index(*end_coords)

        if self.speed_raster[start.row, start.col] <= 0:
            raise ValueError("Стартовая точка находится на суше или вне зоны покрытия.")
        if self.speed_raster[goal.row, goal.col] <= 0:
            raise ValueError("Конечная точка находится на суше или вне зоны покрытия.")

        max_speed = np.max(self.speed_raster)
        pq = [(0.0, 0.0, start.row, start.col)]
        times_to_reach = {start: 0.0}
        came_from = {start: None}

        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        while pq:
            _, current_time, r, c = heapq.heappop(pq)
            current = Point(r, c)

            if current == goal:
                return self._reconstruct_and_smooth_path(came_from, current, times_to_reach[current])

            if current_time > times_to_reach.get(current, float('inf')):
                continue

            dx_km = (math.radians(self.d_lon) * EARTH_RADIUS_KM) * self.cos_lats[r]

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                
                speed = self.speed_raster[nr, nc]
                if speed <= 0:
                    continue
                
                dist = dx_km if dr == 0 else (self.dy_km if dc == 0 else math.sqrt(dx_km**2 + self.dy_km**2))
                new_time = current_time + (dist / speed)
                neighbor = Point(nr, nc)
                
                if new_time < times_to_reach.get(neighbor, float('inf')):
                    times_to_reach[neighbor] = new_time
                    came_from[neighbor] = current
                    h_time = self._haversine_distance(neighbor, goal) / max_speed
                    heapq.heappush(pq, (new_time + h_time, new_time, nr, nc))

        return None

    def _reconstruct_and_smooth_path(self, came_from: dict, current: Point, total_time: float) -> RouteResult:
        path = []
        while current is not None:
            path.append(self._index_to_coord(current))
            current = came_from[current]
        path.reverse()
        
        # Сглаживание Чайкина
        smoothed = path
        for _ in range(2):
            if len(smoothed) <= 2: break
            new_path = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p0, p1 = smoothed[i], smoothed[i+1]
                new_path.extend([(0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]),
                                 (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])])
            new_path.append(smoothed[-1])
            smoothed = new_path
            
        dist = sum(self._haversine_distance(self._coord_to_index(*smoothed[i]), 
                   self._coord_to_index(*smoothed[i+1])) for i in range(len(smoothed)-1))
        
        return RouteResult(path_coords=smoothed, total_distance_km=dist, total_time_hours=total_time)