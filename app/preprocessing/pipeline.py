import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from typing import Optional
import os

class RasterPreprocessor:
    """
    Пайплайн подготовки данных АИС.
    Заменяет шаги 1-5 из legacy .bat скриптов.
    Формирует итоговый растр эффективной скорости и сглаженный растр SD для Монте-Карло.
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.meta = None
        self.transform = None
        self.shape = None

    def _read_raster(self, filepath: str) -> np.ndarray:
        """Чтение растра в формате float32."""
        with rasterio.open(filepath) as src:
            if self.meta is None:
                self.meta = src.meta.copy()
                self.transform = src.transform
                self.shape = (src.height, src.width)
            
            arr = src.read(1).astype(np.float32)
            # Обработка NoData
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = 0.0
            return arr

    def _write_raster(self, arr: np.ndarray, filepath: str):
        """Сохранение итогового растра."""
        out_meta = self.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "dtype": 'float32',
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": 0.0
        })
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with rasterio.open(filepath, 'w', **out_meta) as dest:
            dest.write(arr, 1)
        print(f"✅ Сохранен файл: {filepath}")

    def _create_land_mask_from_geojson(self, geojson_path: str) -> np.ndarray:
        """Создает маску суши (1 - суша, 0 - вода) из GeoJSON полигонов океанов."""
        print(f"🌊 Генерация маски суши из: {os.path.basename(geojson_path)}")
        gdf = gpd.read_file(geojson_path)
        
        # Полигоны в файле - это ОКЕАНЫ (вода). 
        # Значит, мы заливаем весь растр единицами (суша), 
        # а там где есть геометрия океана - прожигаем нулями (вода).
        shapes = ((geom, 0.0) for geom in gdf.geometry)
        
        land_mask = rasterize(
            shapes=shapes,
            out_shape=self.shape,
            transform=self.transform,
            fill=1.0,           # Фон по умолчанию = Суша
            all_touched=True,
            dtype=np.float32
        )
        return land_mask

    def process(
        self, count_tif: str, vel_tif: str, sd_tif: str, 
        out_eff_vel: str, out_sd: str, oceans_geojson: Optional[str] = None
    ):
        import gc
        print("🚀 Старт memory-optimized препроцессинга...")

        # --- ФАЗА 1: Маски и счетчики ---
        print("1. Загрузка count_tif...")
        count = self._read_raster(count_tif) # ~9.5 ГБ
        has_tracks = count > 0               # ~2.4 ГБ (boolean)

        print("2. Формирование гибридной маски суши...")
        if oceans_geojson and os.path.exists(oceans_geojson):
            land_mask = self._create_land_mask_from_geojson(oceans_geojson) # ~9.5 ГБ
            # In-place замена: где есть треки, делаем воду (0.0)
            np.copyto(land_mask, 0.0, where=has_tracks)
            print("🗺️ Применен гибридный подход.")
        else:
            land_mask = (~has_tracks).astype(np.float32)

        print("3. Сглаживание count (in-place)...")
        count_flt = gaussian_filter(count, sigma=1.2, mode='reflect')
        np.maximum(count, count_flt, out=count) # count = max(count, count_flt)
        del count_flt
        gc.collect()

        # Обнуляем счетчик на суше (count теперь является count_final)
        count[land_mask == 1.0] = 0.0

        print("4. Расчет сигмоиды (in-place)...")
        # Чтобы не тратить еще 9.5 ГБ, превращаем массив count в множитель сигмоиды
        count -= 20.0
        count *= -0.1962959
        np.clip(count, -50, 50, out=count)
        np.exp(count, out=count)
        count += 1.0
        np.reciprocal(count, out=count) # Теперь count = 1 / (1 + exp(...))

        # --- ФАЗА 2: Обработка Дисперсии (SD) ---
        print("5. Загрузка vel и sd для обработки дисперсии...")
        vel = self._read_raster(vel_tif)
        # Поднимаем минимальную скорость
        vel[has_tracks] = np.maximum(vel[has_tracks], 4.5)

        sd = self._read_raster(sd_tif)
        sd[has_tracks] = np.maximum(sd[has_tracks], 0.1 * vel[has_tracks])
        
        # Маска has_tracks больше не нужна, освобождаем 2.4 ГБ
        del has_tracks
        gc.collect()

        print("6. Сглаживание SD (in-place)...")
        sd_flt = gaussian_filter(sd, sigma=1.2, mode='reflect')
        np.maximum(sd, sd_flt, out=sd)
        del sd_flt
        gc.collect()

        # Обнуляем на суше, сохраняем, удаляем SD
        sd[land_mask == 1.0] = 0.0
        self._write_raster(sd, out_sd)
        del sd
        gc.collect()

        # --- ФАЗА 3: Обработка Скорости (Vel) ---
        print("7. Сглаживание скорости (in-place)...")
        vel_flt = gaussian_filter(vel, sigma=1.2, mode='reflect')
        np.maximum(vel, vel_flt, out=vel)
        del vel_flt
        gc.collect()

        print("8. Применение сигмоиды к скорости...")
        vel *= count  # Умножаем скорость на готовую сигмоиду
        del count
        gc.collect()

        # Обнуляем сушу, сохраняем, удаляем остатки
        vel[land_mask == 1.0] = 0.0
        self._write_raster(vel, out_eff_vel)
        del vel
        del land_mask
        gc.collect()

        print("🎯 Препроцессинг успешно завершен!")

if __name__ == "__main__":
    BASE_DIR = "./data"
    
    preprocessor = RasterPreprocessor(BASE_DIR)
    
    # Теперь мы передаем наш oceans-seas.geo.json
    preprocessor.process(
        count_tif=f"{BASE_DIR}/count.tif",
        vel_tif=f"{BASE_DIR}/mean_velocity_knots.tif",
        sd_tif=f"{BASE_DIR}/mean_velocity_sd_knots.tif",
        out_eff_vel=f"{BASE_DIR}/processed/effective_velocity_knots.tif",
        out_sd=f"{BASE_DIR}/processed/effective_sd_knots.tif",
        oceans_geojson=f"./app/static/oceans-seas.geo.json"
    )