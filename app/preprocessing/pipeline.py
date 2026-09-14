import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from typing import Optional
import os
import glob
import gc
import json

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

    def init_from_raster(self, filepath: str):
        """Быстрая инициализация метаданных (без загрузки всего массива в RAM)."""
        print(f"🔧 Инициализация сетки из: {os.path.basename(filepath)}")
        with rasterio.open(filepath) as src:
            self.meta = src.meta.copy()
            self.transform = src.transform
            self.shape = (src.height, src.width)

    def _read_and_fix_geojson(self, filepath: str) -> gpd.GeoDataFrame:
        """Считывает GeoJSON и принудительно замыкает все полигоны (защита от ошибки LinearRing)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for feat in data.get('features', []):
            geom = feat.get('geometry')
            if not geom: continue
            
            if geom['type'] == 'Polygon':
                for ring in geom['coordinates']:
                    if ring and ring[0] != ring[-1]:
                        ring.append(ring[0])
            elif geom['type'] == 'MultiPolygon':
                for poly in geom['coordinates']:
                    for ring in poly:
                        if ring and ring[0] != ring[-1]:
                            ring.append(ring[0])
                            
        return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

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

    def _write_raster(self, arr: np.ndarray, filepath: str, dtype: str = 'float32', nodata: float = 0.0):
        """Сохранение итогового растра с настраиваемым типом."""
        out_meta = self.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "dtype": dtype,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": nodata
        })
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with rasterio.open(filepath, 'w', **out_meta) as dest:
            dest.write(arr, 1)
        print(f"✅ Сохранен файл: {filepath}")

    def process_seca_mask(self, seca_geojson: str, out_seca_tif: str):
        """Создает растр маски SECA (1 - SECA, 0 - остальное). uint8 для экономии RAM."""
        if not self.shape or not self.transform:
            raise ValueError("Сначала проинициализируйте meta")
            
        print(f"🗺️ Генерация маски SECA из: {os.path.basename(seca_geojson)}")
        gdf = self._read_and_fix_geojson(seca_geojson)
        shapes = ((geom, 1) for geom in gdf.geometry)
        
        seca_mask = rasterize(
            shapes=shapes,
            out_shape=self.shape,
            transform=self.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        self._write_raster(seca_mask, out_seca_tif, dtype='uint8', nodata=0)

    def compile_chokepoints_folder(self, folder_path: str, out_geojson: str, out_config: str):
        """Автоматически склеивает множество KML/SHP файлов узких мест в единый GeoJSON и формирует конфиг."""
        import glob
        import math
        import pandas as pd
        import fiona
        
        # Включаем поддержку чтения KML в fiona/geopandas
        fiona.drvsupport.supported_drivers['KML'] = 'rw'
        fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
        
        files = glob.glob(os.path.join(folder_path, "*.kml")) + glob.glob(os.path.join(folder_path, "*.shp"))
        if not files:
            print(f"⚠️ Файлы узких мест не найдены в {folder_path}")
            return
            
        print(f"⚓ Найдено {len(files)} файлов узких мест. Начинаем парсинг и склейку...")
        
        all_gdfs = []
        config_data = {}
        current_id = 1
        
        for fpath in files:
            try:
                gdf = gpd.read_file(fpath)
                for _, row in gdf.iterrows():
                    # Пытаемся достать названия (EN или RU)
                    name = row.get("Name_EN") if "Name_EN" in row and pd.notna(row.get("Name_EN")) else row.get("Name_RU")
                    if pd.isna(name) or not name: 
                        name = os.path.basename(fpath).rsplit('.', 1)[0]
                    
                    # Достаем лимиты, если они есть
                    draft = float(row["Draught"]) if "Draught" in row and pd.notna(row["Draught"]) else None
                    width = float(row["Width"]) if "Width" in row and pd.notna(row["Width"]) else None
                    length = float(row["Length"]) if "Length" in row and pd.notna(row["Length"]) else None
                    
                    # Сохраняем в конфигурационный словарь
                    config_data[str(current_id)] = {"name": str(name)}
                    if length and not math.isnan(length): config_data[str(current_id)]["max_length"] = length
                    if width and not math.isnan(width): config_data[str(current_id)]["max_width"] = width
                    if draft and not math.isnan(draft): config_data[str(current_id)]["max_draft"] = draft
                    
                    # Оставляем только нужную геометрию и сгенерированный id
                    clean_row = gpd.GeoDataFrame({"id": [current_id], "geometry": [row.geometry]}, crs=gdf.crs)
                    all_gdfs.append(clean_row)
                    
                    current_id += 1
            except Exception as e:
                print(f"❌ Ошибка обработки файла {fpath}: {e}")
                
        if all_gdfs:
            merged_gdf = pd.concat(all_gdfs, ignore_index=True)
            # Принудительно приводим к EPSG:4326 для консистентности
            if merged_gdf.crs != "EPSG:4326":
                merged_gdf = merged_gdf.to_crs("EPSG:4326")
                
            merged_gdf.to_file(out_geojson, driver="GeoJSON")
            with open(out_config, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Успешно склеено {current_id - 1} полигонов. Сохранены {os.path.basename(out_geojson)} и {os.path.basename(out_config)}")

    def process_chokepoints(self, cp_file: str, out_cp_tif: str):
        """Создает растр узких мест (значение = id из свойств). uint16."""
        if not self.shape or not self.transform:
            raise ValueError("Сначала проинициализируйте meta")
            
        print(f"⚓ Генерация растра узких мест из: {os.path.basename(cp_file)}")
        if cp_file.lower().endswith(".shp"):
            gdf = gpd.read_file(cp_file)
        else:
            gdf = self._read_and_fix_geojson(cp_file)
        
        # GeoDataFrame разворачивает свойства в столбцы. Избегаем ошибки AttributeError
        shapes = ((row.geometry, int(row['id']) if 'id' in row else 1) for _, row in gdf.iterrows())
        
        cp_mask = rasterize(
            shapes=shapes,
            out_shape=self.shape,
            transform=self.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint16
        )
        self._write_raster(cp_mask, out_cp_tif, dtype='uint16', nodata=0)

    def process_gebco(self, gebco_dir: str, out_depth_tif: str):
        """Ресэмплинг тайлов GEBCO под сетку проекта. Сохраняем как int16."""
        if not self.shape or not self.transform:
            raise ValueError("Сначала проинициализируйте meta, считав базовый растр")

        gebco_tiles = glob.glob(os.path.join(gebco_dir, "gebco_*.tif"))
        if not gebco_tiles:
            raise FileNotFoundError(f"Тайлы GEBCO не найдены в директории: {gebco_dir}")

        print(f"🌊 Обработка батиметрии GEBCO (Найдено {len(gebco_tiles)} тайлов)")
        
        # Инициализируем массив значением NoData
        dest_arr = np.full(self.shape, -32768, dtype=np.int16)
        
        for tile_path in gebco_tiles:
            print(f"   -> Ресэмплинг тайла: {os.path.basename(tile_path)} ...")
            with rasterio.open(tile_path) as src:
                # Временный массив для текущего тайла
                temp_arr = np.full(self.shape, -32768, dtype=np.int16)
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=temp_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.transform,
                    dst_crs=self.meta['crs'],
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=-32768
                )
                
                # Накладываем обработанный тайл поверх общего массива
                valid_mask = temp_arr != -32768
                dest_arr[valid_mask] = temp_arr[valid_mask]
                
                # Очищаем память (~7 ГБ RAM на итерацию)
                del temp_arr
                del valid_mask
                gc.collect()
                
        self._write_raster(dest_arr, out_depth_tif, dtype='int16', nodata=-32768)

    def _create_land_mask_from_geojson(self, geojson_path: str) -> np.ndarray:
        """Создает маску суши (1 - суша, 0 - вода) из GeoJSON полигонов океанов."""
        print(f"🌊 Генерация маски суши из: {os.path.basename(geojson_path)}")
        gdf = self._read_and_fix_geojson(geojson_path)
        
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

    def download_waves(self, source_url: str, target_date: str, out_filepath: str):
        """
        Скачивает и подготавливает данные о волнении (погоде).
        В отличие от статичных растров, этот метод предназначен для регулярного запуска.
        """
        import urllib.request
        from urllib.error import URLError
        
        # NOAA обычно требует дату в формате YYYYMMDD
        formatted_date = target_date.replace("-", "")
        url = source_url.replace("{date}", formatted_date)
        
        print(f"🌊 Обновление динамических данных: Волнение за {target_date}...")
        print(f"   -> Источник: {url}")
        
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        
        # Притворяемся обычным браузером, чтобы обойти защиту 403 Forbidden от NOAA
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                with open(out_filepath, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"✅ Файл волнения успешно сохранен: {out_filepath}")
        except URLError as e:
            print(f"❌ Ошибка скачивания волнения: {e}")
            print("   Проверьте доступность интернета или корректность ссылки NOAA в config.json.")

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

        print("🎯 Базовый препроцессинг успешно завершен!")

if __name__ == "__main__":
    import json
    
    # 1. Читаем глобальную конфигурацию
    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.json"))
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    BASE_DIR = config["paths"]["data_dir"]
    preprocessor = RasterPreprocessor(BASE_DIR)
    
    # =====================================================================
    # УПРАВЛЕНИЕ ПАЙПЛАЙНОМ (Toggles)
    # Включайте (True) или выключайте (False) нужные шаги в зависимости от задачи.
    # =====================================================================
    RUN_STEPS = {
        # Базовая инициализация матрицы (Нужна почти всегда, кроме генерации волн)
        "init_grid": True,
        
        # [СТАТИКА] Тяжелый просчет матриц скоростей АИС (Запускать 1 раз)
        "ais_data": False,
        
        # [СТАТИКА] Растеризация экологических зон SECA (Запускать 1 раз)
        "seca_zones": False,
        
        # [СТАТИКА] Склейка тайлов глубин GEBCO (Запускать 1 раз)
        "gebco_bathymetry": False,
        
        # [СТАТИКА] Растеризация узких мест Суэц/Панама (Запускать 1 раз)
        "chokepoints": True,
        
        # [ДИНАМИКА] Скачивание погоды/волнения (Запускать регулярно по CRON)
        "waves_weather": False
    }

    print("\n=== СТАРТ ПАЙПЛАЙНА ПРЕДОБРАБОТКИ ===")
    
    if RUN_STEPS["init_grid"]:
        count_file = f"{BASE_DIR}/count.tif"
        if os.path.exists(count_file):
            preprocessor.init_from_raster(count_file)
        else:
            print("⚠️ Файл базовой сетки не найден. Инициализация пропущена.")

    if RUN_STEPS["ais_data"]:
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["paths"]["static_dir"]))
        preprocessor.process(
            count_tif=f"{BASE_DIR}/count.tif",
            vel_tif=f"{BASE_DIR}/mean_velocity_knots.tif",
            sd_tif=f"{BASE_DIR}/mean_velocity_sd_knots.tif",
            out_eff_vel=f"{config['paths']['processed_dir']}/eff_vel_all.tif",
            out_sd=f"{config['paths']['processed_dir']}/eff_sd_all.tif",
            oceans_geojson=f"{static_dir}/oceans-seas.geo.json"
        )

    if RUN_STEPS["seca_zones"]:
        preprocessor.process_seca_mask(
            seca_geojson=f"{config['paths']['raw_dir']}/seca_zones.geo.json",
            out_seca_tif=f"{config['paths']['processed_dir']}/seca_mask.tif"
        )

    if RUN_STEPS["gebco_bathymetry"]:
        preprocessor.process_gebco(
            gebco_dir=f"{BASE_DIR}/gebco",
            out_depth_tif=f"{config['paths']['processed_dir']}/depth_meters.tif"
        )

    if RUN_STEPS["chokepoints"]:
        raw_cp_folder = f"{config['paths']['raw_dir']}/chokepoints_raw"
        cp_geo_out = f"{config['paths']['raw_dir']}/chokepoints.geo.json"
        cp_config_out = f"{config['paths']['processed_dir']}/chokepoints_config.json"
        
        # Шаг 1: Автоматическая склейка KML/SHP из папки
        if os.path.exists(raw_cp_folder):
            preprocessor.compile_chokepoints_folder(raw_cp_folder, cp_geo_out, cp_config_out)
        
        # Шаг 2: Растеризация единого файла
        if os.path.exists(cp_geo_out):
            preprocessor.process_chokepoints(
                cp_file=cp_geo_out,
                out_cp_tif=f"{config['paths']['processed_dir']}/chokepoints.tif"
            )
        else:
            print(f"⏭️ Пропуск chokepoints: файл {cp_geo_out} не найден.")

    if RUN_STEPS["waves_weather"]:
        w_conf = config["waves"]
        wave_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", w_conf["output_file"]))
        preprocessor.download_waves(
            source_url=w_conf["source_url"],
            target_date=w_conf["target_date"],
            out_filepath=wave_out
        )

    print("=== ПАЙПЛАЙН ЗАВЕРШЕН ===\n")