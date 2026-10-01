import os
import glob
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import urllib.request
from datetime import datetime, timedelta

class MeteoRasterBase(ABC):
    """
    Абстрактный класс для поставщиков гидрометеорологических данных.
    Гарантирует, что любой источник (NOAA, Морской портал и др.) 
    будет отдавать данные в едином формате для нашего Router Engine.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def fetch_forecast(self, target_date: datetime, hours_ahead: int = 72) -> List[str]:
        """
        Скачивает прогноз на X часов вперед.
        Возвращает список путей к локальным растрам (таймлайн).
        """
        pass

    @abstractmethod
    def get_layer_metadata(self) -> Dict[str, str]:
        """Возвращает информацию о слоях (волны, ветер, лед)."""
        pass

class NOAAMeteoGateway(MeteoRasterBase):
    """
    Текущая реализация (пока нет интеграции с Морским порталом).
    Стягивает GFS Wave данные.
    """
    
    def __init__(self, output_dir: str, base_url: str):
        super().__init__(output_dir)
        self.base_url = base_url

    def fetch_forecast(self, target_date: datetime, hours_ahead: int = 72) -> List[str]:
        saved_files = []
        date_str = target_date.strftime("%Y%m%d")
        
        # NOAA отдает прогнозы с шагом в 3 часа
        steps = range(0, hours_ahead + 1, 3)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': '*/*'
        }
        
        for step in steps:
            step_str = f"{step:03d}"
            url = self.base_url.format(date=date_str, step=step_str)
            out_file = os.path.join(self.output_dir, f"waves_{date_str}_f{step_str}.grib2")
            
            if not os.path.exists(out_file):
                print(f"🌊 Скачивание прогноза на +{step_str}ч: {url}")
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as response:
                        with open(out_file, 'wb') as f:
                            f.write(response.read())
                except Exception as e:
                    print(f"❌ Ошибка скачивания {url}: {e}")
                    continue
                    
            saved_files.append(out_file)
            
        # TODO: Запуск ресэмплинга (rasterio.warp) grib2 файлов в нашу сетку eff_vel_all.tif
        return saved_files

    def get_layer_metadata(self) -> Dict[str, str]:
        return {
            "waves": "Significant wave height (meters)",
            "wind": "Wind speed (knots)"
        }

class PortalMeteoGateway(MeteoRasterBase):
    """
    Заглушка для будущей интеграции с "Морским порталом" клиента.
    """
    def __init__(self, output_dir: str, api_token: str, portal_url: str):
        super().__init__(output_dir)
        self.api_token = api_token
        self.portal_url = portal_url

    def fetch_forecast(self, target_date: datetime, hours_ahead: int = 72) -> List[str]:
        # В будущем здесь будет логика обращения к внутреннему API клиента
        # и получение уже готовых слоев льда и погоды.
        print("Интеграция с Морским порталом в разработке...")
        return []

    def get_layer_metadata(self) -> Dict[str, str]:
        return {
            "waves": "Significant wave height (meters)",
            "ice": "Ice concentration (%)",
            "wind": "Wind speed (knots)"
        }