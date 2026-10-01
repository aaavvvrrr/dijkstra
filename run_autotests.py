import os
import glob
import json
import time
import requests

API_URL = "http://127.0.0.1:8000/api/route"
TOLERANCE_PCT = 0.05  # Допустимое отклонение 5%

def run_tests():
    tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tests", "cases"))
    test_files = glob.glob(os.path.join(tests_dir, "*.json"))
    
    if not test_files:
        print(f"⚠️ Тесты не найдены в папке {tests_dir}. Сохраните их через веб-интерфейс.")
        return

    print(f"🚀 Запуск автотестов (найдено {len(test_files)} файлов)...\n")
    
    passed = 0
    failed = 0

    for filepath in test_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            test_data = json.load(f)
            
        test_name = test_data.get("name", filename)
        payload = test_data.get("payload", {})
        expected_geojson = test_data.get("expected_geojson", {})
        
        print(f"⏳ Тест: {test_name} ...", end="", flush=True)
        
        t0 = time.time()
        try:
            # Увеличиваем сетевой таймаут HTTP-клиента (берем из payload + 60 сек запаса, или 1 час по умолчанию)
            client_timeout = payload.get("timeout_seconds", 3600) + 60
            response = requests.post(API_URL, json=payload, timeout=client_timeout)
        except requests.exceptions.RequestException as e:
            print(f" [❌ ОШИБКА СЕТИ] Сервер недоступен: {e}")
            failed += 1
            continue
            
        elapsed = time.time() - t0

        if response.status_code != 200:
            print(f" [❌ ПРОВАЛ] Ошибка API: {response.status_code} - {response.text}")
            failed += 1
            continue
            
        actual_geojson = response.json()
        
        # Нечеткое сравнение свойств
        act_props = actual_geojson.get("properties", {})
        exp_props = expected_geojson.get("properties", {})
        
        act_dist = act_props.get("distance_total_km", 0)
        exp_dist = exp_props.get("distance_total_km", 0)
        dist_diff = abs(act_dist - exp_dist) / max(exp_dist, 1)
        
        act_time = act_props.get("time_total_hours", 0)
        exp_time = exp_props.get("time_total_hours", 0)
        time_diff = abs(act_time - exp_time) / max(exp_time, 1)

        # Вывод результатов
        if dist_diff <= TOLERANCE_PCT and time_diff <= TOLERANCE_PCT:
            print(f" [✅ ПРОЙДЕН за {elapsed:.1f}c] Дистанция: {act_dist:.0f}км (Δ={dist_diff:.1%}), Время: {act_time:.1f}ч (Δ={time_diff:.1%})")
            passed += 1
        else:
            print(f" [❌ ПРОВАЛ за {elapsed:.1f}c]")
            print(f"    -> Ожидалось: Дистанция ~{exp_dist:.0f}км, Время ~{exp_time:.1f}ч")
            print(f"    -> Получено:  Дистанция ~{act_dist:.0f}км, Время ~{act_time:.1f}ч")
            failed += 1

    print("\n" + "="*40)
    print(f"📊 ИТОГО: Успешно {passed}, Провалено {failed} (Всего {passed+failed})")
    print("="*40)
    
    if failed > 0:
        exit(1)

if __name__ == "__main__":
    run_tests()