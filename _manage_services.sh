#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

export HOST_NODE_NAME=$(hostname)

# Список всех доступных сервисов
ALL_SERVICES=("maritime-router")

# --- Функции проверки и помощи ---
show_help() {
    echo -e "${CYAN}Использование:${NC} $0 <команда> [сервисы...]"
    echo -e "${CYAN}Команды:${NC} start, stop, restart, down, logs, exec, status, monitor, images, menu, help"
    echo -e "${CYAN}Примеры:${NC} $0 start web db | $0 logs web | $0 monitor rec | $0 images"
    echo -e "${CYAN}Сервисы:${NC} ${ALL_SERVICES[*]}"
}

check_docker() {
    if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
        echo -e "${RED}Ошибка: Docker или Docker Compose не установлены.${NC}"
        exit 1
    fi
}

get_user_selection() {
    local prompt_msg=$1
    echo -e "\n${WHITE}${prompt_msg}${NC}"
    echo -e "${YELLOW}Подсказка: 'a' - ВСЕ сервисы, '1 3' - выбор по номерам${NC}\n"
    
    for i in "${!ALL_SERVICES[@]}"; do
        printf "  ${GREEN}%2d${NC}) %s\n" "$((i+1))" "${ALL_SERVICES[$i]}"
    done
    
    echo ""
    read -p "Ваш выбор > " selected_indices
    SELECTED_SERVICES=()

    if [[ "$selected_indices" == "a" || "$selected_indices" == "A" ]]; then
        SELECTED_SERVICES=("${ALL_SERVICES[@]}")
        echo -e "${GREEN}>> Выбраны ВСЕ сервисы (${#SELECTED_SERVICES[@]})${NC}"
        return 0
    fi

    for index in $selected_indices; do
        if [[ "$index" =~ ^[0-9]+$ ]] && [ "$index" -ge 1 ] && [ "$index" -le "${#ALL_SERVICES[@]}" ]; then
            SELECTED_SERVICES+=("${ALL_SERVICES[$((index-1))]}")
        else
            echo -e "${RED}!! Пропущен неверный номер: $index${NC}"
        fi
    done

    if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
        echo -e "${RED}Ни один сервис не выбран.${NC}"
        return 1
    fi
    echo -e "${GREEN}>> Принято: ${SELECTED_SERVICES[*]}${NC}"
    return 0
}

# --- Основные действия ---
action_start() {
    echo -e "\n${GREEN}🚀 Запуск: ${*}${NC}"
    sudo HOST_NODE_NAME="$HOST_NODE_NAME"  docker compose up -d "$@"
    [ $? -eq 0 ] && echo -e "${GREEN}✅ Успешно запущено.${NC}" || echo -e "${RED}❌ Ошибка запуска.${NC}"
}

action_stop() {
    echo -e "\n${YELLOW}🛑 Остановка: ${*}${NC}"
    sudo HOST_NODE_NAME="$HOST_NODE_NAME"  docker compose stop "$@"
    [ $? -eq 0 ] && echo -e "${GREEN}✅ Успешно остановлено.${NC}" || echo -e "${RED}❌ Ошибка остановки.${NC}"
}

action_restart() {
    echo -e "\n${CYAN}🔄 Перезапуск: ${*}${NC}"
    sudo HOST_NODE_NAME="$HOST_NODE_NAME"  docker compose restart "$@"
    [ $? -eq 0 ] && echo -e "${GREEN}✅ Успешно перезапущено.${NC}" || echo -e "${RED}❌ Ошибка перезапуска.${NC}"
}

action_down() {
    echo -e "\n${RED}⚠️  Удаление контейнеров: ${*}${NC}"
    read -p "Вы уверены? (y/n): " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        sudo HOST_NODE_NAME="$HOST_NODE_NAME"  docker compose down "$@"
        [ $? -eq 0 ] && echo -e "${GREEN}✅ Контейнеры удалены.${NC}" || echo -e "${RED}❌ Ошибка удаления.${NC}"
    else
        echo "Отменено."
    fi
}

action_status() {
    echo -e "\n${CYAN}📊 Статус сервисов:${NC}"
    sudo HOST_NODE_NAME="$HOST_NODE_NAME"  docker compose ps
}

action_logs() {
    local service=$1
    echo -e "\n${CYAN}📜 Логи сервиса '${service}' (Ctrl+C для выхода)${NC}"
    trap '' INT
    sudo HOST_NODE_NAME="$HOST_NODE_NAME"  docker compose logs -f --tail=100 "$service"
    trap - INT
    echo -e "\n${YELLOW}Просмотр логов завершен.${NC}"
}

action_exec() {
    local service=$1
    echo -e "\n${CYAN}💻 Подключение к '${service}' (Ctrl+D или exit для выхода)${NC}"
    if ! sudo docker exec -it "$service" /bin/bash; then
        sudo docker exec -it "$service" /bin/sh || echo -e "${RED}❌ Не удалось подключиться.${NC}"
    fi
}

# --- НОВАЯ ФУНКЦИЯ: Мониторинг TMPFS ---
get_tmpfs_stats() {
    local service=$1
    local container_name=$service # Обычно имя контейнера совпадает с сервисом, если не задано иное
    
    # Проверяем, запущен ли контейнер
    if [ "$(sudo docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" != "true" ]; then
        echo -e "${RED}⚠️  Контейнер '$service' не запущен.${NC}"
        return 1
    fi

    echo -e "${MAGENTA}----------------------------------------${NC}"
    echo -e "${CYAN}📂 Статистика TMPFS в '${service}':${NC}"
    echo -e "${MAGENTA}----------------------------------------${NC}"

    # 1. Проверка temp_video_record
    echo -e "${WHITE}📁 Папка: /app/temp_video_record${NC}"
    # Получаем размер в человекочитаемом виде (например, 1.2G)
    local size_rec=$(sudo docker exec "$container_name" du -sh /app/temp_video_record 2>/dev/null | awk '{print $1}')
    # Считаем количество файлов (не директорий)
    local count_rec=$(sudo docker exec "$container_name" find /app/temp_video_record -type f 2>/dev/null | wc -l)
    
    if [ -z "$size_rec" ]; then
        echo -e "   ${RED}Недоступно (папка пуста или ошибка прав)${NC}"
    else
        echo -e "   ${GREEN}Размер:${NC} $size_rec"
        echo -e "   ${GREEN}Файлов:${NC} $count_rec"
    fi

    echo ""

    # 2. Проверка video_cache
    echo -e "${WHITE}📁 Папка: /app/video_cache${NC}"
    local size_cache=$(sudo docker exec "$container_name" du -sh /app/video_cache 2>/dev/null | awk '{print $1}')
    local count_cache=$(sudo docker exec "$container_name" find /app/video_cache -type f 2>/dev/null | wc -l)

    if [ -z "$size_cache" ]; then
        echo -e "   ${RED}Недоступно (папка пуста или ошибка прав)${NC}"
    else
        echo -e "   ${GREEN}Размер:${NC} $size_cache"
        echo -e "   ${GREEN}Файлов:${NC} $count_cache"
    fi
    
    echo -e "${MAGENTA}----------------------------------------${NC}"
}

action_monitor() {
    local service=$1
    if [ -z "$service" ]; then
        # Если сервис не указан, спрашиваем
        get_user_selection "Для какого сервиса смотреть占用 TMPFS?"
        if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then return 1; fi
        service=${SELECTED_SERVICES[0]}
    fi
    
    get_tmpfs_stats "$service"
}

# --- Список Docker-образов ---
action_images() {
    echo -e "\n${CYAN}🖼️  Docker образы:${NC}"

    if sudo docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}"; then
        echo -e "${GREEN}✅ Список образов получен.${NC}"
    else
        echo -e "${RED}❌ Ошибка получения списка Docker-образов.${NC}"
    fi
}

# --- Подменю УПРАВЛЕНИЯ ---
menu_manage() {
    while true; do
        echo -e "\n${RED}========================================${NC}"
        echo -e "${RED}   ⚙️  УПРАВЛЕНИЕ (ОСТАНОВКА / ПЕРЕЗАПУСК)${NC}"
        echo -e "${RED}========================================${NC}"
        echo "1) Остановить (stop)"
        echo "2) Перезапустить (restart)"
        echo "3) Удалить контейнеры (down)"
        echo "0) ⬅️  Назад"
        echo -e "${RED}----------------------------------------${NC}"
        
        read -p "Выберите действие: " choice
        case $choice in
            1) get_user_selection "Какие сервисы ОСТАНОВИТЬ?" && action_stop "${SELECTED_SERVICES[@]}" ;;
            2) get_user_selection "Какие сервисы ПЕРЕЗАПУСТИТЬ?" && action_restart "${SELECTED_SERVICES[@]}" ;;
            3) get_user_selection "Какие сервисы УДАЛИТЬ (down)?" && action_down "${SELECTED_SERVICES[@]}" ;;
            0) return ;;
            *) echo -e "${RED}Неверный выбор.${NC}" ;;
        esac
    done
}

# --- Главное меню ---
# --- Главное меню ---
show_main_menu() {
    check_docker
    while true; do
        echo -e "\n${CYAN}========================================${NC}"
        echo -e "${CYAN}   🐳 DOCKER COMPOSE MANAGER${NC}"
        echo -e "${CYAN}========================================${NC}"
        echo "1) 🟢 Запустить сервисы (up)"
        echo "2) ⚙️  Управление (stop, restart, down)"
        echo "3) 📊 Статус (ps)"
        echo "4) 📜 Просмотреть логи (logs)"
        echo "5) 💻 Войти в контейнер (exec)"
        echo "6) 📈 Мониторинг TMPFS (RAM)"
        echo "7) 🖼️  Docker образы (images)"
        echo "0) 🚪 Выход"
        echo -e "${CYAN}----------------------------------------${NC}"
        
        read -p "Выберите раздел [0-7]: " choice
        case $choice in
            1) get_user_selection "Какие сервисы ЗАПУСТИТЬ?" && action_start "${SELECTED_SERVICES[@]}" ;;
            2) menu_manage ;;
            3) action_status ;;
            4) get_user_selection "Для какого сервиса смотреть логи?" && action_logs "${SELECTED_SERVICES[0]}" ;;
            5) get_user_selection "В какой контейнер войти?" && action_exec "${SELECTED_SERVICES[0]}" ;;
            6) get_user_selection "Какой сервис мониторить (TMPFS)?" && action_monitor "${SELECTED_SERVICES[0]}" ;;
            7) action_images ;;
            0) echo "Выход."; exit 0 ;;
            *) echo -e "${RED}Неверный выбор.${NC}" ;;
        esac
    done
}

# Обработка аргументов CLI
if [ $# -gt 0 ]; then
    command=$1
    shift
    target_service=${1:-""} 
    
    # Для команд, требующих сервис, но не получивших его, можно обработать отдельно, 
    # но пока оставим как есть, передавая массив services ниже
    
    if [[ ("$command" == "logs" || "$command" == "exec" || "$command" == "monitor") && -z "$target_service" ]]; then
         # Если сервис не передан явно, попробуем взять из остальных аргументов или выведем ошибку
         if [ $# -eq 0 ]; then
             echo -e "${RED}Ошибка: Для команды '$command' необходимо указать имя сервиса.${NC}"
             exit 1
         fi
    fi

    services=("$@")
    # Если массив пуст (был только один аргумент команды), то services будет содержать пустую строку или быть пустым
    if [ ${#services[@]} -eq 0 ] || [ -z "${services[0]}" ]; then
        services=("${ALL_SERVICES[@]}")
    fi

    case $command in
        start|up) check_docker; action_start "${services[@]}" ;;
        stop) check_docker; action_stop "${services[@]}" ;;
        restart) check_docker; action_restart "${services[@]}" ;;
        down) check_docker; action_down "${services[@]}" ;;
        logs) check_docker; action_logs "$target_service" ;;
        exec) check_docker; action_exec "$target_service" ;;
        monitor) check_docker; action_monitor "$target_service" ;;
        status|ps) check_docker; action_status ;;
        images|image) check_docker; action_images ;;
        menu) show_main_menu ;;
        help|-h) show_help; exit 0 ;;
        *) echo -e "${RED}Неизвестная команда: $command${NC}"; show_help; exit 1 ;;
    esac
else
    show_main_menu
fi