#!/usr/bin/env python3
"""
Скрипт для создания единого markdown/pdf файла с кодовой базой для LLM промпта.
Рекурсивно сканирует директорию и объединяет файлы указанных типов.
Результат сохраняется в папку со скриптом.
"""

import sys
from pathlib import Path
from typing import Set, List, Optional

# Настройки по умолчанию
DEFAULT_EXTENSIONS = {'.py', '.js', '.css', '.html', '.json','','.yml','.yaml','.txt'}
DEFAULT_IGNORE_FILES = ['fix_locales.py','login.html','ar.json','de.json','ru.json','build_embeddings_index.py','multiview.js',
                        'build_embeddings_index.py','analyse_dataset.py','fix_locales.py','ar.json','de.json','.env','error_report.html','make_tumbnails.py','rare_skus_report.html'
                        ]

DEFAULT_IGNORE_DIRS = {
    '.git', '__pycache__', 'node_modules', 'venv', '.venv',
    'env', '.env', 'dist', 'build', '.idea', '.vscode', 'make_prompt', 
    'doc','backups','datasets','json','minio-data','models','runs','temp_video','tests',
    'utils','explore','manual_frames','reports',
    'vendor','dashboard','files','minio-storage',
    'manuals',
    'census',
    # 'service',
    'workspace',
    'locales',
}
MAX_FILE_SIZE = 200 * 1024  # 200KB

# Соответствие расширений языкам для подсветки синтаксиса
EXTENSION_TO_LANG = {
    '.py': 'python',
    '.js': 'javascript',
    '.md': 'markdown',
    '.css': 'css',
    '.html': 'html',
    '.json': 'json',
}


def find_files(root_dir: Path, extensions: Set[str], ignore_dirs: Set[str]) -> List[Path]:
    """
    Рекурсивно находит файлы с указанными расширениями.
    
    Args:
        root_dir: Корневая директория для поиска
        extensions: Множество расширений файлов
        ignore_dirs: Множество игнорируемых директорий
    
    Returns:
        Отсортированный список найденных файлов
    """
    found_files = []
    
    for item in root_dir.rglob('*'):
        if not item.is_file():
            continue
            
        if item.suffix.lower() not in extensions:
            continue
        
        # Проверяем, не находится ли файл в игнорируемой директории
        try:
            if item.name in DEFAULT_IGNORE_FILES:
                continue
            rel_path = item.relative_to(root_dir)
            if any(part in ignore_dirs for part in rel_path.parts[:-1]):
                continue
        except ValueError:
            continue
        
        found_files.append(item)
    
    return sorted(found_files)


def read_file_safely(file_path: Path, max_size: int) -> Optional[str]:
    """
    Безопасно читает файл с обработкой ошибок и проверкой размера.
    
    Args:
        file_path: Путь к файлу
        max_size: Максимальный размер файла в байтах
    
    Returns:
        Содержимое файла или сообщение об ошибке
    """
    try:
        file_size = file_path.stat().st_size
        if file_size > max_size:
            return f"⚠️ Файл пропущен: размер {file_size:,} байт превышает лимит {max_size:,} байт"
        
        # Пробуем разные кодировки
        for encoding in ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        return "⚠️ Не удалось прочитать файл: ошибка кодировки"
    except PermissionError:
        return "⚠️ Нет доступа к файлу: permission denied"
    except Exception as e:
        return f"⚠️ Ошибка чтения файла: {e}"


def create_prompt_file(
    root_dir: Path,
    output_file: Path,
    extensions: Set[str] = None,
    ignore_dirs: Set[str] = None,
    max_size: int = MAX_FILE_SIZE
) -> None:
    """
    Создает единый markdown/pdf файл с содержимым всех найденных файлов.

    Args:
        root_dir: Корневая директория проекта
        output_file: Путь к выходному файлу
        extensions: Расширения файлов для включения
        ignore_dirs: Директории для игнорирования
        max_size: Максимальный размер файла в байтах
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    if ignore_dirs is None:
        ignore_dirs = DEFAULT_IGNORE_DIRS

    print(f"🔍 Сканирую директорию: {root_dir}")
    print(f"🎯 Ищу файлы с расширениями: {', '.join(sorted(extensions))}")
    print(f"🚫 Игнорирую папки: {', '.join(sorted(ignore_dirs))}")
    print(f"📏 Лимит размера файла: {max_size:,} байт")

    files_to_process = find_files(root_dir, extensions, ignore_dirs)

    if not files_to_process:
        print("\n❌ Файлы не найдены.")
        return

    # Определяем формат по расширению файла
    is_pdf = output_file.suffix.lower() == '.pdf'
    
    if is_pdf:
        create_pdf_output(root_dir, output_file, files_to_process, max_size)
    else:
        create_markdown_output(root_dir, output_file, files_to_process, max_size)


def create_markdown_output(
    root_dir: Path,
    output_file: Path,
    files_to_process: List[Path],
    max_size: int
) -> None:
    """Создает markdown файл с содержимым файлов."""
    print(f"\n✅ Найдено файлов: {len(files_to_process)}")
    print(f"💾 Создаю промпт-файл: {output_file}")

    success_count = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        # Заголовок
        f.write("# Промпт с кодовой базой\n\n")
        f.write(f"**Сгенерировано:** {Path().cwd().name}\n\n")
        f.write(f"**Источник:** `{root_dir}`\n\n")
        f.write(f"**Файлов найдено:** {len(files_to_process)}\n\n")

        # Оглавление
        f.write("## Оглавление\n\n")
        for file_path in files_to_process:
            rel_path = file_path.relative_to(root_dir)
            anchor = str(rel_path).replace(' ', '-').replace('.', '')
            f.write(f"- [{rel_path}](#{anchor})\n")
        f.write("\n---\n\n")

        # Содержимое файлов
        for idx, file_path in enumerate(files_to_process, 1):
            rel_path = file_path.relative_to(root_dir)

            # Заголовок файла
            anchor = str(rel_path).replace(' ', '-').replace('.', '')
            f.write(f"<a id='{anchor}'></a>\n\n")
            f.write(f"* {rel_path}\n\n")

            # Содержимое
            content = read_file_safely(file_path, max_size)
            lang = EXTENSION_TO_LANG.get(file_path.suffix.lower(), '')

            f.write(f"```{lang}\n")
            f.write(content)
            f.write("\n```\n\n")
            f.write("---\n\n")
            # Статус в консоль
            if content.startswith('⚠️'):
                error_count += 1
                print(f"❌ [{idx}/{len(files_to_process)}] {rel_path} - ОШИБКА")
            else:
                success_count += 1
                print(f"✅ [{idx}/{len(files_to_process)}] {rel_path}")
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    import sys
    import base64
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    sys.stdout.write(f"\033]52;c;{encoded_content}\007")
    sys.stdout.flush()


    print(f"\n🎉 Готово!")
    print(f"   ✅ Успешно обработано: {success_count}")
    print(f"   ❌ Ошибок: {error_count}")
    print(f"   📄 Результат: {output_file}")

    # Показываем примерный размер
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   📊 Размер файла: {size_mb:.2f} MB")


def create_pdf_output(
    root_dir: Path,
    output_file: Path,
    files_to_process: List[Path],
    max_size: int
) -> None:
    """Создает PDF файл с содержимым файлов. """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
        from reportlab.lib.enums import TA_LEFT
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        print("\n❌ Для создания PDF требуется библиотека reportlab.")
        print("   Установите: pip install reportlab")
        sys.exit(1)

    print(f"\n✅ Найдено файлов: {len(files_to_process)}")
    print(f"📄 Создаю PDF файл: {output_file}")

    success_count = 0
    error_count = 0

    # Создаем документ
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    # Регистрируем шрифт с поддержкой кириллицы
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'))
        font_name = 'DejaVu'
    except Exception:
        try:
            pdfmetrics.registerFont(TTFont('DejaVu', 'C:/Windows/Fonts/DejaVuSansMono.ttf'))
            font_name = 'DejaVu'
        except Exception:
            font_name = 'Helvetica'  # Fallback, но кириллица не будет работать

    # Стили
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='PromptHeading1',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='PromptHeading2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name='PromptNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=12
    ))
    styles.add(ParagraphStyle(
        name='PromptCode',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8,
        leading=10,
        textColor=colors.darkblue
    ))
    styles.add(ParagraphStyle(
        name='PromptFilePath',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        textColor=colors.darkgreen,
        spaceBefore=6,
        spaceAfter=6
    ))

    content_elements = []

    # Заголовок
    content_elements.append(Paragraph("Промпт с кодовой базой", styles['PromptHeading1']))
    content_elements.append(Paragraph(f"<b>Сгенерировано:</b> {Path().cwd().name}", styles['PromptNormal']))
    content_elements.append(Paragraph(f"<b>Источник:</b> {root_dir}", styles['PromptNormal']))
    content_elements.append(Paragraph(f"<b>Файлов найдено:</b> {len(files_to_process)}", styles['PromptNormal']))
    content_elements.append(Spacer(1, 0.2*inch))

    # Оглавление
    content_elements.append(Paragraph("Оглавление", styles['PromptHeading2']))
    for idx, file_path in enumerate(files_to_process, 1):
        rel_path = file_path.relative_to(root_dir)
        content_elements.append(Paragraph(f"{idx}. {rel_path}", styles['PromptNormal']))
    content_elements.append(PageBreak())

    # Содержимое файлов
    for idx, file_path in enumerate(files_to_process, 1):
        rel_path = file_path.relative_to(root_dir)

        # Заголовок файла
        content_elements.append(Paragraph(f"#{idx} {rel_path}", styles['PromptFilePath']))

        # Содержимое
        file_content = read_file_safely(file_path, max_size)
        
        if file_content.startswith('⚠️'):
            error_count += 1
            content_elements.append(Paragraph(file_content, styles['PromptCode']))
            print(f"❌ [{idx}/{len(files_to_process)}] {rel_path} - ОШИБКА")
        else:
            success_count += 1
            # Экранируем специальные символы для PDF
            escaped_content = file_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            content_elements.append(Preformatted(escaped_content, styles['PromptCode']))
            print(f"✅ [{idx}/{len(files_to_process)}] {rel_path}")

        # Добавляем разрыв страницы каждые 10 файлов
        if idx % 10 == 0:
            content_elements.append(PageBreak())

    # Build PDF
    doc.build(content_elements)

    print(f"\n🎉 Готово!")
    print(f"   ✅ Успешно обработано: {success_count}")
    print(f"   ❌ Ошибок: {error_count}")
    print(f"   📄 Результат: {output_file}")

    # Показываем примерный размер
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   📊 Размер файла: {size_mb:.2f} MB")


def parse_arguments():
    """Парсинг аргументов командной строки."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Создает единый markdown/pdf файл с кодовой базой для LLM промпта',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Примеры использования:
  # Стандартный запуск в текущей директории (создает prompt.md в папке со скриптом)
  python {sys.argv[0]}

  # Создать PDF файл вместо markdown
  python {sys.argv[0]} -o prompt.pdf

  # Указать конкретную директорию для сканирования
  python {sys.argv[0]} -d /path/to/project

  # Указать выходной файл (сохранится в папку со скриптом)
  python {sys.argv[0]} -o my_prompt.md

  # Добавить дополнительные расширения
  python {sys.argv[0]} -e .txt .yaml .sh

  # Исключить дополнительные папки
  python {sys.argv[0]} -i logs temp cache

  # Увеличить лимит размера файла
  python {sys.argv[0]} --max-size 500000
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Корневая директория для сканирования (по умолчанию: текущая)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='prompt.md',
        help='Имя выходного файла (по умолчанию: prompt.md). Файл сохраняется в папку со скриптом.'
    )
    
    parser.add_argument(
        '-e', '--extensions',
        nargs='*',
        type=str,
        default=[],
        help='Дополнительные расширения файлов для включения (например: .txt .yaml)'
    )
    
    parser.add_argument(
        '-i', '--ignore',
        nargs='*',
        type=str,
        default=[],
        help='Дополнительные папки для игнорирования (например: logs temp)'
    )
    
    parser.add_argument(
        '--max-size',
        type=int,
        default=MAX_FILE_SIZE,
        help=f'Максимальный размер файла в байтах (по умолчанию: {MAX_FILE_SIZE:,})'
    )
    
    return parser.parse_args()


def main():
    """Точка входа в приложение."""
    args = parse_arguments()
    
    # Определяем директорию, в которой находится сам скрипт
    script_dir = Path(__file__).resolve().parent
    
    # Подготовка путей
    root_dir = Path(args.directory).resolve()
    
    # Формируем полный путь к выходному файлу
    # Если указан относительный путь, склеиваем его с папкой скрипта.
    # Если указан абсолютный путь, оставляем как есть.
    output_arg = Path(args.output)
    if output_arg.is_absolute():
        output_file = output_arg
    else:
        output_file = script_dir / output_arg
    
    # Проверки
    if not root_dir.exists():
        print(f"❌ Директория не существует: {root_dir}")
        sys.exit(1)
    
    if not root_dir.is_dir():
        print(f"❌ Указанный путь не является директорией: {root_dir}")
        sys.exit(1)
    
    # Объединяем настройки
    extensions = DEFAULT_EXTENSIONS | set(args.extensions)
    ignore_dirs = DEFAULT_IGNORE_DIRS | set(args.ignore)
    
    # Запуск
    try:
        create_prompt_file(root_dir, output_file, extensions, ignore_dirs, args.max_size)
    except KeyboardInterrupt:
        print("\n\n⛔ Прервано пользователем.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()