# Оценка качества генеративных моделей музыки через FAD

Проект для сравнительной оценки 5 open-source моделей генерации музыки
с использованием Frechet Audio Distance (FAD) и связанных метрик.

## Модели

| Модель | Архитектура | Источник |
|--------|------------|----------|
| MusicGen (Meta) | Autoregressive Transformer + EnCodec | `facebook/musicgen-small` |
| AudioLDM | Latent Diffusion + CLAP | `cvssp/audioldm-m-full` |
| MusicLDM | Latent Diffusion + CLAP | `ucsd-reach/musicldm` |
| Riffusion | Fine-tuned Stable Diffusion | `riffusion/riffusion-model-v1` |
| AudioLDM Large | Latent Diffusion + CLAP (large) | `cvssp/audioldm-l-full` |

## Метрики

- **FAD** — Frechet Audio Distance (стандартный)
- **FAD-inf** — экстраполяция к бесконечной выборке (устранение sample size bias)
- **Per-song FAD** — для обнаружения выбросов
- **CLAP score** — соответствие промпту

Эмбеддинги: CLAP, MERT, EnCodec, VGGish.
Референсные датасеты: FMA-Pop, MusicCaps, GTZAN.

## Установка

```bash
pip install -r requirements.txt
```

Для fallback-подготовки `MusicCaps` по `ytid + start_s` также требуется `ffmpeg`
в системе (CLI). На macOS:

```bash
brew install ffmpeg
```

## Запуск

```bash
# 1. Генерация музыки всеми моделями
python generate.py --models all --num-prompts 250

# 2. Вычисление FAD метрик
python evaluate.py --embeddings clap mert encodec vggish --references fma gtzan musiccaps --download-references

# 3. Анализ и визуализация
python analyze.py
```

## Структура

```
├── config.py              # Конфигурация
├── generate.py            # Генерация музыки
├── evaluate.py            # Вычисление FAD
├── analyze.py             # Визуализация результатов
├── generators/            # Генераторы для каждой модели
├── evaluation/            # Модули оценки
├── prompts/               # Текстовые промпты
└── outputs/               # Результаты
    ├── generated/         # Сгенерированная музыка
    ├── embeddings/        # Кешированные эмбеддинги
    ├── results/           # CSV/JSON метрики
    └── plots/             # Графики
```
