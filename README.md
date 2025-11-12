# Universe Project

Deze repository bevat de game-logica in `Universe.py`.

## Vereisten installeren

```bash
pip install -r requirements.txt
```

## Codekwaliteit

Formatter en linter worden geconfigureerd via `pyproject.toml`.

### Black

```bash
black .
```

### Ruff

```bash
ruff check .
```

## Pre-commit (optioneel)

1. Installeer pre-commit:
   ```bash
   pip install pre-commit
   ```
2. Activeer de hooks:
   ```bash
   pre-commit install
   ```
3. (Optioneel) Draai alle hooks handmatig:
   ```bash
   pre-commit run --all-files
   ```

