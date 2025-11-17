# Taller 2 PLN - Embeddings y Fine-tuning

## Enlaces

- **Repositorio GitHub**: [https://github.com/Yenreh/Workshop2PLN](https://github.com/Yenreh/Workshop2PLN)
- **Modelos en Hugging Face**:
  - [xlm-roberta-large-tass-sentiment-bs8](https://huggingface.co/Yenreh/xlm-roberta-large-tass-sentiment-bs8)
  - [xlm-roberta-large-tass-sentiment-bs16](https://huggingface.co/Yenreh/xlm-roberta-large-tass-sentiment-bs16)

## Requisitos del Sistema

- Python 3.8+
- CUDA 12.1+ (para GPU NVIDIA RTX 5070)
- 16GB RAM mínimo
- 50GB espacio en disco

## Configuración del Entorno

### Crear entorno conda

```bash
conda create -n pln_taller2 python=3.10
conda activate pln_taller2
```

### Instalación de PyTorch para RTX 5070

```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Descargar recursos NLTK

```python
import nltk
nltk.download('stopwords')
```

## Estructura del Proyecto

```
Workshop2PLN/
├── Tarea2_Punto1.ipynb    # Embeddings de palabras (Word2Vec/FastText)
├── Tarea2_Punto2.ipynb    # Embeddings de oraciones (Sentence Transformers)
├── Tarea2_Punto3.ipynb    # Fine-tuning (xml-roberta-large)
├── input/
│   └── pdfs/              # Documentos PDF para procesar
├── models/                # Modelos entrenados
└── output/                # Visualizaciones y resultados
```

## Ejecución

### Punto 1: Word2Vec y FastText

```bash
jupyter notebook Tarea2_Punto1.ipynb
```

Procesa el dataset `spanish_billion_words` y entrena modelos de embeddings de palabras.

### Punto 2: Sentence Transformers

```bash
jupyter notebook Tarea2_Punto2.ipynb
```

Procesa documentos PDF y genera embeddings de oraciones con 4 modelos diferentes.

### Punto 3: Fine-tuning (Requiere GPU)

```bash
jupyter notebook Tarea2_Punto3.ipynb
```

Fine-tuning de xml-roberta-large para clasificación de sentimientos. Requiere:
- Token de Hugging Face
- Dataset de Tweets
- GPU

## Configuración GPU

Verificar GPU disponible:

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Datasets Requeridos

- `spanish_billion_words`: Disponible en Hugging Face
- PDFs: Colocar en `./input/pdfs/`
- Tweets: Dataset propio para clasificación de sentimientos

## Notas Técnicas

- El Punto 3 consume aproximadamente 12-15GB de VRAM
- Los modelos Word2Vec/FastText se guardan en `./models/`
- Las visualizaciones se exportan a `./output/`
- Batch size recomendado para RTX 5070: 16

## Troubleshooting

### Error de memoria GPU
Reducir batch_size en TrainingArguments

### Dataset no encontrado
Verificar rutas en variables PDF_DIR

### Token Hugging Face inválido
Obtener token en https://huggingface.co/settings/tokens
