import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial


def ensure_directories(punto):
    """
    Crea los directorios necesarios para cada punto
    """
    os.makedirs(f"./output/punto{punto}", exist_ok=True)
    os.makedirs(f"./models/punto{punto}", exist_ok=True)
    os.makedirs("./results", exist_ok=True)


def save_model(model, model_name, punto):
    """
    Guarda un modelo en el directorio correspondiente
    """
    ensure_directories(punto)
    model_path = f"./models/punto{punto}/{model_name}.model"
    model.save(model_path)
    return model_path


def save_experiment_results(punto, experiment_name, results):
    """
    Guarda los resultados de un experimento en JSON
    """
    ensure_directories(punto)
    results_file = f"./results/punto{punto}_results.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    results['timestamp'] = datetime.now().isoformat()
    results['experiment_name'] = experiment_name
    all_results.append(results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return results_file


def load_experiment_results(punto):
    """
    Carga los resultados de experimentos de un punto
    """
    results_file = f"./results/punto{punto}_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def visualize_embeddings_tsne(model, model_name, punto, num_words=100, perplexity=30):
    """
    Visualiza embeddings usando t-SNE
    """
    ensure_directories(punto)
    
    words = list(model.wv.index_to_key[:num_words])
    vectors = np.array([model.wv[word] for word in words])
    
    print(f"Aplicando t-SNE para {model_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(words)-1))
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'Visualización t-SNE de Embeddings - {model_name}', fontsize=14)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = f'./output/punto{punto}/{model_name}_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico guardado: {output_path}")
    return output_path


def visualize_embeddings_pca(model, model_name, punto, num_words=100):
    """
    Visualiza embeddings usando PCA
    """
    ensure_directories(punto)
    
    words = list(model.wv.index_to_key[:num_words])
    vectors = np.array([model.wv[word] for word in words])
    
    print(f"Aplicando PCA para {model_name}...")
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'Visualización PCA de Embeddings - {model_name}', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = f'./output/punto{punto}/{model_name}_pca.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    variance_explained = float(sum(pca.explained_variance_ratio_))
    print(f"Gráfico guardado: {output_path}")
    print(f"Varianza explicada total: {variance_explained:.2%}")
    
    return output_path, variance_explained


def query_similar_words(model, word, topn=10):
    """
    Consulta las palabras más similares a una palabra dada
    """
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        results = {
            'word': word,
            'similar_words': [
                {'word': w, 'similarity': float(sim)} 
                for w, sim in similar_words
            ]
        }
        return results
    except KeyError:
        return {
            'word': word,
            'error': 'Palabra no encontrada en vocabulario'
        }


def print_experiment_summary(punto):
    """
    Imprime un resumen de todos los experimentos de un punto
    """
    results = load_experiment_results(punto)
    
    if not results:
        print(f"No hay resultados guardados para el Punto {punto}")
        return
    
    print("\n" + "="*80)
    print(f"RESUMEN DE EXPERIMENTOS - PUNTO {punto}")
    print("="*80)
    
    for i, exp in enumerate(results, 1):
        print(f"\nExperimento {i}: {exp['experiment_name']}")
        print(f"Timestamp: {exp['timestamp']}")
        print("-"*80)
        
        if 'model_type' in exp:
            print(f"Tipo de modelo: {exp['model_type']}")
        if 'vector_size' in exp:
            print(f"Dimensión de embeddings: {exp['vector_size']}")
        if 'dataset_size' in exp:
            print(f"Tamaño del dataset: {exp['dataset_size']:,} sentencias")
        if 'vocab_size' in exp:
            print(f"Vocabulario: {exp['vocab_size']:,} palabras")
        if 'training_time' in exp:
            print(f"Tiempo de entrenamiento: {exp['training_time']:.2f} segundos")
        if 'variance_explained' in exp:
            print(f"Varianza explicada (PCA): {exp['variance_explained']:.2%}")
        
        if 'similar_words_sample' in exp:
            print(f"\nEjemplo de similitud para '{exp['similar_words_sample']['word']}':")
            for sw in exp['similar_words_sample']['similar_words'][:3]:
                print(f"  {sw['word']}: {sw['similarity']:.4f}")
        
        if 'model_path' in exp:
            print(f"\nModelo guardado en: {exp['model_path']}")
        if 'visualizations' in exp:
            print(f"Visualizaciones:")
            for viz in exp['visualizations']:
                print(f"  {viz}")
    
    print("\n" + "="*80)


def clear_model_from_memory(model):
    """
    Limpia un modelo de la memoria
    """
    del model
    import gc
    gc.collect()


def save_preprocessed_sentences(sentences, filename, punto):
    """
    Guarda sentencias preprocesadas en formato pickle
    """
    ensure_directories(punto)
    cache_dir = f"./input/preprocessed/punto{punto}"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = f"{cache_dir}/{filename}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Sentencias guardadas en: {filepath}")
    print(f"Total: {len(sentences):,} sentencias")
    return filepath


def load_preprocessed_sentences(filename, punto):
    """
    Carga sentencias preprocesadas desde pickle
    """
    cache_dir = f"./input/preprocessed/punto{punto}"
    filepath = f"{cache_dir}/{filename}.pkl"
    
    if os.path.exists(filepath):
        print(f"Cargando sentencias desde cache: {filepath}")
        with open(filepath, 'rb') as f:
            sentences = pickle.load(f)
        print(f"Cargadas {len(sentences):,} sentencias")
        return sentences
    else:
        print(f"Cache no encontrado: {filepath}")
        return None


def preprocess_single_text(text_data, stop_words):
    """
    Procesa un solo texto (para multiprocesamiento)
    """
    import re
    import string
    
    sent, idx = text_data
    
    if len(sent) <= 1:
        return None
    
    words = sent.split()
    words = [w for w in words if not w.isdigit()]
    words = [re.sub(r'[0-9]', '', w) for w in words]
    words = [w for w in words if w.lower() not in stop_words]
    
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    words = [re_punc.sub('', w) for w in words]
    words = [re.sub(r"\!|\'|\?|\¿|\¡|\«|\»", "", w) for w in words]
    words = [w.lower() for w in words if w != '']
    
    if len(words) > 0:
        return words
    return None


def preprocess_text_parallel(dataset_sample, stop_words, n_workers=None):
    """
    Preprocesa textos usando multiprocesamiento
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"Procesando {len(dataset_sample['text']):,} textos con {n_workers} workers...")
    
    texts_with_idx = [(text, idx) for idx, text in enumerate(dataset_sample['text'])]
    
    process_func = partial(preprocess_single_text, stop_words=stop_words)
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_func, texts_with_idx, chunksize=1000)
    
    sentences = [result for result in results if result is not None]
    
    print(f"Total de oraciones procesadas: {len(sentences):,}")
    return sentences


# ============================================================================
# FUNCIONES PARA EL PUNTO 2 - EMBEDDINGS DE ORACIONES
# ============================================================================

def save_embeddings(embeddings, filename, punto):
    """
    Guarda embeddings en formato numpy
    """
    ensure_directories(punto)
    cache_dir = f"./input/preprocessed/punto{punto}"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = f"{cache_dir}/{filename}.npy"
    np.save(filepath, embeddings)
    
    print(f"Embeddings guardados en: {filepath}")
    print(f"Shape: {embeddings.shape}")
    return filepath


def load_embeddings(filename, punto):
    """
    Carga embeddings desde archivo numpy
    """
    cache_dir = f"./input/preprocessed/punto{punto}"
    filepath = f"{cache_dir}/{filename}.npy"
    
    if os.path.exists(filepath):
        print(f"Cargando embeddings desde: {filepath}")
        embeddings = np.load(filepath)
        print(f"Shape: {embeddings.shape}")
        return embeddings
    else:
        print(f"Archivo no encontrado: {filepath}")
        return None


def save_chunks(chunks, filename, punto):
    """
    Guarda chunks de texto en formato pickle
    """
    ensure_directories(punto)
    cache_dir = f"./input/preprocessed/punto{punto}"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = f"{cache_dir}/{filename}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Chunks guardados en: {filepath}")
    print(f"Total: {len(chunks):,} chunks")
    return filepath


def load_chunks(filename, punto):
    """
    Carga chunks desde pickle
    """
    cache_dir = f"./input/preprocessed/punto{punto}"
    filepath = f"{cache_dir}/{filename}.pkl"
    
    if os.path.exists(filepath):
        print(f"Cargando chunks desde: {filepath}")
        with open(filepath, 'rb') as f:
            chunks = pickle.load(f)
        print(f"Cargados {len(chunks):,} chunks")
        return chunks
    else:
        print(f"Cache no encontrado: {filepath}")
        return None


def visualize_pca_comparison(query, search_results, embeddings_by_model, model_key, punto, num_words=100):
    """
    Visualiza con PCA la consulta y los fragmentos más similares para comparación
    """
    ensure_directories(punto)
    
    from sklearn.decomposition import PCA
    
    emb_data = embeddings_by_model[model_key]
    config = emb_data['config']
    all_embeddings = emb_data['embeddings']
    
    top_result = search_results[model_key]['top_chunks'][0]
    best_chunk_idx = top_result[0]
    similarity_score = top_result[1]
    
    query_embedding = search_results[model_key]['query_embedding']
    best_chunk_embedding = all_embeddings[best_chunk_idx].reshape(1, -1)
    
    combined_embeddings = np.vstack([
        query_embedding,
        best_chunk_embedding
    ])
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(combined_embeddings)
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], 
               c='red', s=200, marker='*', label='Query', zorder=3)
    
    plt.scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], 
               c='green', s=200, marker='o', label='Mejor Fragmento', zorder=3)
    
    plt.annotate('Query', xy=(embeddings_2d[0, 0], embeddings_2d[0, 1]),
                xytext=(10, 10), textcoords='offset points', 
                fontsize=10, fontweight='bold')
    
    plt.annotate(f'Fragmento\n(sim: {similarity_score:.3f})', 
                xy=(embeddings_2d[1, 0], embeddings_2d[1, 1]),
                xytext=(10, -20), textcoords='offset points', 
                fontsize=10, fontweight='bold')
    
    plt.plot([embeddings_2d[0, 0], embeddings_2d[1, 0]], 
            [embeddings_2d[0, 1], embeddings_2d[1, 1]], 
            'b--', alpha=0.5, linewidth=1)
    
    plt.title(f'Visualización PCA - {config["name"]}\n'
             f'Query: "{query[:50]}..."', fontsize=12)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = f"./output/punto{punto}/{model_key.replace('/', '_')}_pca.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico guardado: {output_path}")
    return output_path


# ============================================================================
# FUNCIONES PARA EL PUNTO 3 - FINE-TUNING
# ============================================================================

def save_training_history(history, model_name, punto):
    """
    Guarda el historial de entrenamiento
    """
    ensure_directories(punto)
    history_file = f"./results/punto{punto}_{model_name}_history.json"
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"Historial guardado en: {history_file}")
    return history_file


def load_training_history(model_name, punto):
    """
    Carga el historial de entrenamiento
    """
    history_file = f"./results/punto{punto}_{model_name}_history.json"
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def plot_training_history(history, model_name, punto, output_dir=None):
    """
    Visualiza el historial de entrenamiento
    Compatible con el formato del MetricsHistoryCallback del punto 3
    
    Args:
        history: Diccionario con el historial (puede tener 'epoch' como lista)
        model_name: Nombre del modelo para el titulo
        punto: Numero del punto (para directorio de salida)
        output_dir: Directorio de salida personalizado (opcional)
    """
    ensure_directories(punto)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Determinar epochs: puede ser lista (punto 3) o calcular desde train_loss
    if 'epoch' in history and history['epoch']:
        epochs = history['epoch']
        num_epochs = len(epochs)
    else:
        num_epochs = len(history.get('eval_loss', history.get('train_loss', [])))
        epochs = range(1, num_epochs + 1)
    
    # 1. Loss (train y eval)
    ax = axes[0, 0]
    if history.get('train_loss'):
        # Para punto 3: train_loss tiene mas puntos que eval_loss
        train_steps = np.linspace(0, num_epochs, len(history['train_loss']))
        ax.plot(train_steps, history['train_loss'], 'o-', label='Train Loss', alpha=0.6, linewidth=2)
    if history.get('eval_loss'):
        ax.plot(epochs, history['eval_loss'], 's-', label='Eval Loss', linewidth=2)
    ax.set_title('Loss durante el entrenamiento', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    if history.get('eval_accuracy'):
        ax.plot(epochs, history['eval_accuracy'], 's-', label='Accuracy', color='green', linewidth=2)
    ax.set_title('Accuracy durante el entrenamiento', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1 Score
    ax = axes[1, 0]
    if history.get('eval_f1'):
        ax.plot(epochs, history['eval_f1'], 's-', label='F1 Score', color='red', linewidth=2)
    ax.set_title('F1 Score durante el entrenamiento', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Precision y Recall
    ax = axes[1, 1]
    if history.get('eval_precision'):
        ax.plot(epochs, history['eval_precision'], 'o-', label='Precision', linewidth=2)
    if history.get('eval_recall'):
        ax.plot(epochs, history['eval_recall'], 's-', label='Recall', linewidth=2)
    ax.set_title('Precision y Recall durante el entrenamiento', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Historial de Entrenamiento - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Usar output_dir personalizado si se proporciona
    if output_dir:
        output_path = f"{output_dir}{model_name}_training_history.png"
    else:
        output_path = f"./output/punto{punto}/{model_name}_training_history.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico guardado: {output_path}")
    plt.show()
    
    return output_path


def create_experiments_comparison(all_experiments, output_dir, results_dir):
    """
    Crea un DataFrame comparativo y visualizaciones para experimentos del punto 3
    
    Args:
        all_experiments: Diccionario con todos los experimentos
        output_dir: Directorio de salida para graficas
        results_dir: Directorio de salida para CSV
    
    Returns:
        tuple: (comparison_df, best_batch, comparison_plot_path, comparison_csv_path)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Crear DataFrame con resultados comparativos
    comparison_data = []
    for exp_name, exp_results in all_experiments.items():
        batch_size = exp_results['batch_size']
        test_metrics = exp_results['test_metrics']
        train_time = exp_results['training_time']
        
        comparison_data.append({
            'Batch Size': batch_size,
            'Test Accuracy': test_metrics['test_accuracy'],
            'Test F1': test_metrics['test_f1'],
            'Test Precision': test_metrics['test_precision'],
            'Test Recall': test_metrics['test_recall'],
            'Test Loss': test_metrics['test_loss'],
            'Training Time (min)': train_time / 60
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Batch Size')
    
    # Identificar mejor modelo por F1
    best_idx = comparison_df['Test F1'].idxmax()
    best_batch = int(comparison_df.loc[best_idx, 'Batch Size'])
    
    # Visualizaciones comparativas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparacion de metricas principales
    ax = axes[0, 0]
    metrics = ['Test Accuracy', 'Test F1', 'Test Precision', 'Test Recall']
    x = np.arange(len(comparison_df))
    width = 0.2
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, comparison_df[metric], width, label=metric.replace('Test ', ''))
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Score')
    ax.set_title('Comparacion de Metricas por Batch Size')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Batch Size'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Test Loss comparison
    ax = axes[0, 1]
    ax.bar(comparison_df['Batch Size'], comparison_df['Test Loss'], color='coral')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss por Batch Size')
    ax.grid(True, alpha=0.3)
    
    # 3. Training Time comparison
    ax = axes[1, 0]
    ax.bar(comparison_df['Batch Size'], comparison_df['Training Time (min)'], color='skyblue')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Tiempo (minutos)')
    ax.set_title('Tiempo de Entrenamiento por Batch Size')
    ax.grid(True, alpha=0.3)
    
    # 4. Accuracy vs F1
    ax = axes[1, 1]
    ax.plot(comparison_df['Batch Size'], comparison_df['Test Accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(comparison_df['Batch Size'], comparison_df['Test F1'], 's-', label='F1 Score', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy vs F1 Score por Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = f"{output_dir}experiments_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Guardar comparacion en CSV
    comparison_csv_path = f"{results_dir}punto3_experiments_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    return comparison_df, best_batch, comparison_plot_path, comparison_csv_path


def plot_confusion_matrix(confusion_matrix, class_labels, title, output_path):
    """
    Visualiza una matriz de confusion
    
    Args:
        confusion_matrix: Matriz de confusion como array numpy o lista
        class_labels: Lista con nombres de las clases
        title: Titulo del grafico
        output_path: Ruta donde guardar el grafico
    
    Returns:
        str: Ruta del archivo guardado
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    cm = np.array(confusion_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax)
    ax.set_xlabel('Prediccion', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path
