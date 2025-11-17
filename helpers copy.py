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
    os.makedirs(f"/content/drive/MyDrive/Colab/output/punto{punto}", exist_ok=True)
    os.makedirs(f"/content/drive/MyDrive/Colab/models/punto{punto}", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/Colab/results", exist_ok=True)


def save_model(model, model_name, punto):
    """
    Guarda un modelo en el directorio correspondiente
    """
    ensure_directories(punto)
    model_path = f"/content/drive/MyDrive/Colab/models/punto{punto}/{model_name}.model"
    model.save(model_path)
    return model_path


def save_experiment_results(punto, experiment_name, results):
    """
    Guarda los resultados de un experimento en JSON
    """
    ensure_directories(punto)
    results_file = f"/content/drive/MyDrive/Colab/results/punto{punto}_results.json"
    
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
    results_file = f"/content/drive/MyDrive/Colab/results/punto{punto}_results.json"
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
    
    output_path = f'/content/drive/MyDrive/Colab/output/punto{punto}/{model_name}_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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
    
    output_path = f'/content/drive/MyDrive/Colab/output/punto{punto}/{model_name}_pca.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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
    cache_dir = f"/content/drive/MyDrive/Colab/input/preprocessed/punto{punto}"
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
    cache_dir = f"/content/drive/MyDrive/Colab/input/preprocessed/punto{punto}"
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
    cache_dir = f"/content/drive/MyDrive/Colab/input/preprocessed/punto{punto}"
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
    cache_dir = f"/content/drive/MyDrive/Colab/input/preprocessed/punto{punto}"
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
    cache_dir = f"/content/drive/MyDrive/Colab/input/preprocessed/punto{punto}"
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
    cache_dir = f"/content/drive/MyDrive/Colab/input/preprocessed/punto{punto}"
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
    
    output_path = f"/content/drive/MyDrive/Colab/output/punto{punto}/{model_key.replace('/', '_')}_pca.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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
    history_file = f"/content/drive/MyDrive/Colab/results/punto{punto}_{model_name}_history.json"
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"Historial guardado en: {history_file}")
    return history_file


def load_training_history(model_name, punto):
    """
    Carga el historial de entrenamiento
    """
    history_file = f"/content/drive/MyDrive/Colab/results/punto{punto}_{model_name}_history.json"
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def plot_training_history(history, model_name, punto):
    """
    Visualiza el historial de entrenamiento
    """
    ensure_directories(punto)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['eval_loss'], 'r-', label='Eval Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history['eval_accuracy'], 'g-', label='Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, history['eval_f1'], 'c-', label='F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history['eval_precision'], 'm-', label='Precision')
    axes[1, 1].plot(epochs, history['eval_recall'], 'y-', label='Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle(f'Training History - {model_name}', fontsize=16)
    plt.tight_layout()
    
    output_path = f"/content/drive/MyDrive/Colab/output/punto{punto}/{model_name}_training_history.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {output_path}")
    return output_path
