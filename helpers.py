import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime


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
    
    output_path = f'./output/punto{punto}/{model_name}_pca.png'
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
