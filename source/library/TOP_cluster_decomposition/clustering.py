# %%
import numpy as np 
from typing import List, Set
from classes.problem_instances.top_instances import TOPInstance
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from functools import partial

# --------------------------------------------------- CLUSTERING FUNCTIONS ---------------------------------------------- #
def cluster_hc_base(linkage: str, top_instance: TOPInstance, M: int) -> List[Set[int]]:
    """Clusters the nodes with a TOP instance"""
    positions = np.array([node.pos for node in top_instance.nodes])
    distance_matrix = np.array([[np.linalg.norm(p - q) for p in positions] for q in positions])

    labels = AgglomerativeClustering(n_clusters=M, metric="precomputed", linkage=linkage).fit_predict(distance_matrix)

    assert labels[0] != labels[-1], "Source and sink must be in different clusters."

    # Reorder clusters so that C_1 contains the source and C_M contains the sink.
    clusters = ([[i for i in range(len(positions)) if labels[i] == labels[0]]] + 
                [list() for _ in range(M - 2)] + 
                [[i for i in range(len(positions)) if labels[i] == labels[-1]]])
    
    for i, label in enumerate([i for i in range(M) if not (i in {labels[0], labels[-1]})]):
        clusters[i + 1] = [i for i in range(len(positions)) if labels[i] == label]

    return clusters

cluster_hc_single_linkage = partial(cluster_hc_base, "single")
cluster_hc_average_linkage = partial(cluster_hc_base, "average")
cluster_hc_complete_linkage = partial(cluster_hc_base, "complete")

cluster_hc_single_linkage.__name__ = "single"
cluster_hc_average_linkage.__name__ = "average"
cluster_hc_complete_linkage.__name__ = "complete"

def cluster_spectral(top_instance: TOPInstance, M: int) -> List[Set[int]]:
    """Clusters the nodes with a TOP instance"""
    positions = np.array([node.pos for node in top_instance.nodes])
    distance_matrix = np.array([[np.linalg.norm(p - q) for p in positions] for q in positions])
    similarity_matrix = np.exp(-distance_matrix**2 / (2. * (0.5**2))) 
    labels = SpectralClustering(n_clusters=M, affinity="precomputed").fit_predict(similarity_matrix)

    assert labels[0] != labels[-1], "Source and sink must be in different clusters."

    # Reorder clusters so that C_1 contains the source and C_M contains the sink.
    clusters = ([{i for i in range(len(positions)) if labels[i] == labels[0]}] + 
                [set() for _ in range(M - 2)] + 
                [{i for i in range(len(positions)) if labels[i] == labels[-1]}])
    
    for i, label in enumerate([i for i in range(M) if not (i in {labels[0], labels[-1]})]):
        clusters[i + 1] = {i for i in range(len(positions)) if labels[i] == label}

    return clusters

def cluster_kmeans(top_instance: TOPInstance, M: int) -> List[Set[int]]:
    """Clusters the nodes with a TOP instance using k-medoids."""
    positions = np.array([node.pos for node in top_instance.nodes])

    labels = KMeans(n_clusters=M).fit_predict(positions)
    
    # Step 5: Reorder clusters so source is in C_1 and sink in C_M
    clusters = ([[i for i in range(len(positions)) if labels[i] == labels[0]]] + 
                [list() for _ in range(M - 2)] + 
                [[i for i in range(len(positions)) if labels[i] == labels[-1]]])

    mid_labels = [i for i in range(M) if i not in {labels[0], labels[-1]}]
    for i, label in enumerate(mid_labels):
        clusters[i + 1] = [j for j in range(len(positions)) if labels[j] == label]

    return clusters

