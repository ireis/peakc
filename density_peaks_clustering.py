
import numpy
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

def get_kd_tree(X):
    return KDTree(X)

def get_counts(kd_tree, X, d_c):
    counts = kd_tree.query_radius(X, d_c, count_only=True)
    counts_asort = numpy.argsort(counts)
    return counts, counts_asort

def get_distances(X, objs, other_objs, distance = 'euclidean'):
    
    if distance == 'euclidean':
        if len(X[objs].shape) == 1:
            X_objs = X[objs].reshape(1,-1)
        else:
            X_objs = X[objs]
            
        if len(X[other_objs].shape) == 1:
            X_other_objs = X[other_objs].reshape(1,-1)
        else:
            X_other_objs = X[other_objs]
            
        distances = euclidean_distances(X_objs, X_other_objs)
        
    elif distance == 'precomputed':
        distances = X[obj, other_objs]
    
    return distances

def calc_deltas(X, counts_asort, distance = 'euclidean'):
    
    nof_objects = X.shape[0]
    delta = numpy.zeros(nof_objects)
    
    for i in tqdm(range(nof_objects-1)):
        distances = get_distances(X, counts_asort[i], counts_asort[(i+1):], distance)
        min_distance = numpy.min(distances)
        delta[counts_asort[i]] = min_distance
        
    distances = get_distances(X, counts_asort[-1], counts_asort, distance)
    delta[counts_asort[-1]] = numpy.max(distances)
    
    return delta

def get_deltas_and_counts(X, d_c, distance = 'euclidean'):
    
    kd_tree = get_kd_tree(X)
    counts, counts_asort = get_counts(kd_tree, X, d_c)
    deltas = calc_deltas(X, counts_asort, distance)
    
    return deltas, counts

def get_centers(delta, counts, delta_cut=None, counts_cut=None, n_std=None):
    
    inds = None
    if (not delta_cut is None) or (not counts_cut is None):
        if (counts_cut is None):
            inds = numpy.where( (delta > delta_cut) )[0]
        elif (delta_cut is None):
            inds = numpy.where( (counts > counts_cut) )[0]
        else:
            inds = numpy.where( (counts > counts_cut) & (delta > delta_cut) )[0]
        return inds
    else:
        return get_centers_bins(delta, counts, n_std)
            
    
from statsmodels import robust

def get_centers_bins(delta, counts, n_std=None):
    
    grps = numpy.array_split(numpy.argsort(counts), 10)
    
    if n_std is None:
        n_std = 5.5
        
    inds = []
    for g in grps:
        cmean = numpy.median(delta[g])
        cstd = robust.mad(delta[g])
        grp_inds = g[numpy.where( (delta[g] - cmean ) > (n_std * cstd) )[0]]
        inds += [grp_inds]
        
    return numpy.concatenate(inds)
        
    
    

def get_assingments(X, centers, distance = 'euclidean'):

    nof_objects = X.shape[0]
    distances_to_centers = get_distances(X, numpy.arange(nof_objects), centers, distance)
    assingment = numpy.zeros(nof_objects)
    
    for i in tqdm(range(nof_objects)):
        distances = distances_to_centers[i]
        assingment[i] = numpy.argmin(distances) 
    
    return assingment

def get_clusters_indx(assingment):
    
    clusters = numpy.unique(assingment).astype(int)
    cluster_members = []
    for c in clusters:
        members = numpy.where(assingment == c)[0]
        cluster_members += [members]
        
    return clusters, cluster_members


def get_border_density(kd_tree, X, d_c, counts, centers, clusters, cluster_members, distance):
    
    nof_centers = len(centers)
    border_density = numpy.zeros(nof_centers)
    
    dmat_ngbs = kd_tree.query_radius(X, d_c, count_only=False)

    for c in tqdm(clusters):
        members = cluster_members[c]
        border_density_all = numpy.zeros(len(members))

        for c_test in clusters:
            cc_dist = get_distances(X, centers[c], centers[c_test], distance)

            if (c_test != c) and (cc_dist < 15*d_c):
                c_test_members = cluster_members[c_test]
                intersect = numpy.array([len(numpy.intersect1d(q,c_test_members)) for q in dmat_ngbs[members]])
                
                border_density_all[intersect > 1] = counts[members[intersect > 1]]
                
        border_density[c] = numpy.max(border_density_all)
        
    return border_density
    
def get_core_cluster_members(nof_objects, clusters, cluster_members, border_density, counts):
    
    core_cluster_members = []
    assingment = numpy.ones(nof_objects)*(-1)
    for c in clusters:
        members = cluster_members[c]
        core_members = members[numpy.where(counts[members] > border_density[c])[0]]
        core_cluster_members += [core_members]
        assingment[core_members] = c
        if False:
            print('Cluster', c, len(core_members), 'core members', core_members[:10])
    
    return core_cluster_members, assingment

def get_clusters(X, delta, counts, d_c, delta_cut=None, counts_cut=None, distance = 'euclidean', kd_tree=None, n_std=None):
    
    nof_objects = X.shape[0]
    centers = get_centers(delta, counts, delta_cut, counts_cut, n_std)
    
    if centers is None:
        print('No centers found, change cuts')
        return
    
    if len(centers) == 0:
        print('No centers found, change cuts')
        return
    
    assingments_full = get_assingments(X, centers, distance)
    cluster_inds, cluster_members = get_clusters_indx(assingments_full)
    
    if kd_tree is None:
        kd_tree = get_kd_tree(X)
        
    border_density = get_border_density(kd_tree, X, d_c, counts, centers, cluster_inds, cluster_members, distance)

    core_cluster_members, assingment_core = get_core_cluster_members(nof_objects, cluster_inds, 
                                                                     cluster_members, border_density, counts)
            
    return centers, core_cluster_members, assingment_core


def delta_counts_plot(delta, counts, delta_cut=None, counts_cut=None, log=False, n_std=None):
    
    plt.figure(figsize = (15,7))
    
    delta_plt = delta.copy()
    if log:
        delta_plt[delta_plt > 0] = numpy.log(delta_plt[delta_plt > 0])
        delta_plt[delta_plt == 0] = numpy.nan
        ylabel = r'$Log(\delta)$'
    else:
        ylabel = r'$\delta$'
    plt.scatter(counts, delta_plt)
    inds = None
        
    inds = get_centers(delta, counts, delta_cut, counts_cut, n_std)

    if len(inds) > 0:
        plt.scatter(counts[inds], delta_plt[inds], s = 250)
    else:
        print('No objects selected')


    plt.xlabel('Counts', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid()
    
    plt.show()
    return inds
    
