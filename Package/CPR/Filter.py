from sklearn.neighbors import NearestNeighbors

def hardFilterFunction(cluster_function, pick, data_set):
    check = cluster_function.predict([pick])
    return data_set[cluster_function.predict(data_set)==check]

def nearestFilterFunction(pick, data_set, cut = .50):
    cutt = int(round(len(data_set)*cut, 0))
    neigh = NearestNeighbors(n_neighbors=cutt)
    neigh.fit(data_set)
    return data_set.iloc[neigh.kneighbors([pick], cutt, return_distance=False)[0].tolist()]