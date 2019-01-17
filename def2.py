import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import math


# Funcao de processamento do dataset
def readImages ():
    # Leitura das imagens
    imgs = glob.glob("DatasetMML/*.gif")
    base = [Image.open(i).convert('L') for i in imgs]
    
    # Tamanho do dataset
    size = len(base)
    
    # Passar as imagens para um array
    X = np.array([base[i].getdata() for i in range(size)])
    return X, base, size


# Implementacao do PCA
def pca(X, num_comp=0, confidence=0.8):
    # Media do dataset
    mean = np.mean(X,0)
    
    # Centrar os dados
    phi = X - mean
    
    # Calcular os vetores e valores proprios atraves do SVD
    eigenvectors, sigma, variance = np.linalg.svd(phi.transpose(), full_matrices=False)
    eigenvalues = sigma*sigma
    
    # Ordenacao dos valores pp
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Determinar o n. de vectores pp a usar
    k = 0
    traco = np.sum(eigenvalues)
    while(np.sum(eigenvalues[:k])/traco < confidence):
        k = k+1
    print(k)
    
    # Escolher os vetores pp associados
    eigenvectors = eigenvectors[:,0:num_comp]
    return eigenvalues, eigenvectors, phi, mean, variance


# Calculo dos coeficientes da projeccao
def coefProj(phi, eigenvectors, size):
    '''print((eigenvectors.shape[1], size))
    #coef_proj=np.zeros((eigenvectors.shape[1], size))
    coef_proj=np.array([])
    for i in range(size):
        coef_proj2 += np.append(coef_proj, np.array(np.dot(phi[i], eigenvectors)))'''
            
    coef_proj = [np.dot(phi[i], eigenvectors) for i in range(size)]
    coef_proj = np.reshape(coef_proj, (eigenvectors.shape[1], size))
    return coef_proj


# Verificar se identifica ou nao o input
def testar(input_img, mean, eigenvectors, eigenvalues, limit, size, coef_proj):
    # Centrar o input
    gamma = np.array(input_img.getdata())
    test_phi = gamma - mean
    
    # Calcular os coeficientes da projeccao do input
    test_coef_proj = np.dot(test_phi, eigenvectors)
    
    
    #dist = [mahalanobis(coef_proj, test_coef_proj, eigenvalues, eigenvectors.shape[1]) for i in range(size)]
    
    dist = mahalanobis(coef_proj, test_coef_proj, eigenvalues, eigenvectors.shape[1])
    #print(dist)
    d_min = np.min(dist)
    if d_min < limit:
        print('Imagem nr.: '+str(np.argmin(dist))+'\n'+'Distancia minima: '+str(d_min))
    else:
        print('Falhou no reconhecimento.')
        print('Imagem nr.: '+str(np.argmin(dist))+'\n'+'Distancia minima: '+str(d_min))
    return test_coef_proj

# Distancia euclidiana
def euclidian(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    distance = math.sqrt(sum(z**2))
    return round(distance, 2)

# Distance de Mahalanobis
def mahalanobis(x, y, eigenvalues, k):
    if x.shape[0] != y.shape[0]:
        return (-1)
    N = x.shape[1]
    distance=[]
    for i in range(N):
        distance.append(np.sum(np.multiply((x[:,i]-y)**2, eigenvalues[:k])))
        print((x[:,i]-y)**2)
    return distance


