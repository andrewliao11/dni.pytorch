import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pickle as pkl

def plot(data=None, path=None, name='DNI'):
    assert data is not None or path is not None
    if path is not None:
        data = pkl.load(open(path))
    
    grad_loss = data['grad_loss']
    classify_loss = data['classify_loss']
    x = np.arange(len(classify_loss))

    plt.plot(classify_loss)
    plt.title('classify_loss')
    plt.savefig(name+'_classify_loss.png')    
    plt.close()

    plt.semilogy(grad_loss, 'r')
    plt.title('grad_loss')
    plt.savefig(name+'_grad_loss.png')
    plt.close()

