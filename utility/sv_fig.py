import matplotlib.pyplot as plt

def savefig(filename, crop = True):
    plt.savefig('{}.pdf'.format(filename))