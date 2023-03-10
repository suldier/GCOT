import sys

sys.path.append('.')

from GCOT import GCOT
import numpy as np

def order_sam_for_diag(x, y):
    """
    rearrange samples
    :param x: feature sets
    :param y: ground truth
    :return:
    """
    x_new = np.zeros(x.shape)
    y_new = np.zeros(y.shape)
    index = np.zeros(y.shape,dtype=np.int32)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        index[idx] = np.arange(start,stop) 
        start = stop
    return x_new, y_new, index 



if __name__ == '__main__':
    from Toolbox.Preprocessing import Processor
    from sklearn.preprocessing import minmax_scale, normalize
    from sklearn.decomposition import PCA
    import time
    import matplotlib.pyplot as plt
    root = 'HSI_Datasets/'

    #im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    #im_, gt_ = 'PaviaU', 'PaviaU_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print('\nDataset: ', img_path)

    PATCH_SIZE = 9  # # 9 default, normally the bigger the better
    nb_comps = 4  # # num of PCs, 4 default, it can be moderately increased
    # load img and gt
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)

    # # take a smaller sub-scene for computational efficiency
    if im_ == 'SalinasA_corrected':
        eps1, eps2, k, rho =0.073, 0.068, 10, 0.2
    if im_ == 'Indian_pines_corrected':
        img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
        eps1, eps2, k, rho =0.3, 0.32, 20, 0.3
    if im_ == 'PaviaU':
        img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
        eps1, eps2, k, rho =1.9, 0.9, 23, 0.3 
    n_row, n_column, n_band = img.shape
    x_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    print('original img shape: ', x_img.shape)
    # reduce spectral bands using PCA
    pca = PCA(n_components=nb_comps)
    img = minmax_scale(pca.fit_transform(img.reshape(n_row * n_column, n_band))).reshape(n_row, n_column, nb_comps)
    x_patches, y_ = p.get_HSI_patches_rw(img, gt, (PATCH_SIZE, PATCH_SIZE))

    print('reduced img shape: ', img.shape)
    print('x_patch tensor shape: ', x_patches.shape)
    n_samples, n_width, n_height, n_band = x_patches.shape
    x_patches_2d = np.reshape(x_patches, (n_samples, -1))
    y, lab_map = p.standardize_label(y_)

    # reorder samples according to gt
    x_patches_2d, y, index = order_sam_for_diag(x_patches_2d, y)
    # normalize data
    x_patches_2d = normalize(x_patches_2d)
    print('final sample shape: %s, labels: %s' % (x_patches_2d.shape, np.unique(y)))
    N_CLASSES = np.unique(y).shape[0]  # Indian : 8  SalinasA : 6 PaviaU : 8

    # ========================
    # performing  GCOT
    # ========================
    gcot = GCOT(n_clusters=N_CLASSES)

    # ========================
    # Results and Visualization
    # ========================
    C = gcot.fit_base(x_patches_2d, eps1)
    C = gcot.fit_gcot(x_patches_2d, eps2, C, k)
    acc, _, y_best = gcot.call_acc(C, y, rho)
    print('%10s %10s %10s' % ('OA', 'Kappa','NMI'))
    print('%10.4f %10.4f %10.4f' % (acc[0], acc[2], acc[1]))
    y_best = np.array([lab_map[i] for i in y_best[index]])
    p.show_pre1(gt, np.arange(y.shape[0]), y_best, '%s_gcot.pdf' % im_)
