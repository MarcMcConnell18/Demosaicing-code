import glob
import os
import cv2
import numpy as np
from multiprocessing import Pool

patch_size, stride = 40, 10
aug_modes = [0] 
def data_aug(img, mode=0):
    
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    
def gen_patches(file_name):

    #read image
    img = cv2.imread(file_name) # read in color
    h, w, _ = img.shape # _ captures the color channels
    patches = []

    for mode in aug_modes:
        # extract patches
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                x = img[i:i+patch_size, j:j+patch_size]
                x_aug = data_aug(x, mode=mode)
                patches.append(x_aug)
    return patches

if __name__ == '__main__':
    src_dir = './data/set68_sigma5/'
    save_dir = './data/set68_sigma5_npydata/'    
    file_list = glob.glob(src_dir+'*.png')  # get name list of all .png files
    num_threads = 16   
    max_patches_per_file = 5000
    file_count = 0
    print('Start...')
    # initrialize
    res = []
    # generate patches
    for i in range(0, len(file_list), num_threads):
        # use multi-process to speed up
        p = Pool(num_threads)
        patch = p.map(gen_patches,file_list[i:min(i+num_threads,len(file_list))])
        p.close()
        p.join()

        for x in patch:
            res += x
            if len(res) >= max_patches_per_file:
                save_path = os.path.join(save_dir, f'clean_patches_{file_count}.npy')
                np.save(save_path, np.array(res, dtype='uint8'))
                print(f'Saved {save_path}')
                res = []
                file_count += 1
        print(f'Processed images {i} to {min(i+num_threads, len(file_list))}')
        
    if len(res) > 0:
        save_path = os.path.join(save_dir, f'clean_patches_{file_count}.npy')
        np.save(save_path, np.array(res, dtype='uint8'))
        print(f'Saved {save_path}')
