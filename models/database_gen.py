#!/usr/bin/env python3
import PIL

BASE_FACE_ID_ADDRESS = '../resources/face_identification/FDDB-folds/'

fold1_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-01.txt'
fold2_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-02.txt'
fold3_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-03.txt'
fold4_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-04.txt'
fold5_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-05.txt'
fold6_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-06.txt'
fold7_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-07.txt'
fold8_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-08.txt'
fold9_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-09.txt'
fold10_address = BASE_FACE_ID_ADDRESS + 'FDDB-fold-10.txt'

fold_addresses = [fold1_address, 
                  fold2_address, 
                  fold3_address, 
                  fold4_address, 
                  fold5_address, 
                  fold6_address, 
                  fold7_address, 
                  fold8_address, 
                  fold9_address, 
                  fold10_address]

image_addresses = {
        'Image': []
    }

def read_fold(f):
    for line in f:
        img = '../resources/face_identification/' + line
        img = img.rstrip(img[-1])
        image_addresses['Image'].append(img)
        tmp = open(img)
        PIL.Image.open(tmp)

if __name__ == '__main__':
    
    for fold in fold_addresses:
        fold_f = open(fold)
        read_fold(fold_f)
        fold_f.close()
    
    print(len(image_addresses['Image']))