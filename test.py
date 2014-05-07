import os
import imghdr
import cv2
from scipy.ndimage import measurements
from scipy.misc import imsave

def isimagefile(path):
    if imghdr.what(path) in ['gif', 'jpeg']:
        return True
    return False

images_dir = 'images/'
modified_images_dir = 'modified_images/'

def openclose(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
    return cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

if not os.path.exists(modified_images_dir):
    os.makedirs(modified_images_dir)

for root, dirs, files in os.walk(images_dir):
    
    for name in files:

        modified_root = os.path.join(modified_images_dir, root.replace(images_dir, ''))  + '/' + name.split('.')[0] + name.split('.')[1] + '/'
        
        if not os.path.exists(modified_root):
            os.makedirs(modified_root)

        if not os.path.exists(modified_root + "slices/"):
            os.makedirs(modified_root + "slices/")

        if isimagefile(os.path.join(root, name)):

            original = cv2.imread(os.path.join(root, name))
            original_thumb = cv2.resize(original, (150,150))
            cv2.imwrite(modified_root + "original_thumb." + name.split('.')[1], original_thumb)

            greyscale = cv2.split(original)[1]
            greyscale_thumb = cv2.resize(greyscale, (150,150))
            cv2.imwrite(modified_root + "greyscale." + name.split('.')[1], greyscale)
            cv2.imwrite(modified_root + "greyscale_thumb." + name.split('.')[1], greyscale_thumb)

            blur = cv2.medianBlur(greyscale, 5)
            blur_thumb = cv2.resize(blur, (150,150))
            cv2.imwrite(modified_root + "blur." + name.split('.')[1], blur)
            cv2.imwrite(modified_root + "blur_thumb." + name.split('.')[1], blur_thumb)

            equalize = cv2.equalizeHist(blur)
            equalize_thumb = cv2.resize(equalize, (150,150))
            cv2.imwrite(modified_root + "equalize." + name.split('.')[1], equalize)
            cv2.imwrite(modified_root + "equalize_thumb." + name.split('.')[1], equalize_thumb)

            otsu = cv2.threshold(equalize, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            otsu_thumb = cv2.resize(otsu, (150,150))
            cv2.imwrite(modified_root + "otsu." + name.split('.')[1], otsu)
            cv2.imwrite(modified_root + "otsu_thumb." + name.split('.')[1], otsu_thumb)

            mask = openclose(otsu)
            mask_thumb = cv2.resize(mask, (150,150))
            cv2.imwrite(modified_root + "mask." + name.split('.')[1], mask)
            cv2.imwrite(modified_root + "mask_thumb." + name.split('.')[1], mask_thumb)

            masked = original
            cv2.bitwise_xor(original, original, masked, mask)
            masked_thumb = cv2.resize(masked, (150,150))
            cv2.imwrite(modified_root + "masked." + name.split('.')[1], masked)
            cv2.imwrite(modified_root + "masked_thumb." + name.split('.')[1], masked_thumb)            

            cc = measurements.label(mask==0, structure=[
                [1,1,1],
                [1,1,1],
                [1,1,1]
            ])[0]

            objects = measurements.find_objects(cc)
            subimages = [masked[s] for s in objects]

            c = 1

            for img in subimages:
                if len(img)>2:
                    imsave(modified_root + "slices/slice_"+str(c)+".jpg",img)
                    c+=1

            print name + " done!"





