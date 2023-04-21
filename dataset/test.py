import os
import pdb
from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':

    for i in [6751, 14879, 6814, 3110]:
        for j in range(22):
            file_name = os.path.join('../../../dataset/dl/unlabeled', f'video_{i}/image_{j}.png')
            try:
                Image.open(file_name).convert('RGB')
            except:
                print(f"Can't open: {file_name}")