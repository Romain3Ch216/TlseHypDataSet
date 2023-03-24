from torch.utils.data import Dataset
import os


class TlseHypDataSet(Dataset):
    """

    """
    def __init__(self, root_path: str, patch_size: int):
        self.wv = None
        self.bbl = None
        self.E_dir = None
        self.E_dif = None
        self.root_path = root_path

        self.images = [
            'TLS_3d_2021-06-15_11-10-12_reflectance_rect.bsq',
            'TLS_1c_2021-06-15_10-41-20_reflectance_rect.bsq',
            'TLS_3a_2021-06-15_11-10-12_reflectance_rect.bsq',
            'TLS_5c_2021-06-15_11-40-13_reflectance_rect.bsq',
            'TLS_1d_2021-06-15_10-41-20_reflectance_rect.bsq',
            'TLS_9c_2021-06-15_12-56-29_reflectance_rect.bsq',
            'TLS_1b_2021-06-15_10-41-20_reflectance_rect.bsq',
            'TLS_1e_2021-06-15_10-41-20_reflectance_rect.bsq'
        ]

        self.gt = 'ground_truth.shp'

        dirs_in_root = os.listdir(root_path)
        assert ('images' in dirs_in_root) and ('gt' in dirs_in_root), \
            "Root directory should include an 'images' and a 'gt' folder."

        for image in self.images:
            assert image in os.listdir(os.path.join(root_path, 'images', image)), "Image {} misses".format(image)
            header = image[:-3] + 'hdr'
            assert image in os.listdir(os.path.join(root_path, 'images', header)), "Header {} misses".format(header)

        for ext in ['cpg', 'dbf', 'shp', 'prj', 'shx']:
            assert self.gt in os.listdir(os.path.join(root_path, 'gt', self.gt[:-3] + ext))

        self.read_metadata()

    def read_metadata(self):
        assert 'metadata.txt' in os.listdir(self.root_path)
        self.wv = []
        self.bbl = []
        self.E_dir = []
        self.E_dif = []

        with open(os.path.join(self.root_path, 'metadata.txt'), 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    self.wv.append(line[0])
                    self.bbl.append(line[1])
                    self.E_dir.append(line[2])
                    self.E_dif.append(line[3])





dataset = TlseHypDataSet('/home/rothor/Documents/ONERA/Datasets/Toulouse', patch_size=32)

