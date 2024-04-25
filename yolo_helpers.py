from exhentai_helpers import *
import os
from PIL import Image
import math
import yaml
from tqdm import tqdm


class YoloMaker:
    def __init__(self, driver, path='yolo'):
        self.driver = driver
        self.gallery_downloader = GalleryDownloader(self.driver)
        self.id = 0
        self.path = path
        self.labels = []
    def dump(self, images, bboxes, zfill=10):
        try:
            os.makedirs(os.path.join(self.path, 'images'))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join(self.path, 'labels'))
        except FileExistsError:
            pass
        for i,x in enumerate(images):
            x.save(os.path.join(self.path,'images',str(self.id).zfill(zfill)+'.jpg'))
            with open(os.path.join(self.path,'labels',str(self.id).zfill(zfill)+'.txt'), 'w') as f:
                w,h = x.size
                output = []
                for y in bboxes[i]:
                    box = (self.labels.index(y[0]),(y[3]+y[1])/2/w,(y[4]+y[2])/2/h,(y[3]-y[1])/w,(y[4]-y[2])/h)
                    output.append(' '.join([str(z) for z in box]))
                f.write('\n'.join(output))
            self.id += 1
    def process_pair(self, pair, zfill=10):
        gallery1 = self.gallery_downloader.download(pair['url1'], write=False)
        gallery2 = self.gallery_downloader.download(pair['url2'], write=False)
        images = []
        bboxes = []
        for i,(x,y) in tqdm(enumerate(pair['map'])):
            img1 = gallery1.images[x]
            img2 = gallery2.images[y]
            img2 = img2.resize(img1.size)
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')
            images.extend([img1,img2])
            bb = pair['boxes'][i]
            bboxes1 = [(pair['lang1'],) + z for z in bb]
            bboxes2 = [(pair['lang2'],) + z for z in bb]
            bboxes.extend([bboxes1, bboxes2])
        self.dump(images,bboxes,zfill=zfill)
    def make(self, pairs):
        count = sum([len(pairs[x]['boxes']) for x in pairs])*2
        for key in pairs:
            if pairs[key]['lang1'] not in self.labels:
                self.labels.append(pairs[key]['lang1'])
            if pairs[key]['lang2'] not in self.labels:
                self.labels.append(pairs[key]['lang2'])
        for key in pairs:
            print(key)
            self.process_pair(pairs[key], zfill=math.ceil(math.log(count,10)))
        with open(os.path.join(self.path,'yolo.yaml'), 'w') as f:
            output = dict()
            output['path'] = self.path
            output['train'] = 'images'
            output['val'] = 'images'
            output['names'] = {j:y for j,y in enumerate(self.labels)}
            f.write(yaml.dump(output))
        self.gallery_downloader.dump()

if __name__ == '__main__':
    driver = login_exhentai()
    pairs = pickle.load(open('pairs.p', 'rb'))
    yolo_maker = YoloMaker(driver)
    #temp = dict()
    #for key in pairs:
    #    temp[key] = pairs[key]
    #    break 
    yolo_maker.make(pairs)
