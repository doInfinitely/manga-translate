import urllib.request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import selenium
import time
import urllib.request
from PIL import Image, ImageChops
import os
import numpy as np
import cv2
import random
import re 
import pickle
import time
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher
from tqdm import tqdm
from multiprocessing import Pool, Queue
import bisect


def login_exhentai():
    driver = webdriver.Firefox()
    driver.get('https://e-hentai.org/bounce_login.php?b=d&bt=1-1')
    for element in driver.find_elements(By.XPATH, '//input'):
        if element.get_attribute('name') == 'UserName':
            element.send_keys("djochei")
        elif element.get_attribute('name') == 'PassWord':
            element.send_keys("passw0rd")
        elif element.get_attribute('name') == 'ipb_login_submit':
            element.click()
    time.sleep(2)
    driver.get('https://exhentai.org/')
    time.sleep(1)
    return driver

exhentai_button_images = {'https://exhentai.org/img/f.png', 'https://exhentai.org/img/p.png', 'https://exhentai.org/img/n.png', 'https://exhentai.org/img/l.png', 'https://exhentai.org/img/b.png', 'https://exhentai.org/img/mr.gif'}

def get_exhentai_image(driver):
    filename = None
    for element in driver.find_elements(By.XPATH, '//img'):
        if element.get_attribute('src') not in exhentai_button_images:
            filename = 'temp_' + element.get_attribute('src').split('.')[0].split('/')[-1] + '.' +element.get_attribute('src').split('.')[-1]
            while True:
                try:
                    urllib.request.urlretrieve(element.get_attribute('src'),filename)
                    break
                except urllib.error.URLError:
                    print('image error')
                    pass
                except ConnectionResetError:
                    print('image error')
                    pass
            break
    if filename is not None:
        img = Image.open(filename)
        os.remove(filename)
        #img.show()
        return img
    return None

def pil_to_cv2(image, mode="BGR"):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def strip_square_brackets(string):
    output = []
    depth = 0
    i = 0
    while i < len(string):
        x = string[i]
        if x == '[':
            #print('[', i)
            if depth == 0:
                start = i
            depth += 1
        elif x == ']':
            #print(']', i)
            depth -= 1
            if depth == 0:
                output.append(string[start:i+1])
                string = string[:start] + string[i+1:]
                i = start
        i += 1
    return string.strip(), output    

def get_swatches_by_brightness_helper(args):
    diff, width, height, items = args
    output = []
    for x,y in items:
        output.append((np.average(pil_to_cv2(diff.crop((x,y,x+width,y+height)), "RGB")), (x,y,x+width,y+height)))
    return output

def get_swatches_by_brightness(diff, width=64, height=64, processes=1):
    w,h = diff.size
    if processes == 1:
        output = []
        for x in tqdm(range(w-width)):
            for y in range(h-height):
                output.append((np.average(pil_to_cv2(diff.crop((x,y,x+width,y+height)), "RGB")), (x,y,x+width,y+height)))
                #print(output[-1])
        #output.sort(reverse=True)
        return output
    else:
        items = [(x,y) for x in range(w-width) for y in range(h-height)]
        queue = Queue()
        output = []
        with Pool(processes=processes) as pool:
            temp = pool.map(get_swatches_by_brightness_helper,[(diff, width, height, items[int(len(items)/processes*i):min(int(len(items)/processes*(i+1)),len(items))]) for i in range(processes)])
        for x in temp:
            output.extend(x)
        return output

def interval_overlap(int1, int2):
    if int1[1] < int2[0]:
        return None
    if int2[1] < int1[0]:
        return None
    return (max(int1[0],int2[0]), min(int1[1],int2[1]))
    
def overlap(box1, box2):
    x_overlap = interval_overlap((box1[0],box1[2]),(box2[0],box2[2]))
    y_overlap = interval_overlap((box1[1],box1[3]),(box2[1],box2[3]))
    if x_overlap is None or y_overlap is None:
        return 0
    return (x_overlap[1]-x_overlap[0])*(y_overlap[1]-y_overlap[0])

SWATCH_BRIGHTNESS_CUTOFF = 64
def get_manhattan_hull(boxes):
    return (min(x[0] for x in boxes),min(x[1] for x in boxes),max(x[2] for x in boxes),max(x[3] for x in boxes))
def get_components(swatches):
    #print("Swatch 00", swatches[0][0])
    #components = [{x} for b,x in swatches if b > 0.5*swatches[0][0] and b >= SWATCH_BRIGHTNESS_CUTOFF]
    components = [{x} for b,x in swatches if b >= SWATCH_BRIGHTNESS_CUTOFF]
    count = 0
    last = 0
    cache = dict()
    while last < 1000 and len(components) > 1:
        #print(count, len(components))
        i = random.choice(range(len(components)-1))
        x = components[i]
        j = random.choice(range(i+1,len(components)))
        y = components[j]
        #if i >= j:
        #    continue
        if frozenset(x) not in cache:
            hull1 = get_manhattan_hull(x)
            cache[frozenset(x)] = hull1
        else:
            hull1 = cache[frozenset(x)]
        if frozenset(y) not in cache:
            hull2 = get_manhattan_hull(y)
            cache[frozenset(y)] = hull2
        else:
            hull2 = cache[frozenset(y)]
        if overlap(hull1,hull2):
            x.update(y)
            del components[j]
            last = 0
        last += 1
        count += 1
    #print(count, len(components))
    return components

def parse_gallery_header(driver):
    temp = None
    output = dict()
    output['Title'] = driver.find_elements(By.XPATH, '//h1')[0].get_attribute('innerHTML')
    for element in driver.find_elements(By.XPATH, '//td'):
        if element.get_attribute('innerHTML')[-1] == ':':
            temp = element.get_attribute('innerHTML')[:-1]
        elif temp is not None:
            output[temp] = element.get_attribute('innerHTML')
            temp = None
    return output

def get_landing_links(driver):
    output = dict()
    for element in driver.find_elements(By.XPATH, '//a'):
        href = element.get_attribute('href')
        m = re.match(r'https://exhentai.org/g/.*', href)
        if m:
            title = element.find_elements(By.XPATH, './div')[0].get_attribute('innerHTML')
            output[href] = title
    return output

def get_prev_link(driver):
    output = dict()
    for element in driver.find_elements(By.XPATH, '//a'):
        href = element.get_attribute('href')
        m = re.match(r'https://exhentai.org/\?prev=.*', href)
        if m:
            inner = element.get_attribute('innerHTML')
            if inner == '&lt; Prev':
                return href

def get_page_links(driver):
    output = []
    for element in driver.find_elements(By.XPATH, "//div[@class='gdtl']/a"):
        href = element.get_attribute('href')
        if href:
            m = re.match(r'https://exhentai.org/s/.*', href)
            if m:
                output.append(href)
    return output

def get_landing_next_link(driver):
    output = dict()
    for element in driver.find_elements(By.XPATH, '//a'):
        href = element.get_attribute('href')
        if href:
            m = re.match(r'https://exhentai.org/g/.*', href)
            if m:
                inner = element.get_attribute('innerHTML')
                if inner == '&gt;':
                    return href

def get_all_page_links(driver):
    output = []
    output.extend(get_page_links(driver))
    next_landing = get_landing_next_link(driver)
    while next_landing:
        driver.get(next_landing)
        output.extend(get_page_links(driver))
        next_landing = get_landing_next_link(driver)
    return output

def download_all_titles(driver, sleep=10):
    while True:
        try:
            titles = pickle.load(open('titles.p','rb'))
            current = pickle.load(open('titles_checkpoint.p', 'rb'))
            break
        except:
            pass
            #titles = dict()
            #current = 'https://exhentai.org/?prev=1'
    count = 0
    while current:
        driver.get(current)
        temp = get_landing_links(driver)
        print(temp)
        titles.update(temp)
        current = get_prev_link(driver)
        count += 1
        time.sleep(sleep)
        if count % 1 == 0:
            pickle.dump(titles, open('titles.p', 'wb'))
            if current:
                pickle.dump(current, open('titles_checkpoint.p', 'wb'))
    pickle.dump(titles, open('titles.p', 'wb'))
    if current:
        pickle.dump(current, open('titles_checkpoint.p', 'wb'))

class Gallery:
    def __init__(self, url=None, metadata=None, images=None):
        self.url = url
        self.metadata = metadata
        self.images = images
class GalleryDownloader:
    def __init__(self, driver):
        self.driver = driver
        self.cache = dict()
        self.id = 0
        self.zfill = 16
        self.cache = pickle.load(open('gallery_lookup.p', 'rb'))
        self.id = max([int(self.cache[key]) for key in self.cache])+1
    def download(self, url, write=True, sleep=0, retry=5):
        if url in self.cache:
            return pickle.load(open('galleries/{0}.p'.format(self.cache[url]), 'rb'))
        images = []
        self.driver.get(url)
        metadata = parse_gallery_header(self.driver)
        for x in get_all_page_links(self.driver):
            for i in range(min(1,retry)):
                try:
                    self.driver.get(x)
                    time.sleep(sleep)
                    images.append(get_exhentai_image(self.driver))
                    break
                except selenium.common.exceptions.TimeoutException:
                    print('TimeoutException', x)
        gallery = Gallery(url, metadata, images)
        self.cache[url] = str(self.id).zfill(self.zfill)
        self.id += 1
        pickle.dump(gallery, open('galleries/{0}.p'.format(self.cache[url]), 'wb'))
        pickle.dump(self.cache, open('gallery_lookup.p','wb'))
        #if write:
        #    pickle.dump(self.cache, open('gallery_downloader_cache.p', 'wb'))
        return gallery
    def dump(self):
        pass
        #pickle.dump(self.cache, open('gallery_downloader_cache.p', 'wb'))

# used to split the gallery downloader cache once it got to large
class GallerySharder:
    def __init__(self):
        self.zfill = 16
        self.cache = pickle.load(open('gallery_downloader_cache.p', 'rb'))
        self.lookup = {key:str(i).zfill(self.zfill) for i,key in enumerate(self.cache)}
    def shard(self):
        for i,key in enumerate(self.lookup):
            print(key, i, len(self.lookup))
            with open('galleries/{0}.p'.format(self.lookup[key]), 'wb') as f:
                pickle.dump(self.cache[key], f)
        pickle.dump(self.lookup, open('gallery_lookup.p', 'wb'))

def choose_all(items, n):
    if n == 0:
        return set()
    if n == 1:
        return {frozenset({x}) for x in items}
    output = set()
    for x in choose_all(items, n-1):
        for y in items:
            if y not in x:
                output.add(frozenset(x | {y}))
    return output

class GalleryComparator:
    def __init__(self, driver):
        self.driver = driver
        self.gallery_downloader = GalleryDownloader(self.driver)
    def compare(self, url1, url2, sleep=0):
        gallery1 = self.gallery_downloader.download(url1, sleep=sleep)
        gallery2 = self.gallery_downloader.download(url2, sleep=sleep)
        diffs = dict()
        if len(gallery1.images) == len(gallery2.images): #if galleries are the same size try the default mapping
            for i,x in enumerate(gallery1.images):
                x = x.convert('RGB')
                y = gallery2.images[i]
                y = y.resize(x.size)
                y = y.convert('RGB')
                diff = ImageChops.difference(x, y)
                diffs[(i,i)] = np.average(pil_to_cv2(diff, "RGB"))
                #print(i,i,diffs[(i,i)])
            return sum(diffs[(i,i)] for i in range(len(gallery1.images)))/len(gallery1.images), [(i,i) for i in range(len(gallery1.images))]
        diff_sum = 0
        mapping = []
        mini = (float("inf"), None)
        if len(gallery1.images) < len(gallery2.images):
            lendiff = len(gallery2.images) - len(gallery1.images)
            for i,x in tqdm(enumerate(gallery1.images)):
                x = x.convert('RGB')
                for j in range(lendiff+1):
                    j += i
                    y = gallery2.images[j]
                    y = y.resize(x.size)
                    y = y.convert('RGB')
                    diff = ImageChops.difference(x, y)
                    diffs[(i,j)] = np.average(pil_to_cv2(diff, "RGB"))
                    #print(i,j,diffs[(i,j)])
            for exclude in tqdm(choose_all([i for i in range(len(gallery2.images))], lendiff)):
                indices = [i for i in range(len(gallery2.images)) if i not in exclude]
                diff = sum(diffs[(i,indices[i])] for i in range(len(gallery1.images)))/len(gallery1.images)
                #print(exclude, diff)
                if diff < mini[0]:
                    mini = (diff, [(i,indices[i]) for i in range(len(gallery1.images))]) 
        else:
            lendiff = len(gallery1.images) - len(gallery2.images)
            for i,x in tqdm(enumerate(gallery2.images)):
                x = x.convert('RGB')
                for j in range(lendiff+1):
                    j += i
                    y = gallery1.images[j]
                    y = y.resize(x.size)
                    y = y.convert('RGB')
                    diff = ImageChops.difference(x, y)
                    diffs[(j,i)] = np.average(pil_to_cv2(diff, "RGB"))
                    #print(j,i,diffs[(j,i)])
            for exclude in tqdm(choose_all([i for i in range(len(gallery1.images))], lendiff)):
                indices = [i for i in range(len(gallery1.images)) if i not in exclude]
                diff = sum(diffs[(indices[i],i)] for i in range(len(gallery2.images)))/len(gallery2.images)
                #print(exclude, diff)
                if diff < mini[0]:
                    mini = (diff, [(indices[i],i) for i in range(len(gallery2.images))])
        return mini
                
def title_similarity(url, title_dict, preprocess=lambda x:x):
    #print(title_dict[url])
    title1 = preprocess(title_dict[url])
    output = dict()
    for key in title_dict:
        title2 = preprocess(title_dict[key])
        output[key] = len(SequenceMatcher(None,title1,title2).find_longest_match(0,len(title1),0,len(title2)))/max(min(len(title1),len(title2)),1)
    return output

def title_distance(url, title_dict, preprocess=lambda x:x):
    print(title_dict[url])
    return {key:levenshtein_distance(preprocess(title_dict[url]),preprocess(title_dict[key])) for key in title_dict}            

def all_distance(title_dict, preprocess=lambda x:x):
    return {key:title_distance(key, title_dict, preprocess) for key in title_dict}


LEV_CUTOFF = 5
#SUBSTRING_CUTOFF = 0.9
GALLERY_LEN_DIFF_CUTOFF = 4
GALLERY_DIFF_CUTTOFF = 64
class GalleryMatcher:
    def __init__(self, driver, title_dict):
        self.driver = driver
        self.title_dict = title_dict
        self.gallery_comparator = GalleryComparator(self.driver)
    def match(self, url):
        self.driver.get(url)
        metadata1 = parse_gallery_header(self.driver)
        gallery1len = len(get_all_page_links(self.driver))
        distances = title_distance(url, self.title_dict, lambda x:strip_square_brackets(x)[0].lower())
        #similarities = title_similarity(url, self.title_dict, lambda x:strip_square_brackets(x)[0].lower())
        output = []
        title_matches = [(distances[key], key) for key in distances if distances[key] <= LEV_CUTOFF  and url != key]
        #print('title matches: ', len(title_matches))
        title_matches = sorted(title_matches)[:min(len(title_matches),20)]
        for d,x in title_matches:
            self.driver.get(x)
            metadata2 = parse_gallery_header(self.driver)
            gallery2len = len(get_all_page_links(self.driver))
            if 'Language' in metadata1 and 'Language' in metadata2 and metadata1['Language'] != metadata2['Language']:
                language1 = metadata1['Language'].split()[0]
                language2 = metadata2['Language'].split()[0]
                #print(gallery1len, gallery2len)
                if abs(gallery1len-gallery2len) <= GALLERY_LEN_DIFF_CUTOFF:
                    diff, mapping = self.gallery_comparator.compare(url,x)
                    print(url, x, language1, language2, diff, mapping)
                    if diff <= GALLERY_DIFF_CUTTOFF:
                        output.append({'url1':url,'url2':x,'lang1':language1,'lang2':language2,'diff':diff,'map':mapping})
        return output

SWATCH_WIDTH = 48
SWATCH_HEIGHT = 48
class MatchConsumer:
    def __init__(self, driver):
        self.driver = driver
        self.gallery_downloader = GalleryDownloader(self.driver)
    def consume(self, match, sleep=0):
        gallery1 = self.gallery_downloader.download(match['url1'], sleep=sleep)
        gallery2 = self.gallery_downloader.download(match['url2'], sleep=sleep)
        output = dict()
        output.update(match)
        output['boxes'] = []
        print(len(match['map']))
        for l,(i,j) in tqdm(enumerate(match['map'])):
            img1 = gallery1.images[i]
            img2 = gallery2.images[j]
            img2 = img2.resize(img1.size)
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')
            diff = ImageChops.difference(img1, img2)
            #print((i,j),l,len(match['map']))
            swatches = get_swatches_by_brightness(diff, SWATCH_WIDTH, SWATCH_HEIGHT, processes=10)
            components = get_components(swatches)
            boxes = []
            if len(components) > 100:
                components = []
            for k,x in enumerate(components):
                hull = get_manhattan_hull(x)
                if k < 10:
                    pass
                    #diff.crop(hull).show()
                boxes.append(hull)
            #print([np.average(pil_to_cv2(diff.crop(x))) for x in boxes])
            #input()
            #output[match['lang1']].append(img1)
            #output[match['lang2']].append(img2)
            #output['Diff'].append(diff)
            output['boxes'].append(boxes)
        return output
            
            

'''
if __name__ == '__main__':
    driver = login_exhentai()
    driver.get('https://exhentai.org/s/160f3025e7/2058493-1')
    time.sleep(1)
    img1 = get_exhentai_image(driver)
    driver.get('https://exhentai.org/s/0b218fc5bc/2658325-1')
    time.sleep(1)
    img2 = get_exhentai_image(driver)
    img2 = img2.resize(img1.size)
    diff = ImageChops.difference(img1, img2)
  
    # showing the difference
    diff.show()
    print(np.average(pil_to_cv2(diff, "RGB")))
'''

'''
if __name__ == '__main__':
    title = "[Kezukaya (Various)] Welcome to Kemmoner's Rift!! Tri (League of Legends) [Digital]"
    print(strip_square_brackets(title))
'''


if __name__ == '__main__':
    driver = login_exhentai()
    #driver.get('https://exhentai.org/s/9631ef149e/2779613-4')
    driver.get('https://exhentai.org/s/b3b443afc0/136-5')
    time.sleep(1)
    img1 = get_exhentai_image(driver)
    #driver.get('https://exhentai.org/s/5fafbf8f85/1338586-5')
    driver.get('https://exhentai.org/s/d80eec1748/657428-5')
    time.sleep(1)
    img1 = img1.convert('RGB')
    img2 = get_exhentai_image(driver)
    img2 = img2.resize(img1.size)
    img2 = img2.convert('RGB')
    #img2 = img2.convert(img1.mode)
    diff = ImageChops.difference(img1, img2)
  
    #img1.show()
    #img2.show()
    #print(img1.mode)
    # showing the difference
    diff.show()
    print(np.average(pil_to_cv2(diff, "RGB")))
    t0 = time.process_time()
    swatches = get_swatches_by_brightness(diff, processes=10)
    t1 = time.process_time()
    print(t1-t0)
    swatches.sort(reverse=True)
    diff.crop(swatches[0][1]).show()
    components = get_components(swatches)
    for x in components[:min(20, len(components))]:
        diff.crop(get_manhattan_hull(x)).show()

'''
if __name__ == '__main__':
    driver = login_exhentai()
    #driver.get('https://exhentai.org/g/1338586/bd10e2a9d0/')
    #driver.get('https://exhentai.org/g/2779765/ff7d63f98d/')
    #print(parse_gallery_header(driver))
    #get_page_links(driver)
    #driver.get('https://exhentai.org/g/2780392/81e2f08de4/')
    driver.get('https://exhentai.org/g/134/7ebb3b8534/')
    print(len(get_all_page_links(driver)))
'''
'''
if __name__ == '__main__':
    driver = login_exhentai()
    driver.get('https://exhentai.org/?prev=1')
    print(get_landing_links(driver))
    print(get_prev_link(driver))
'''

'''
if __name__ == '__main__':
    driver = login_exhentai()
    download_all_titles(driver)
'''

'''
if __name__ == '__main__':
    driver = login_exhentai()
    gallery_downloader = GalleryDownloader(driver)
    for x in gallery_downloader.download('https://exhentai.org/g/136044/a8d6e3de4b/').images:
        x.show()
'''

'''
if __name__ == '__main__':
    #print(choose_all([1,2,3,4,5], 2))
    driver = login_exhentai()
    gallery_comparator = GalleryComparator(driver)
    url1 = 'https://exhentai.org/g/2058493/b67e02048b/'
    url2 = 'https://exhentai.org/g/2658325/faef23d993/'
    print(gallery_comparator.compare(url1, url2))
    print(gallery_comparator.compare(url2, url1))
    #url1 = 'https://exhentai.org/g/1018410/125e2a92d4/'
    #url2 = 'https://exhentai.org/g/1009810/4dcc8e3abb/'
    #print(gallery_comparator.compare(url1, url2))
'''
'''
if __name__ == '__main__':
    title_dict = pickle.load(open('titles.p','rb'))
    driver = login_exhentai()
    gallery_downloader = GalleryDownloader(driver)
    for url in title_dict:
        break
    print(title_dict[url], url)
    driver.get(url)
    metadata1 = parse_gallery_header(driver)
    distances = title_distance(url, title_dict, lambda x:strip_square_brackets(x)[0].lower())
    gallery_comparator = GalleryComparator(driver)
    for d,x in sorted([(distances[key], key) for key in distances if key != url]):
        driver.get(x)
        metadata2 = parse_gallery_header(driver)
        print(metadata['Language'].split()[0])
        print(gallery_comparator.compare(url,x))
        break
'''

'''
if __name__ == '__main__':
    title_dict = pickle.load(open('titles.p','rb'))
    #distances = all_distance(title_dict, lambda x:strip_square_brackets(x)[0].lower())
    #for key1 in distances:
    #    print(title_dict[key1])
    #    print([(d,title_dict[x]) for d,x in [(distances[key1][key2], key2) for key2 in distances[key1] if distances[key1][key2] <= LEV_CUTOFF]])
    for key1 in title_dict:
        #print(title_dict[key1])
        distances = title_distance(key1, title_dict, lambda x:strip_square_brackets(x)[0].lower())
        print([(d,title_dict[x]) for d,x in [(distances[key2], key2) for key2 in distances if distances[key2] <= LEV_CUTOFF and key1 != key2]])
'''
'''
if __name__ == '__main__':
    driver = login_exhentai()
    title_dict = pickle.load(open('titles.p','rb'))
    gallery_matcher = GalleryMatcher(driver, title_dict)
    for i,key in enumerate(title_dict):
        if i < 7:
            continue
        matches = gallery_matcher.match(key)
        match_consumer = MatchConsumer(driver)
        for x in matches:
            match_consumer.consume(x)
'''
'''
if __name__ == '__main__':
    driver = login_exhentai()
    title_dict = pickle.load(open('titles.p','rb'))
    gallery_matcher = GalleryMatcher(driver, title_dict)
    try:
        matches = pickle.load(open('matches.p', 'rb'))
    except:
        matches = dict()
    try:
        pairs = pickle.load(open('pairs.p', 'rb'))
    except:
        pairs = dict()
    for i,key in enumerate(title_dict):
        print(key, title_dict[key])
        if key not in matches:
            matches[key] = gallery_matcher.match(key)
            pickle.dump(matches, open('matches.p', 'wb'))
        match_consumer = MatchConsumer(driver)
        for x in matches[key]:
            if frozenset({x['url1'],x['url2']}) not in pairs:
                pairs[frozenset({x['url1'],x['url2']})] = match_consumer.consume(x)
                pickle.dump(pairs, open('pairs.p', 'wb'))
'''
