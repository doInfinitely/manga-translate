from exhentai_helpers import *

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
    match_consumer = MatchConsumer(driver)
    for i,key in enumerate(title_dict):
        print(key, title_dict[key])
        if key not in matches:
            matches[key] = gallery_matcher.match(key)
            pickle.dump(matches, open('matches.p', 'wb'))
        for x in matches[key]:
            if frozenset({x['url1'],x['url2']}) not in pairs:
                pairs[frozenset({x['url1'],x['url2']})] = match_consumer.consume(x)
                pickle.dump(pairs, open('pairs.p', 'wb'))
