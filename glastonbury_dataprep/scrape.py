import pandas as pd 
from youtubesearchpython import VideosSearch, Playlist, Video
from youtubesearchpython import *

#%% From Search Query
queries = ['Glastonbury festival',
            'Glastonbury festival 2019',
            'Glastonbury featival 2018',
            'Glastonbury featival 2017',
            'Glastonbury featival 2016',
            'Glastonbury featival 2015',
            'Glastonbury featival 2014',
            'Glastonbury featival 2013',
            'Glastonbury featival 2012',
            'Glastonbury featival 2011',
            'Glastonbury featival 2010',

            'Glastonbury festival vlog',
            'Glastonbury festival vlog 2019',
            'Glastonbury featival vlog 2018',
            'Glastonbury featival vlog 2017',
            'Glastonbury featival vlog 2016',
            'Glastonbury featival vlog 2015',
            'Glastonbury featival vlog 2014',
            'Glastonbury featival vlog 2013',
            'Glastonbury featival vlog 2012',
            'Glastonbury featival vlog 2011',
            'Glastonbury featival vlog 2010',
           
            'Glastonbury festival highlights',  
            'Glastonbury festival highlights 2019',
            'Glastonbury festival highlights 2018',
            'Glastonbury festival highlights 2017',
            'Glastonbury festival highlights 2016',
            'Glastonbury festival highlights 2015', 
            'Glastonbury festival highlights 2014',
            'Glastonbury festival highlights 2013',
            'Glastonbury festival highlights 2012',
            'Glastonbury festival highlights 2011',
            'Glastonbury festival highlights 2010',
        
            'Best of Glastonbury festival',  
            'Best of Glastonbury festival  2019',
            'Best of Glastonbury festival  2018',
            'Best of Glastonbury festival  2017',
            'Best of Glastonbury festival  2016',
            'Best of Glastonbury festival  2015', 
            'Best of Glastonbury festival  2014',
            'Best of Glastonbury festival  2013',
            'Best of Glastonbury festival  2012',
            'Best of Glastonbury festival  2011',
            'Best of Glastonbury festival  2010',
           
            'Glastonbury Festival BBC',
            'Glastonbury Festival BBC Higlights',
            'Best of BBC Glastonbury Festival',
            'Glastonbury Festival BBC Music',
            'Glastonbury Festival experience',
            'Glastonbury Festival Exhibition',
            'Glastonbury Fun',
            'Glastonbury Festival circus',
            'Glastonbury festival food'         
]

s1 = []
for ii, query in enumerate(queries):
    if ii%10==0:
        print(f'Completed {ii} queries')
    videosSearch = VideosSearch(query, limit = 500)
    for page in range(10):
        searchresult = videosSearch.result()['result']
        s1.append(searchresult)
        videosSearch.next()

s1 = [x for y in s1 for x in y]        

#%% From GlastonburyOfficial Channel

channel_id = "UCAhOAgddivzwMwBB5HHbc0g" # ID for GlastonburyOfficial channel
playlist = Playlist(playlist_from_channel_id(channel_id))
while playlist.hasMoreVideos:
    playlist.getNextVideos()
    print(f'Videos Retrieved: {len(playlist.videos)}')

s2 = playlist.videos
result = s1+s2

print(f'\nFound len(result) videos in total!! \n')

# Get detailed metadata using each video query.
metadata = []
invalid_urls = []
print('Extracting detailed metadata now!!')
for ii, x in enumerate(result):
    link = x['link']
    try:
        videoInfo = Video.getInfo(link)
        metadata.append(videoInfo)
    except:
        
        invalid_urls.append(link)
    

    if ii%100==0:
        print(f'Extraction: [{ii}/{len(result)}]')

df = pd.DataFrame(metadata)
df.to_csv('metadata.csv')

with open('invalid_links.txt', 'w') as f:
    f.writelines("\n".join(invalid_urls))


#%%  Removing duplicates
if not 'df' in locals():
    df = pd.read_csv('metadata.csv')
    
df_uni = df.drop_duplicates(subset='link')
df_uni.to_csv('meta.csv')




