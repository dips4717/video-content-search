# scrape youtube videos

import pandas as pd 
from youtubesearchpython import VideosSearch
# Channel Search
from youtubesearchpython import ChannelsSearch
from youtubesearchpython import Hashtag
from youtubesearchpython import PlaylistsSearch
from youtubesearchpython import Suggestions



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

           'Glastonbury Festival Exhibition'
           'Glastonbury Festival Experience',
           'Glastonbury Fun',
           'Glastonbury Festival circus',
           'Glastonbury festival food'         
]
          

# for q in queries:

s1 = []
videosSearch = VideosSearch('Glastonbury festival 2010', limit = 500)
for page in range(10):
    searchresult = videosSearch.result()['result']
    s1.append(searchresult)
    videosSearch.next()
s1 = [x for y in s1 for x in y]        

df = pd.DataFrame(s1)
df.drop_duplicates(subset = 'id')
df.to_csv('temptest2010.csv')
# df = df.append(searchresult)


#%%  ======================================================================
# ------------    Getting playlist information using link      ------------
#=========================================================================
from youtubesearchpython import ResultMode
from youtubesearchpython import Playlist, Video

playlist = Playlist.get('https://www.youtube.com/watch?v=DxsjQ967kV8&list=PLV5BFkI9SS7iKM8v84HwHc1ZSnJTxfnLx')
videos = playlist['videos']  # list of videos  this has 8 metadata fields




#%%  ======================================================================
# ------------    Getting full meta data of the video given url      -
#=========================================================================
videoInfo = Video.getInfo('https://www.youtube.com/watch?v=nrrb3h8GYQU&list=PLV5BFkI9SS7hSXfh1F0UX8m4aDxYzpZe7&ab_channel=GlastonburyOfficial')
print(videoInfo)  # this has 13 metadata feild




#%%  ======================================================================
# ------------   Download all video from a channel ====================   
#=========================================================================
from youtubesearchpython import *


channel_id = "UCAhOAgddivzwMwBB5HHbc0g" # ID for GlastonburyOfficial channel
playlist = Playlist(playlist_from_channel_id(channel_id))

print(f'Videos Retrieved: {len(playlist.videos)}')

while playlist.hasMoreVideos:
    print('Getting more videos...')
    playlist.getNextVideos()
    print(f'Videos Retrieved: {len(playlist.videos)}')

print('Found all the videos.')































