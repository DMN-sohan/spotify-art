################################################################################
# Author: Orion Crocker
# Filename: images.py
# Date: 02/06/20
# 
# Get Spotify Album Art
# Retrieves album art from Spotify's spotify
################################################################################

import api
import os
import requests
from zipfile import ZipFile


def rename(name):
  return name.lower().replace(' ','_').replace('/','_').replace('[','(').replace(']',')').replace(':','').replace('?','').replace('*','').replace('#','')


def zip_images(directory):
  zip_this = ZipFile(directory + '.zip', 'w')
  os.chdir(directory)
  for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
      zip_this.write(file)
  zip_this.close()


def url_to_uri(uri, typeof):
  offset = uri.find(typeof)
  return 'spotify:' + uri[offset:].replace('/',':')


def get_images(url, directory=None, verbose=False, zip_this=False):
  typeof = ''
  results = ''
  if 'artist' in url:
    typeof = 'artist'
    results = api.get_artist(url)
  elif 'playlist' in url:
    typeof = 'playlist'
    results = api.get_playlist(url)
  if results == '':
    print("No results found, check URL and try again.")
    exit(1)
  else:
    print(typeof + " found, downloading...")

  if typeof == 'artist':
    name = results['items'][0]['artists'][0]['name']
    results = results['items']
  elif typeof == 'playlist':
    name = results['name']
    results = results['tracks']['items']

  if verbose:
    print('Name: ' + name + '\nType: ' + typeof)
    
  if directory:
    directory = directory + '/' + rename(name)
  else:
    directory = 'results/' + rename(name)

  if not os.path.exists(directory):
    os.makedirs(directory)

  count = 0
  pics = []

  for track in results:
    if typeof == 'artist':
      url = track['images'][0]['url']
      name = rename(track['name'])
    elif typeof == 'playlist':
      url = track['track']['album']['images'][0]['url']
      name = rename(track['track']['album']['name'])
      
    path = directory + '/' + name + '.jpeg'
    if os.path.exists(path):
      continue
    
    
    pic = requests.get(url, allow_redirects=True)
    
    if verbose:
      print(path)

    try:
      open(path, 'wb').write(pic.content)
      print(f"Downloaded {name}")
    except:
      print(f"Error in downloading {name}")
      continue
    
    count += 1
    pics.append(pic)

  print(str(len(pics)) + " saved to " + directory)

  if zip_this:
    zip_images(directory)

  return directory
