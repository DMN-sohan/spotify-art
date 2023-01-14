##############################################################################
# Author: Orion Crocker
# Filename: main.py
# Date: 01/13/20
# 
# Spotify Collage Creator
#   Automatically downloads all album art from Spotify playlist and assembles a
#     collage
################################################################################

import argparse
import images, collage
import cv2
from cv2 import dnn_superres
import os
from PIL import Image

def main():

  parse = argparse.ArgumentParser(description='Spotify image gatherer and creator of collages')
  parse.add_argument('url', nargs='?')
  parse.add_argument('-c', '--collage', action='count', default=0, help='Create a collage out of images gathered from "playlist" or "artist" argument.')
  parse.add_argument('-d', '--directory', dest='directory', type=str, help='Specify the a target directory to output results')
  parse.add_argument('-v', '--verbose', action='count', default=0, help='See the program working instead of just believing that it is working')
  parse.add_argument('-z', '--zip', action='count', default=0, help='Output the directory into a zip file')

  args = parse.parse_args()
  if args.url is None:
    print('Spotify URL is required.')
    exit(1)

  c = args.collage
  d = args.directory
  v = args.verbose
  z = args.zip
  
  directory = images.get_images(args.url, directory=args.directory, verbose=args.verbose, zip_this=args.zip)

  if c:
    collage.make_collage(directory=directory, verbose=args.verbose)

def upscale_dir():

    for i in [[ os.path.join(j,i) for i in os.listdir(j)] for j in [os.path.join(os.getcwd(),"results",j) for j in os.listdir(os.path.join(os.getcwd(),"results"))]][0]:

        png_path = i.replace("jpeg","png")
        
        img = Image.open(i)

        img.save(png_path)
        sr = dnn_superres.DnnSuperResImpl_create()
        
        # Read image
        image = cv2.imread(png_path)
        
        # Read the desired model
        path = "ESPCN_x4.pb"
        sr.readModel(path)
        
        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("espcn", 4)
        
        # Upscale the image
        result = sr.upsample(image)
        os.remove(i)
        os.remove(png_path)
        i = i.replace("jpeg","jpg")
        
        # Save the image
        cv2.imwrite(i, result)

if __name__ == '__main__':
  main()
  print("\n\nUpscaling images, please hold")
  upscale_dir()
