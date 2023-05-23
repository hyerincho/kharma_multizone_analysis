import sys
import os

def folderToMovie(inFolder, outMovie, framerate=8, patternMatch="%*.png"):
    if inFolder[-1] != '/':
        inFolder += '/'
    os.system("ffmpeg -y -framerate "+str(int(framerate))+" -i "+inFolder+patternMatch+" -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "+outMovie)

if __name__ == '__main__':
    inFolder = sys.argv[1]
    outMovie = sys.argv[2]
    folderToMovie(inFolder, outMovie)