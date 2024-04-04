ffmpeg -r 20 -f image2 -s 1000x500 -i plots$1/plot%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p out$1.mp4
