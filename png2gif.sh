ffmpeg -r 20 -f image2 -s 1000x500 -i plots$1/plot%04d.png -vf palettegen palette.png
ffmpeg -r 20 -f image2 -s 1000x500 -i plots$1/plot%04d.png -i palette.png -lavfi paletteuse out$1.gif