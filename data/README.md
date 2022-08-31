find Dog -name '\*.jpg' -execdir mogrify -resize 224x224! {} +
mogrify -colorspace gray Cat/*.jpg
