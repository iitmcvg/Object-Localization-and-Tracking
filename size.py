import glob
from PIL import Image

from resizeimage import resizeimage

counter = 1
for imga in glob.glob("*.jpg"):
	img = Image.open(imga)
	if (img.size[0] > 256 and img.size[1] > 256):
		print img.size
		cover = resizeimage.resize_cover(img,[256,256])
		cover.save(str(counter)+".jpg",img.format)
		counter = counter+1