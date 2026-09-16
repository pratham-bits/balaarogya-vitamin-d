from PIL import Image

img = Image.new("RGB", (50, 50), (120, 120, 120))
img.save("data/raw/test_bad_image.jpg")
print("Created test_bad_image.jpg")
