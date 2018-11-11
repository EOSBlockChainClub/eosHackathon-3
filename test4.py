import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "faces")
list = os.listdir(image_dir) # dir is your directory path
print(list[1:])