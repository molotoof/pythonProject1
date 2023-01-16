import os

img_dir = "C:\\Users\\rusanovma\\Desktop\\ЮНИИИТ\\2022\\ХАКАТОН\\картинки\\Ягоды\\Черника - возможно"
start = 1

img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, name))]

for count, current_file in enumerate(img_paths, start=start):
    dir_name = os.path.dirname(current_file)
    pathname, extension = os.path.splitext(current_file)
    basename = os.path.basename(pathname)
    new_name = os.path.join(dir_name, f'{count}{extension}')
    os.rename(current_file, new_name)