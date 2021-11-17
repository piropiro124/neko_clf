import os
from PIL import Image
import imagehash

userpath = 'yancha_150'  # 検索するパス

def d_hash(img,otherimg):
    hash = imagehash.phash(Image.open(img))
    other_hash = imagehash.phash(Image.open(otherimg))
    return hash-other_hash


def minhash(img,otherimg):
    hash_size = Image.open(img).size
    otherhash_size = Image.open(otherimg).size
    if hash_size<otherhash_size: return 0
    else: return 1



image_files = []
delete_file = []


f = [os.path.join(userpath, path) for path in os.listdir(userpath)]
for i in f:
    if i.endswith('.jpg') or i.endswith('.png'):
        image_files.append(i)

"""
imgs = {}
for img in sorted(image_files):
    hash = imagehash.average_hash(Image.open(img))
    if hash in imgs:
        print('Similar image :', img, imgs[hash])
    else:
        imgs[hash] = img

"""
print(len(image_files))
num = 0
length = len(image_files)
print(range(num,length))



for index in range(num+1,length):
    #進捗状況
    print('num = ',str(num)+'/'+str(length))
    switch = 0
    for next_index in range(num + 2, length):
        if d_hash(image_files[index], image_files[next_index])<10:
            print(image_files[index]+' | vs | '+image_files[next_index])
            #画像サイズの小さい方のパスをdeleteリストに保存
            if minhash(image_files[index], image_files[next_index]) == 0:
                delete_file.append(image_files[index])
            else: delete_file.append(image_files[next_index])
        
            switch = 1
            break
        if switch != 0:break
    num += 1

print(delete_file)
