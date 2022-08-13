import os


src_path=r'D:\AI\colab\proj\v012\syn-shallow'
dst_path=r'D:\AI\colab\proj\v012'
text=''
dataset='train'
jpg_names = [f for f in os.listdir(src_path) if f.endswith('.jpg')]
# print(dir_names)
for jpg_name in jpg_names:
        text = text + f'{dataset}/syn-shallow/{jpg_name} {dataset}/syn-shallow/{jpg_name.split(".")[0]}_label.png\n'

# print(dataset)
print(text)

# write the text to a file, named as 'train_list.txt'
with open(f'{os.path.join(dst_path)}/{dataset}_list.txt', 'w') as f:
    f.write(text)
