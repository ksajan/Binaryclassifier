

import os
os.getcwd()

collection = "C:/Users/ksaja/Desktop/SiviQuestion(internship)/test/"
for i, filename in enumerate(os.listdir(collection)):
    dst =   str(i) + ".jpg"
    os.rename("C:/Users/ksaja/Desktop/SiviQuestion(internship)/test/" + filename, "C:/Users/ksaja/Desktop/SiviQuestion(internship)/test/" + dst)


"""
import os
def main():
    i=0

    for filename in os.listdir("human dataset"):
        dst = "Human" + str(i) + ".jpg"
        src = "human dataset" + filename 
        dst  = "human dataset" + dst


        os.rename(src, dst)
        i += 1
    if __name__ == '__main__':
        main()
"""

"""
import os 
collection = "C:/Users/ksaja/Downloads/cat-dataset/cats/CAT_06/"

for i, filename in enumerate(os.listdir(collection)):
    if filename.endswith(".jpg.cat"):
        os.remove(os.path.join(collection, filename))

"""