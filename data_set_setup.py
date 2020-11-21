from PIL import Image
import os

def converter(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    idx = 0
    for fileName in files:
        if ".gif" not in fileName: 
            img: Image = Image.open(os.path.join(directory, fileName)).convert("RGB") 
            if (img != None): 
                if (".jpg" in fileName): 
                    img.save(directory + "/" + str(idx) + ".jpg", "jpeg") 
                elif (".jpeg" in fileName): 
                    img.save(directory + "/" + str(idx) + ".jpg", "jpeg") 
                elif (".png" in fileName): 
                    img.save(directory + "/" + str(idx) + ".jpg", "jpeg") 
                elif (".webp" in fileName): 
                    img.save(directory + "/" + str(idx) + ".jpg", "jpeg") 
                
                os.remove(directory + "/" + fileName)
        else:
            os.remove(directory + "/" + fileName) 
        idx += 1


converter("pokemon/4")