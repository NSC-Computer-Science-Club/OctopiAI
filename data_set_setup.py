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
                    img.save(os.path.join(directory, str(idx) + ".jpg"), "jpeg") 
                elif (".jpeg" in fileName): 
                    img.save(os.path.join(directory, str(idx) + ".jpg"), "jpeg") 
                elif (".png" in fileName): 
                    img.save(os.path.join(directory, str(idx) + ".jpg"), "jpeg") 
                elif (".webp" in fileName): 
                    img.save(os.path.join(directory, str(idx) + ".jpg"), "jpeg") 
                
                #os.remove(os.path.join(directory, fileName))
                idx += 1
            else:
                print(fileName + " failed to convert.")
                os.remove(os.path.join(directory, fileName))
        else:
            os.remove(os.path.join(directory, fileName))

converter(os.path.join("pokemon", "4"))