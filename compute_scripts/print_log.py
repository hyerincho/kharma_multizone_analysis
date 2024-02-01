import os
import time
import glob
import pdb

def print_log():
    out_files = sorted(glob.glob("./logs/*out"))
    for fn in out_files:
        out_string = str(time.ctime(os.path.getctime(fn))) + ", " + fn.replace(".out","").replace("./logs/","") + ": "
        #print(fn,time.ctime(os.path.getctime(fn)))
        with open(fn, 'r') as input:
           for line in input:
               if 'saved to' in line:
                   dat_dir = line.replace("Output saved to ../data_products/","").replace("_profiles_all2.pkl.\n"," ")
                   out_string += dat_dir
        print(out_string)

    
if __name__ == "__main__":
    print_log()
