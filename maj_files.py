import pandas as pd
import os.path
import shutil
import os
path = '2 Data/Market data Lab'

"""
Update files from a directory, sub-directories:::
from miniscules to majuscules

"""

for directory, _, files in os.walk(path):
    print('##### %s #####' % directory)
    for file in files:
        print(file)
        fileup, extension = os.path.splitext(file)
        # os.rename(file, file.upper())
        # shutil.move(file, filex)

        # Test the type of the file
        if extension == '.csv':
            filex = '{}{}'.format(fileup.upper(), '.csv')
        print(filex)
        pathx = os.path.join(directory, file)
        pathy = os.path.join(directory, filex)
        shutil.move(pathx, pathy)

