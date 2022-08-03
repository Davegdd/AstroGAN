"""
Place Aladin.jar file in working directory 
as well as the views.eps.aj file created in Aladin 
that contains the desired astronomical observation planes.

Call img_extraction() first to extract and save the images separately 
and then crop_and_concat() to concatenate, crop the watermark 
and save as one file made up of four images.
"""

import subprocess
import os
import numpy as np
import cv2

opticaldirectory = "C:\\Users\\...\\opticaldirectory"
uvdirectory = 'C:\\Users\\...\\uvdirectory'
infradirectory = "C:\\Users\\...\\infradirectory"
opticaldirectory2 = "C:\\Users\\...\\opticaldirectory2"
savedirectory = "C:\\Users\\...\\savedirectory"


def img_extraction():
    p = subprocess.Popen(["java", "-jar", "Aladin.jar"],
                         shell=True,
                         stdin=subprocess.PIPE, encoding = "utf-8"
                           )
    
    p.stdin.write('reset; load C:\\Users\\...\\views.eps.aj\n')
    p.stdin.write('zoom 10arcmin\n')
    
    NGC = 1
    
    while NGC < 7841:
    
        p.stdin.write("NGC"+str(NGC)+"\n")
        
        p.stdin.write('select A1;save ' + uvdirectory + '\\' + "NGC"+str(NGC)+ '.jpg\n')
        p.stdin.write('select B1;save ' + opticaldirectory + '\\' + "NGC"+str(NGC)+ '.jpg\n')
        p.stdin.write('select A2;save ' + infradirectory + '\\' + "NGC"+str(NGC)+'.jpg\n')
        p.stdin.write('select B2;save ' + opticaldirectory2 + '\\' + "NGC"+str(NGC)+'.jpg\n')
        
        NGC += 1
    

def crop_and_concat():
     for opticalfile in os.listdir(opticaldirectory):
        for uvfile in os.listdir(uvdirectory):
            if opticalfile == uvfile:
                for infraredfile in os.listdir(infradirectory):
                    if infraredfile == uvfile:
                        for opticalfile2 in os.listdir(opticaldirectory2):
                            if opticalfile2 == uvfile:
                                optical = cv2.imread(opticaldirectory+"\\"+opticalfile)
                                uv = cv2.imread(uvdirectory+"\\"+uvfile)
                                infra = cv2.imread(infradirectory+"\\"+infraredfile)
                                optical2 = cv2.imread(opticaldirectory2+"\\"+opticalfile2)
                                horizontal_concat = np.concatenate((optical, uv, infra, optical2), axis=1)
                                cv2.imwrite(savedirectory+"\\"+str(uvfile), horizontal_concat[:286, :])
                                break
                 

