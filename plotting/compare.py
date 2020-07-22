import csv
import numpy as np

def readAlex():
    alex_vn = []
    alex_vn_e = []
    alex_v22 = []
    alex_v22_err = []
    alex_v24 = []
    alex_v24_err = []
    alex_v32 = []
    alex_v32_err = []
    alex_v42 = []
    alex_v42_err = []

    for c in range(1,10):
        with open('/home/thoresen/Documents/PhD/hepdata/HEPData-ins1456145-v1-csv/Table'+str(c)+'.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            alex_v22.append([])
            alex_v22_err.append([])
            alex_v24.append([])
            alex_v24_err.append([])
            alex_v32.append([])
            alex_v32_err.append([])
            alex_v42.append([])
            alex_v42_err.append([])
            nrow = -1
            for row in spamreader:
                if(len(row) == 6):
                    if (row[0][0] == '$'):
                        nrow = nrow + 1 
                    else:
                        if (nrow == 0):
                            alex_v22[-1].append(float(row[1]))
                            alex_v22_err[-1].append(float(row[4]))
                        elif (nrow== 1):
                            alex_v24[-1].append(float(row[1]))
                            alex_v24_err[-1].append(float(row[4]))
                        elif (nrow== 2):
                            alex_v32[-1].append(float(row[1]))
                            alex_v32_err[-1].append(float(row[4]))
                        elif (nrow== 3):
                            alex_v42[-1].append(float(row[1]))
                            alex_v42_err[-1].append(float(row[4]))

    alex_vn.append(np.array(alex_v22))
    alex_vn.append(np.array(alex_v32))
    alex_vn.append(np.array(alex_v42))
    alex_vn.append(np.array(alex_v24))

    alex_vn_e.append(np.array(alex_v22_err))
    alex_vn_e.append(np.array(alex_v32_err))
    alex_vn_e.append(np.array(alex_v42_err))
    alex_vn_e.append(np.array(alex_v24_err))

    x_alex = [-3.25,-2.75,-2.25,-1.75,-1.25,-0.75,-0.25,0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75]

    return np.array(x_alex), alex_vn, alex_vn_e
