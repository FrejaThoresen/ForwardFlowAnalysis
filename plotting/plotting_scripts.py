import csv
import matplotlib as mpl
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def load_matplotlib_defaults():
    mpl.rc('font', family='sans-serif', serif='helvet')
    mpl.rc('text', usetex=True)
    # This ensures that no lables are cut of, but the size of the figure changes
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams['text.latex.preamble'] = [
        # It seems that siunitx keeps using serif fonts for numbers!
        r'\usepackage{amsmath}',
        r'\usepackage{amsfonts}',
        r'\usepackage{amssymb}',
        r'\usepackage{helvet}',    # set the normal font here
        r'\usepackage{tgheros}',   # upper case greek letters
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
    ]
#     params = {'legend.fontsize': 'small',
#               'axes.labelsize': 'medium',
#               'axes.titlesize': 'medium',
#               'xtick.labelsize': 'small',
#               'ytick.labelsize': 'small'}
#     mpl.rcParams.update(params)
import time

date = time.strftime("%d-%m-%Y")


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5):

    # x = [1,2,3]
    # y = [1,1,1]
    # xerr = [0.49,0.49,0.49]
    # yerr = [0.05,0.05,0.05]
    # n=0
    # plt.errorbar(x,y,fmt='o')
    # _ = make_error_boxes(plt.gca(), x, y, xerr, yerr, facecolor=colors[n],edgecolor='None', alpha=0.5)  
    # plt.gca().set_xlim(0,4)

    xerror = np.array([xerror,xerror])     
    yerror = np.array([yerror,yerror])     

    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)
        #ye = 1 std
        # yesum = 2 std
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    return


def read_pbpb_run1():
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
                if len(row) == 6:
                    if row[0][0] == '$':
                        nrow = nrow + 1 
                    else:
                        if nrow == 0:
                            alex_v22[-1].append(float(row[1]))
                            alex_v22_err[-1].append(float(row[4]))
                        elif nrow== 1:
                            alex_v24[-1].append(float(row[1]))
                            alex_v24_err[-1].append(float(row[4]))
                        elif nrow== 2:
                            alex_v32[-1].append(float(row[1]))
                            alex_v32_err[-1].append(float(row[4]))
                        elif nrow== 3:
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


