import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from helpers.constants import *
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


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


def makeFlowCentPlot(vn, sigma,sigma_x, xlist, noCent, nrows, ncols,yaxis,nbin,ymin,ymax,ydiff):
    axis = [None]*noCent
    #xlist = np.linspace(-4.0, 6, 24)
    
    if (noCent == 9):
        fig, ((axis[0],axis[1],axis[2]),(axis[3],axis[4],axis[5]),(axis[6],axis[7],axis[8])) = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True);
        axis[6].set_xlabel("$\eta$", fontsize=20)
        axis[7].set_xlabel("$\eta$", fontsize=20)
        axis[8].set_xlabel("$\eta$", fontsize=20)
        axis[0].set_ylabel(yaxis, fontsize=20)
        axis[3].set_ylabel(yaxis, fontsize=20)
        axis[6].set_ylabel(yaxis, fontsize=20)
        fig.text(0.17,0.92,'0-10 %',fontsize=16) 

        fig.text(0.43,0.92,'10-20 %',fontsize=16)
        fig.text(0.70,0.92,'20-30 %',fontsize=16)
        fig.text(0.17,0.66,'30-40 %',fontsize=16)
        fig.text(0.43,0.66,'40-50 %',fontsize=16)
        fig.text(0.70,0.66,'50-60 %',fontsize=16)
        fig.text(0.17,0.40,'60-70 %',fontsize=16)
        fig.text(0.43,0.40,'70-80 %',fontsize=16)
        fig.text(0.70,0.40,'80-90 %',fontsize=16)
        fig.set_figheight(10.5)
        fig.set_figwidth(10)
    if (noCent == 6):

        fig, ((axis[0],axis[1],axis[2]),(axis[3],axis[4],axis[5])) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True);
        axis[3].set_xlabel("$\eta$", fontsize=16)
        axis[4].set_xlabel("$\eta$", fontsize=16)
        axis[5].set_xlabel("$\eta$", fontsize=16)
        axis[0].set_ylabel(yaxis, fontsize=16)
        axis[3].set_ylabel(yaxis, fontsize=16)
        fig.text(0.18,0.91,'0-10 %',fontsize=16)
        fig.text(0.44,0.91,'10-20 %',fontsize=16)
        fig.text(0.71,0.91,'20-30 %',fontsize=16)
        fig.text(0.18,0.51,'30-40 %',fontsize=16)
        fig.text(0.44,0.51,'40-50 %',fontsize=16)
        fig.text(0.71,0.51,'50-60 %',fontsize=16)

        fig.set_figheight(5)
        fig.set_figwidth(10)
        
    if (noCent == 3):
        fig, ((axis[0],axis[1],axis[2])) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True);
        fig.set_figheight(4)
        fig.set_figwidth(15)
    if (noCent == 1):
        fig, axis[0] = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True);
        

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=0.001)



    x = 5

    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots

    for n in range(0,nbin):
        for c in range(0,noCent):
            if (c == 0):
                
                axis[0].errorbar(xlist,vn[n][:,c],yerr=sigma[n][:,c],xerr=sigma_x, label='$v_' + str(n+2)+'$',fmt='o',markersize=4.5)
                axis[0].legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
                
                axis[c].xaxis.set_major_locator(MultipleLocator(1))
                axis[c].xaxis.set_minor_locator(AutoMinorLocator())

                axis[c].tick_params(which='both',direction='in')

            else:

                axis[c].errorbar(xlist,vn[n][:,c],yerr=sigma[n][:,c],xerr=sigma_x,fmt='o',markersize=4.5)
                axis[c].yaxis.set_major_locator(MultipleLocator(ydiff))
                axis[c].yaxis.set_minor_locator(AutoMinorLocator())
                
                axis[c].xaxis.set_major_locator(MultipleLocator(2))
                axis[c].xaxis.set_minor_locator(AutoMinorLocator())
                
                axis[c].tick_params(which='both',direction='in') 
                axis[c].set_xlim([-3.5,5.0])
                axis[c].set_ylim([ymin,ymax])

    handles, labels = axis[0].get_legend_handles_labels()
    axis[0].legend(handles, labels, ncol=2, loc='upper right',columnspacing=-0.1,handletextpad=0.1)
    fig.savefig('hej.pdf')
    return


def makeSingleCentPlot(vn, sigma,sigma_x, xlist, noCent, yaxis,nbin):
    fig = plt.figure(figsize=(6, 5))

    axis = plt.axes()
    axis.set_xlabel("$\eta$", fontsize=17)
    axis.set_ylabel(yaxis, fontsize=17)

    for n in range(0,nbin):
        axis.errorbar(xlist,vn[n][:,noCent],yerr=sigma[n][:,noCent],xerr=sigma_x, label='$v_' + str(n+2)+'$',fmt='o',markersize=4.5)
        axis.legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
                
        axis.xaxis.set_major_locator(MultipleLocator(1))
        axis.xaxis.set_minor_locator(AutoMinorLocator())

        axis.tick_params(which='both',direction='in',top='on',right='on')
                
        axis.xaxis.set_major_locator(MultipleLocator(2))
        axis.xaxis.set_minor_locator(AutoMinorLocator())
                
        axis.tick_params(which='both',direction='in',labelsize=14)
        axis.set_xlim([-3.5,5.0])

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, ncol=2, loc='upper right',columnspacing=-0.1,handletextpad=0.1,fontsize=17)
    return


def makeSingleCentPlot(vn, sigma,sigma_x, xlist, noCent, yaxis,nbin):
    fig = plt.figure(figsize=(6, 5))
    #mystring = str(noCent*10) +'-'+ str(noCent*10 + 10) + ' %'
    #fig.text(0.18,0.91,mystring,fontsize=20) #        fig.text(0.15,0.85,'0-10 %',fontsize=16)
    #fig.set_figheight(5.5)
    #fig.set_figwidth(8)

    axis = plt.axes()
    axis.set_xlabel("$\eta$", fontsize=17)
    axis.set_ylabel(yaxis, fontsize=17)

    for n in range(0,nbin):
        axis.errorbar(xlist,vn[n][:,noCent],yerr=sigma[n][:,noCent],xerr=sigma_x, label='$v_' + str(n+2)+'$',fmt='o',markersize=4.5)
        axis.legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
                
        axis.xaxis.set_major_locator(MultipleLocator(1))
        axis.xaxis.set_minor_locator(AutoMinorLocator())

        axis.tick_params(which='both',direction='in',top='on',right='on')
                
        axis.xaxis.set_major_locator(MultipleLocator(2))
        axis.xaxis.set_minor_locator(AutoMinorLocator())
                
        axis.tick_params(which='both',direction='in',labelsize=14)
        axis.set_xlim([-3.5,5.0])

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, ncol=2, loc='upper right',columnspacing=-0.1,handletextpad=0.1,fontsize=17)
    return


def compare_v4ref(v4 = [0,0,0,0,0,0], v4_e = [0,0,0,0,0,0]):

    noCent = len(v4)
    xlist = makelist(0,100,10)-10

    x_err = np.zeros_like(c4_stat[0])
    x_err[:] = 5


    fig = plt.figure(figsize=(6, 4))
    axes = plt.axes()
    plt.errorbar(xlist[0:6],v4,yerr=v4_e,xerr=x_err,label='$v_2\{4,|\Delta \eta|>0\}$',fmt='o',markersize=4.5)

    xerrminus = [2.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    xval = [7.5, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]

    yval = [0.0353,0.05674 ,0.07464,0.08397,0.08514,0.0797,0.06897]
    yerrminus = [0.0008,0.0013,0.0017,0.0019, 0.00196, 0.00183, 0.00159]
    plt.errorbar(xval,yval,yerr=yerrminus,xerr=xerrminus,label='$v_2\{4\}$ published Pb-Pb 5.02 TeV',fmt='o',markersize=4.5)

    axes.legend(loc="upper left",ncol=2, columnspacing=-0.1,handletextpad=0.1,fontsize=16)
    axes.set_xlabel("Centrality [\%]",fontsize=16)
    axes.set_ylabel("$v_n\{2\}$",fontsize=17)
    axes.yaxis.set_major_locator(MultipleLocator(0.05))
    axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.xaxis.set_major_locator(MultipleLocator(10))
    #axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.tick_params(which='both',direction='in',labelsize=14)
    axes.set_ylim([0,0.17])
    axes.set_xlim([0.0,80.0])
    plt.show()
    plt.tight_layout()
#     fig.savefig('/home/thoresen/Downloads' + '/v4ref.pdf')
#     fig.savefig('/home/thoresen/Downloads' + '/v4ref.png',dpi=100)
    return fig


def compare_vn2(v2=np.array([0,0,0,0,0,0]),v2_e=np.array([0,0,0,0,0,0])):

    noCent = len(v2[0][:])

    xlist = makelist(0,100,10)-10
    yer=v2_e#

    x_err = np.zeros_like(yer)
    x_err[:] = 5



    fig = plt.figure(figsize=(6, 4))
    axes = plt.axes()
    plt.errorbar(xlist[0:6],v2[0],yerr=yer,xerr=x_err,label='$v_2\{2,|\Delta \eta|>0\}$',fmt='o',markersize=4.5)
    plt.errorbar(xlist[0:6],v2[1],yerr=yer,xerr=x_err,label='$v_3\{2,|\Delta \eta|>0\}$',fmt='o',markersize=4.5)
    plt.errorbar(xlist[0:6],v2[2],yerr=yer,xerr=x_err,label='$v_4\{2,|\Delta \eta|>0\}$',fmt='o',markersize=4.5)

    xval = [2.5, 7.5, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
    xerrminus = [2.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    yval = [0.02839, 0.04566, 0.06551, 0.08707, 0.0991, 0.10414, 0.10286, 0.09746, 0.08881]
    yerrminus = [7.140028011149536E-4, 9.411163583744574E-4, 0.0010475208828467334, 0.0013819189556555044, 0.0015882694985423602, 0.0017223530416264838, 0.0018752333188166213, 0.0023645718428502017, 0.004577477471271705]

    v32_hep = [0.02067, 0.0232, 0.02799,0.03093,0.03314,0.03234,0.02705,0.02769]
    v32_hep_e = [0.00063, 0.0008, 0.00047,0.00062,0.00084,0.00136,0.00283,0.00547]

    v42_hep = [0.01146, 0.01291, 0.01383,0.01568,0.01702,0.01629,0.01771,0.01987]
    v42_hep_e = [0.00095,0.0012, 0.00079,0.00104,0.00147,0.00245,0.00407,0.00735]

    plt.errorbar(xval,yval,yerr=yerrminus,xerr=xerrminus,label='$v_2\{2, |\Delta \eta| > 1\}$  pub. Pb-Pb 5.02 TeV',fmt='o',markersize=4.5)

    xval = [2.5, 7.5, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
    xerrminus = [2.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    plt.errorbar(xval,v32_hep,yerr=v32_hep_e,xerr=xerrminus,label='$v_3\{2, |\Delta \eta| > 1\}$ pub. Pb-Pb 5.02 TeV ',fmt='o',markersize=4.5)

    plt.errorbar(xval,v42_hep,yerr=v42_hep_e,xerr=xerrminus,label='$v_4\{2, |\Delta \eta| > 1\}$ pub. Pb-Pb 5.02 TeV ',fmt='o',markersize=4.5)


    axes.legend(loc="upper left",ncol=2, columnspacing=-0.1,handletextpad=0.1,fontsize=16)
    axes.set_xlabel("Centrality [\%]",fontsize=16)
    axes.set_ylabel("$v_n\{2\}$",fontsize=17)
    axes.yaxis.set_major_locator(MultipleLocator(0.05))
    axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.xaxis.set_major_locator(MultipleLocator(10))
    axes.tick_params(which='both',direction='in',labelsize=14)
    axes.set_ylim([0,0.17])
    axes.set_xlim([0.0,80.0])
    plt.show()
    plt.tight_layout()
    #fig.savefig('/home/thoresen/Downloads' + '/v2ref.pdf')
    #fig.savefig('/home/thoresen/Downloads' + '/v2ref.png',dpi=100)
    return fig


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




def makeSingleCentPlots(vn_tpc, vn_tpc_e, vn_fmd, vn_fmd_e, vn_sys, foldername='', savefig=False, doalex=False, domc=False, cumulant2=True,yaxis='vn'):
    xlist = makelist(-4,6,50)
    eta = xlist
    mycolors = [C0,C1,C2]
    for cent in range(0,4):
        for pltn in range(3,4):

            if (cumulant2):
                nbin = pltn
            else:
                nbin = 1
            
            fig = plt.figure(figsize=(7.5, 4))
            mystring = str(cent*10) +'-'+ str(cent*10 + 10) + r' \%'
            fig.text(0.205,0.74,mystring,fontsize=14) #        fig.text(0.15,0.85,'0-10 %',fontsize=16)

            axis = plt.axes()

            p1 = []
            s1 = []   
            p1.append(axis.errorbar(0,float('NaN'),c='black', label="All charged particles",fmt='o',markersize=7.0))
            p1.append(axis.errorbar(0,float('NaN'),c='black', label=r"$0.2 GeV \leq p_T \leq 5 GeV$",fmt='o',markersize=7.0, markerfacecolor='none'))
            #    axis[2].plot((-3.5, 5), (out,out), '--', c=colors[n])

            s1.append("All charged particles")
            s1.append(r"$p_T$ extrapolated")

            l1 = axis.legend(p1,s1,fontsize=14,loc=2,ncol=1)
            axis.add_artist(l1)


            axis.set_xlabel("$\eta$", fontsize=17)
            axis.set_ylabel(yaxis, fontsize=17)
            p2 = []
            s2 = []
            for n in range(0,nbin):
                p2.append(axis.errorbar(xlist,vn_fmd[n][:,cent],yerr=vn_fmd_e[n][:,cent],fmt='o',markersize=4.5,c=mycolors[n]))#xerr=sigma_x,

                s2.append('$v_' + str(n+2)+'$')
                axis.errorbar(xlist,vn_tpc[n][:,cent],yerr=vn_tpc_e[n][:,cent],fmt='o',markersize=4.5,markerfacecolor='none',c=mycolors[n])#xerr=sigma_x,
                if (doalex):
                    if ( cent > 1 and cent < 8):
                        if (len(alex_vn[n][cent]) > 0):
                            if (cumulant2):
                                axis.plot(x_alex,alex_vn[n][cent+1], '-',color=mycolors[n])
                                axis.fill_between(x_alex, np.array(alex_vn[n][cent+1])-np.array(alex_vn_e[n][cent+1]), np.array(alex_vn[n][cent+1])+np.array(alex_vn_e[n][cent+1]),color=mycolors[n],alpha=0.1)
                            else:
                                axis.plot(x_alex,alex_v[3][cent+1], '-',color=C0)
                                axis.fill_between(x_alex, np.array(alex_vn[3][cent+1])-np.array(alex_vn_e[3][cent+1]), np.array(alex_vn[3][cent+1])+np.array(alex_vn_e[3][cent+1]),color=mycolors[n],alpha=0.1)                            

                if (domc):
                    vn_pt[n][:,cent] = vn_pt[n][:,cent]#*0.8
                    vn_pt_e[n][:,cent] = vn_pt_e[n][:,cent]#*0.8

                    axis.plot(xlist,vn_pt[n][:,cent], '-',c=mycolors[n])
                    axis.fill_between(xlist, vn_pt[n][:,cent]-vn_pt_e[n][:,cent], vn_pt[n][:,cent]+vn_pt_e[n][:,cent],color=mycolors[n],alpha=0.8)                

                axis.xaxis.set_major_locator(MultipleLocator(2))
                axis.xaxis.set_minor_locator(AutoMinorLocator())
                axis.yaxis.set_major_locator(MultipleLocator(0.05))
                axis.yaxis.set_minor_locator(AutoMinorLocator())

                axis.set_xlim([-3.5,6.0])

            if (True):#(pltn > 0):
                xerr = np.zeros(50)
                xerr[:] = xlist[1]-xlist[0]
                xerr = np.array([xerr,xerr])            
                _ = make_error_boxes(axis, eta, vn_fmd[0][:,cent], xerr/2, np.array([vn_sys[0][:,cent],vn_sys[0][:,cent]]),C0,alpha=0.5)
                _ = make_error_boxes(axis, eta, vn_tpc[0][:,cent], xerr/2, np.array([vn_sys[0][:,cent],vn_sys[0][:,cent]]),C0,alpha=0.5)
            if(pltn > 1):
                _ = make_error_boxes(axis, eta, vn_fmd[1][:,cent], xerr/2,  np.array([vn_sys[1][:,cent],vn_sys[1][:,cent]]),C1,alpha=0.5)
                _ = make_error_boxes(axis, eta, vn_tpc[1][:,cent], xerr/2,  np.array([vn_sys[1][:,cent],vn_sys[1][:,cent]]),C1,alpha=0.5)
            if (pltn > 2):
                _ = make_error_boxes(axis, eta, vn_fmd[2][:,cent], xerr/2,  np.array([vn_sys[2][:,cent],vn_sys[2][:,cent]]),C2,alpha=0.5)
                _ = make_error_boxes(axis, eta, vn_tpc[2][:,cent], xerr/2,  np.array([vn_sys[2][:,cent],vn_sys[2][:,cent]]),C2,alpha=0.5)

            axis.set_xlim([-3.5,5])
            axis.set_ylim([0,0.15])
        l2 = axis.legend(p2,s2,loc="upper right",ncol=2,fontsize=14)

        axis.add_artist(l2)
        if (savefig):
            fig.savefig(foldername + '/' + date +'_final_v22_sym'+str(cent) + 'n_' + str(pltn) + '' + '.pdf')
            fig.savefig(foldername + '/' + date +'_final_v22_sym'+str(cent) + 'n_' + str(pltn) + '' + '.png',dpi=100)

    plt.show()
