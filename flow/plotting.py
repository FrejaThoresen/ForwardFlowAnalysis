import numpy as np
import matplotlib.pyplot as plt
import helpers.general as gen
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from rootpy.plotting.style import get_style, set_style

set_style('ATLAS',mpl=True)


class Plot:

    def plot_c2(self,foldername,no_cent,c2,c2_e,ntot):

        xlist = [2.5,7.5,15,25,35,45,55,65,76]#gen.makelist(0,no_cent*10,no_cent)-10
        x_err = np.zeros_like(xlist)
        x_err[:] = (xlist[1]-xlist[0])/2.0

        fig = plt.figure(figsize=(6, 4))
        axes = plt.axes()
        for n in range(0,ntot):
            plt.errorbar(xlist,c2[n,0,:],yerr=c2_e[n,0,:],xerr=x_err,label='$c_'+str(n+2)+'\{2\}$',fmt='o',markersize=4.5)

        axes.legend(loc="upper right", columnspacing=-0.1,handletextpad=0.1,fontsize=17)
        axes.set_xlabel(r"Centrality [%]",fontsize=16)
        axes.set_ylabel("$c_n\{2\}$",fontsize=17)

        axes.xaxis.set_major_locator(MultipleLocator(10))
        axes.tick_params(which='both',direction='in',labelsize=14) #,top='on',right='on'
        axes.set_xlim([0.0,no_cent*10])

        plt.show()
        fig.savefig(foldername + '/' + '_c2ref.pdf')
        plt.clf()
        plt.close()
        print foldername + '/' + '_c2ref.pdf'


    def plot_vn2ref(self,foldername,no_cent,vn2,vn2_e,ntot):

        xlist = gen.makelist(0,no_cent*10,no_cent)-10
        x_err = np.zeros_like(xlist)
        x_err[:] = (xlist[1]-xlist[0])/2.0
        fig = plt.figure(figsize=(6, 4))
        axes = plt.axes()
        for n in range(0,ntot):
            plt.errorbar(xlist,vn2[n,0,:],yerr=vn2_e[n,0,:],xerr=x_err,label='$v_'+str(n+2)+'\{2\}$',fmt='o',markersize=4.5)

        axes.legend(loc="upper right", columnspacing=-0.1,handletextpad=0.1,fontsize=17)
        axes.set_xlabel(r"Centrality [%]",fontsize=16)
        axes.set_ylabel("$v_n\{2\}$",fontsize=17)
        axes.xaxis.set_major_locator(MultipleLocator(10))
        axes.tick_params(which='both',direction='in',labelsize=14) #,top='on',right='on'
        axes.set_xlim([0.0,no_cent*10])

        fig.savefig(foldername + '/' + '_vn2ref.pdf')
        plt.show()
        print foldername + '/' + '_vn2ref.pdf'
        plt.clf()
        plt.close()

    def plot_d2(self,foldername,no_cent,d2,d2_e,ntot):
        axis = [None]*no_cent
        yaxis=r'$d_n\{2\}$'
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

        fig.set_figheight(7)
        fig.set_figwidth(13)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=0.001)

        for n in range(0,ntot):
            for c in range(0,6):
                if (c == 0):
                    axis[0].errorbar(self.etalist,d2[n,:,c],yerr=d2_e[n,:,c],label='$d_' + str(n+2)+'$',fmt='o',markersize=4.5)
                    axis[0].legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
                else:
                    axis[c].errorbar(self.etalist,d2[n,:,c],yerr=d2_e[n,:,c],label='$d_' + str(n+2)+'{2}$',fmt='o',markersize=4.5)

                axis[c].xaxis.set_major_locator(MultipleLocator(2))
                axis[c].xaxis.set_minor_locator(AutoMinorLocator())

                axis[c].tick_params(which='both',direction='in') 
                axis[c].set_xlim([-3.5,5.0])

        handles, labels = axis[0].get_legend_handles_labels()
        axis[0].legend(handles, labels, ncol=2, loc='upper right',columnspacing=-0.1,handletextpad=0.1)
        fig.savefig(foldername + '/' + '_d2.pdf')
        plt.show()
        print foldername + '/' + '_d2.pdf'   
        plt.clf()
        plt.close()
        return


    def plot_vn2(self,foldername,no_cent,vn2,vn2_e,ntot):
        axis = [None]*no_cent
        yaxis=r'$v_n\{2\}$'
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

        fig.set_figheight(7)
        fig.set_figwidth(13)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.001, hspace=0.001)

        for n in range(0,ntot):
            for c in range(0,6):
                if (c == 0):
                    axis[0].errorbar(self.etalist,vn2[n,:,c],yerr=vn2_e[n,:,c],label='$v_' + str(n+2)+'$',fmt='o',markersize=4.5)
                    axis[0].legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
                else:
                    axis[c].errorbar(self.etalist,vn2[n,:,c],yerr=vn2_e[n,:,c],label='$v_' + str(n+2)+'{2}$',fmt='o',markersize=4.5)

                axis[c].xaxis.set_major_locator(MultipleLocator(2))
                axis[c].xaxis.set_minor_locator(AutoMinorLocator())

                axis[c].tick_params(which='both',direction='in') 
                axis[c].set_xlim([-3.5,5.0])
            axis[c].set_ylim([0.0,0.15])

        handles, labels = axis[0].get_legend_handles_labels()
        axis[0].legend(handles, labels, ncol=2, loc='upper right',columnspacing=-0.1,handletextpad=0.1)
        fig.savefig(foldername + '/' + '_vn2.pdf')
        plt.show()
        print foldername + '/' + '_vn2.pdf'   
        plt.clf()
        plt.close()
        return

    def plot_c4(self,foldername,no_cent,c4,c4_e,ntot):
        xlist = gen.makelist(0,no_cent*10,no_cent)-10

        x_err = np.zeros_like(xlist)
        x_err[:] = (xlist[1]-xlist[0])/2.0

        fig = plt.figure(figsize=(6, 4))
        axes = plt.axes()
        for n in range(0,ntot):
            plt.errorbar(xlist,-c4[n,0,0:no_cent],yerr=c4_e[n,0,0:no_cent],xerr=x_err,label='$c_'+str(n+2)+'\{4\}$',fmt='o',markersize=4.5)

        axes.legend(loc="upper right", columnspacing=-0.1,handletextpad=0.1,fontsize=17)
        axes.set_xlabel("Centrality [%]",fontsize=16)
        axes.set_ylabel("$-c_n\{4\}$",fontsize=17)

        axes.xaxis.set_major_locator(MultipleLocator(10))
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.tick_params(which='both',direction='in',top='on',right='on')
        axes.set_xlim([0.0,no_cent*10])

        plt.show()
        fig.savefig(foldername + '/c4.pdf')
        plt.clf()
        plt.close()


    def plot_vn4ref(self,foldername,no_cent,vn4,vn4_e,ntot):
        xlist = gen.makelist(0,no_cent*10,no_cent)-10

        x_err = np.zeros_like(xlist)
        x_err[:] = (xlist[1]-xlist[0])/2.0

        fig = plt.figure(figsize=(6, 4))
        axes = plt.axes()
        for n in range(0,ntot):
            plt.errorbar(xlist,vn4[n,0,0:no_cent],yerr=vn4_e[n,0,0:no_cent],xerr=x_err,label='$v_'+str(n+2)+'\{4\}$',fmt='o',markersize=4.5)

        axes.legend(loc="upper right", columnspacing=-0.1,handletextpad=0.1,fontsize=17)
        axes.set_xlabel("Centrality [%]",fontsize=16)
        axes.set_ylabel("$v_n\{4\}$",fontsize=17)

        axes.xaxis.set_major_locator(MultipleLocator(10))
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.tick_params(which='both',direction='in',top='on',right='on')
        axes.set_xlim([0.0,no_cent*10])

        plt.show()
        fig.savefig(foldername + '/vn4ref.pdf')
        plt.clf()
        plt.close()


