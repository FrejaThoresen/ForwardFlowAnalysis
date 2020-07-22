import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

class Plotting:

    def compare_methods(self, foldername = '.',prefix=''):
        etalist = np.linspace(-4,6,self.etabins)
        n = 0
        vbin = 5
        cbin = 0

        fig = plt.figure(figsize=(12,6))
        fig.text(0.14,0.82,'ALICE Simulation',fontsize=14)
        fig.text(0.14,0.76,'All Centralities',fontsize=14)
        vertexBin = 5
        fig.text(0.14,0.7,"%.2f" %(vlist[vbin])  + ' cm $\leq \mathrm{IP}_z < $' + "%.2f" %(vlist[vbin+1]) +' cm',fontsize=14)

        plt.errorbar(etalist,self.cauchy['corr'][n,vbin,:,cbin],yerr=self.cauchy['corr error'][n,vbin,:,cbin],\
                     fmt='o',markersize=4.5,label= r"$f= \exp (2 \cdot \gamma)$")
        plt.errorbar(etalist,self.fft['corr'][n,vbin,:,cbin],yerr=self.fft['corr error'][n,vbin,:,cbin],\
                     fmt='^',markersize=7.5,label="FFT")
        plt.errorbar(etalist,self.two_cauchy['corr'][n,vbin,:,cbin],yerr=self.two_cauchy['corr error'][n,vbin,:,cbin],\
                     fmt='o',markersize=4.5,label= r'$f = N \cdot \exp (2 \cdot \gamma_1)+(1-N)\cdot \exp(2\cdot \gamma_2)$')
        plt.errorbar(etalist,self.fft_const['corr'][n,vbin,:,cbin],yerr=self.fft_const['corr error'][n,vbin,:,cbin],\
                     fmt='^',markersize=7.5,label="FFT - const")

        axis = plt.axes()
        axis.set_xlabel('$\eta$',fontsize=14)

        axis.xaxis.set_major_locator(MultipleLocator(2))
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.set_xlim([-3.5,5.0])
        axis.tick_params(which='both',labelsize=14)

        axis.set_ylabel('Correction to $v_2$',fontsize=14)
        axis.set_ylim(1.0,1.6)       
                
        # Shrink current axis by 20%
        box = axis.get_position()
        axis.set_position([box.x0, box.y0, box.width * 0.65, box.height])

        # Put a legend to the right of the current axis
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
        plt.savefig(foldername + '/' + prefix +'_allcorrections' + '.pdf',bbox_inches='tight')
        plt.savefig(foldername + '/' + prefix +'_allcorrections' + '.png',bbox_inches='tight')
        plt.show()


makedark = False
vlist = np.linspace(-10,10,11)
def plot_chi2(etalist,redchi,foldername='.',prefix=''):
    color=plt.cm.viridis(np.linspace(0.2 if makedark else 0.0,1,9))
    c = 0

    fig = plt.figure()
    fig.set_figwidth(10.0)
    fig.set_figheight(5.0)
    fig.text(0.14,0.82,'ALICE Simulation',fontsize=14)
    #fig.text(0.18,0.84,r'p--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
    #fig.text(0.18,0.79,'DPMJET',fontsize=14)
    fig.text(0.14,0.76,r'Pb--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
    fig.text(0.14,0.71,'AMPT w. String Melting',fontsize=14)
    axis = plt.axes()
    for cbin in range(0,1):

        for vbin in range(1, 9):
            axis.errorbar(etalist,redchi[vbin,:,cbin],fmt='o',c=color[vbin],label= "%.2f" %(vlist[vbin])  + ' cm $\geq \mathrm{IP}_z >$' + "%.2f" %(vlist[vbin+1]) + ' cm')
    plt.plot((-4.0, 6.0), (1.,1.), '--', c='white' if makedark else 'black')

    axis.set_xlabel('$\eta$',fontsize=16)
    axis.set_ylabel(r'$\chi_\nu^2$',fontsize=16)
    axis.xaxis.set_major_locator(MultipleLocator(2))
    axis.xaxis.set_minor_locator(AutoMinorLocator())

    axis.set_xlim([-3.5,5.0])
    #axis.set_ylim(0.0,200.)
    axis.tick_params(which='both',labelsize=14)

    # Shrink current axis by 20%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.65, box.height])

    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
    plt.savefig(foldername +'/' + prefix + '_' + 'chi2.pdf',bbox_inches='tight')
    plt.savefig(foldername +'/' + prefix + '_' + 'chi2.png',bbox_inches='tight',dpi=100)
        
    plt.show()
    plt.clf()
    plt.close()

def plot_gamma(etalist,gamma,err,foldername='.',prefix=''):
    c = 0
    color=plt.cm.viridis(np.linspace(0.2 if makedark else 0.0,1,9))

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.text(0.14,0.82,'ALICE Simulation',fontsize=14)
    fig.text(0.14,0.76,r'Pb--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
    fig.text(0.14,0.72,'AMPT w. String Melting',fontsize=14)
    for cbin in range(0,1):
        for vbin in range(1, 9):
            plt.errorbar(etalist,gamma[vbin,:,cbin]+0.05,yerr=err[vbin,:,cbin],fmt='o',c=color[vbin],label= "%.2f" %(vlist[vbin])  + ' cm $\geq \mathrm{IP}_z >$' + "%.2f" %(vlist[vbin+1]) + ' cm')

    axis = plt.axes()
    axis.set_xlabel('$\eta$',fontsize=14)
    axis.set_ylabel('$\gamma$',fontsize=14)
    axis.xaxis.set_major_locator(MultipleLocator(2))
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_major_locator(MultipleLocator(0.05))
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.set_xlim([-3.5,5.0])
    axis.set_ylim(0.03,0.2)
    axis.tick_params(which='both',labelsize=14)

    # Shrink current axis by 20%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.65, box.height])

    # Put a legend to the right of the current axis
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
    plt.savefig(foldername +'/' + prefix + '_' + 'gammasvalues.pdf',bbox_inches='tight')
    plt.savefig(foldername +'/' + prefix + '_' + 'gammasvalues.png',bbox_inches='tight',dpi=100)
    plt.show()

    print foldername + '/' + prefix + str(c) + 'gammasvalues.pdf'

def plot_two_cauchy_corr(etalist,corr, corr_err, foldername = '.', prefix = ''):
    c = 0
    n=0
    for n in range(0,3):
        color=plt.cm.viridis(np.linspace(0.2 if makedark else 0.0,1,9))
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        fig.text(0.14,0.82,'ALICE Simulation',fontsize=14)
        fig.text(0.14,0.76,r'Pb--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
        fig.text(0.14,0.72,'AMPT w. String Melting',fontsize=14)
        for cbin in range(0,1):
            for vbin in range(1, 9):
                plt.errorbar(etalist,corr[n,vbin,:,cbin],yerr=corr_err[n,vbin,:,cbin]/2,fmt='o',c=color[vbin],label= "%.2f" %(vlist[vbin])  + ' cm $\geq \mathrm{IP}_z >$' + "%.2f" %(vlist[vbin+1]) + ' cm')

        axis = plt.axes()
        axis.set_xlabel('$\eta$',fontsize=14)
        axis.set_ylabel('$N \cdot \exp ('+str(n+2)+' \cdot \gamma_1)+(1-N)\cdot \exp('+str(n+2)+'\cdot \gamma_2)$',fontsize=14)
        axis.xaxis.set_major_locator(MultipleLocator(2))
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.set_xlim([-3.5,5.0])
        axis.set_ylim(1.0,1.8)
        axis.tick_params(which='both',labelsize=14)#direction='in',

        # Shrink current axis by 20%
        box = axis.get_position()
        axis.set_position([box.x0, box.y0, box.width * 0.65, box.height])

        # Put a legend to the right of the current axis
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
        plt.savefig(foldername +'/' + prefix + '_' + 'gammasvalues.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/' + prefix + '_' + 'gammasvalues.png',bbox_inches='tight')
        plt.show()

    print foldername +'/' + prefix + '_' + 'gammasvalues.png'


def plot_fft(etalist, fft, fft_err, foldername ='.',prefix='',const=False):
    color=plt.cm.viridis(np.linspace(0.2 if makedark else 0.0,1,10))
    c = 0

    #num_corr[num_corr_err > 0.2] = np.nan
    #num_corr_err[num_corr_err > 0.2] = np.nan
    for cbin in range(0,1):
        for n in range(0,3):
            fig = plt.figure()
            fig.set_figheight(5)
            fig.set_figwidth(10)            
            fig.text(0.14,0.82,'ALICE Simulation',fontsize=14)
            fig.text(0.14,0.76,r'Pb--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
            fig.text(0.14,0.72,'AMPT w. String Melting',fontsize=14)

            c = 0

            for vbin in range(0, 10):
                plt.errorbar(etalist,fft[n,vbin,:,cbin],yerr=fft_err[n,vbin,:,cbin],fmt='^', c=color[vbin],label= "%.2f" %(vlist[vbin])  + ' cm $\geq \mathrm{IP}_z >$' + "%.2f" %(vlist[vbin+1]) + ' cm',markersize=4)
                
                axis = plt.axes()
                axis.set_xlabel('$\eta$',fontsize=14)
                if (const):
                    axis.set_ylabel('Correction to $v_' + '%.1d' %(n+2) +'$ by FFT - const',fontsize=14)
                else:
                    axis.set_ylabel('Correction to $v_' + '%.1d' %(n+2) +'$ by FFT',fontsize=14)

                axis.xaxis.set_major_locator(MultipleLocator(2))
                axis.xaxis.set_minor_locator(AutoMinorLocator())
                # axis.set_ylim(1.0,1.6)
                axis.set_xlim([-3.5,5.5])
                if (n==0):
                   axis.set_ylim(1.0,1.5)
                elif (n==1):
                   axis.set_ylim(1.0,1.7)
                else:
                   axis.set_ylim(1.0,1.9)
                axis.tick_params(which='both',labelsize=14)#,direction='in'
            axis.set_xlim([-3.5,5.0])
            # Shrink current axis by 20%
            box = axis.get_position()
            axis.set_position([box.x0, box.y0, box.width * 0.65, box.height])

            # Put a legend to the right of the current axis
            axis.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
            plt.savefig(foldername+ '/'+ prefix + '_' + 'fftvalues' + str(n) +'.pdf',bbox_inches='tight')
            plt.savefig(foldername+ '/'+ prefix + '_' + 'fftvalues' + str(n) +'.png',bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.close()
        
    print foldername+ '/'+ prefix + '_' + 'fftvalues' + str(n) +'.png'


