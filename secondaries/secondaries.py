import fitting as secfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotting
import root_numpy as rnp
import seaborn as sns
from helpers.plotting.plotting_scripts import load_matplotlib_defaults
from rootpy.io import root_open


class Secondaries(plotting.Plotting):
    def __init__(self, filename,phibins,etabins,centbins, vertexbins,branchname):
        """
        Init analysis of secondaries in the FMD. Available analysis are:
            - Fast Fourier Transform
            - Fast Fourier Transform with substracted constant background
            - Cauchy fit with substracted constant background
            - Double Cauchy fit

        Parameters
        ----------
        filename: string
        phibins: no. bins in phi
        etabins: no. bins in eta (assumed range is [-4,6])
        centbins: no. bins in centrality or divisions in the FMD
        vertexbins: no. bins in vertex position (assumed range is [-10,10])
        """
        self.phibins = phibins
        self.etabins = etabins
        self.centbins = centbins
        self.vertexbins = vertexbins
        self.filename = filename
        n = 3

        # read data
        myfile = root_open(filename)
        deltas = myfile.Get(branchname).FindObject('Delta')
        nprim = rnp.hist2array(deltas.FindObject('prim').FindObject('fnoPrim'))

        eventInfo = myfile.Get(branchname).FindObject('EventInfo')
        vertex = rnp.hist2array(eventInfo.FindObject('Vertex'))

        # initialization
        self.dphi_eta = {'value' : 0., 'error' : 0.}
        self.dphi_eta['value'] = rnp.hist2array(deltas.FindObject('delta_phi_eta').FindObject('delta_phi_eta'))
        self.dphi_eta['error'] = np.sqrt(self.dphi_eta['value'])
        self.dphi_eta['error'] = self.dphi_eta['error']/nprim[:,None,:]
        #self.dphi_eta['error'] = self.dphi_eta['value']/nprim[:,None,:,:]*np.sqrt(np.power(np.sqrt(nprim[:,None,:,:])/nprim[:,None,:,:],2) + np.power(np.sqrt(self.dphi_eta['value'])/self.dphi_eta['value'],2))
        self.dphi_eta['value'] = self.dphi_eta['value']/nprim[:,None,:]

        self.deta_eta = {'value' : 0., 'error' : 0.}
        self.deta_eta['value'] = rnp.hist2array(deltas.FindObject('delta_eta_eta').FindObject('delta_eta_eta'))
        self.deta_eta['error'] = np.sqrt(self.deta_eta['value'])
        nprim = np.nansum(self.deta_eta['value'],axis=1)
        self.deta_eta['error'] = self.deta_eta['error']/nprim[:,None,:]
        self.deta_eta['error'] = self.deta_eta['value']/nprim[:,None,:]*np.sqrt(np.power(np.sqrt(nprim[:,None,:])/nprim[:,None,:],2) + np.power(np.sqrt(self.deta_eta['value'])/self.deta_eta['value'],2))
        self.deta_eta['value'] = self.deta_eta['value']/nprim[:,None,:]

        self.dphi_phi = {'value' : 0., 'error' : 0.}
        self.dphi_phi['value'] = rnp.hist2array(deltas.FindObject('delta_phi_phi').FindObject('delta_phi_phi'))
        self.dphi_phi['error'] = np.sqrt(self.dphi_phi['value'])
        nprim = np.nansum(self.dphi_phi['value'],axis=1)
        self.dphi_phi['error'] = self.dphi_phi['error']/nprim[:,None,:]
        self.dphi_phi['error'] = self.dphi_phi['value']/nprim[:,None,:]*np.sqrt(np.power(np.sqrt(nprim[:,None,:])/nprim[:,None,:],2) + np.power(np.sqrt(self.dphi_phi['value'])/self.dphi_phi['value'],2))
        self.dphi_phi['value'] = self.dphi_phi['value']/nprim[:,None,:]

        self.deta_phi = {'value' : 0., 'error' : 0.}
        self.deta_phi['value'] = rnp.hist2array(deltas.FindObject('delta_eta_phi').FindObject('delta_eta_phi'))
        self.deta_phi['error'] = np.sqrt(self.deta_phi['value'])
        nprim = np.nansum(self.deta_phi['value'],axis=1)
        self.deta_phi['error'] = self.deta_phi['error']/nprim[:,None,:]
        self.deta_phi['error'] = self.deta_phi['value']/nprim[:,None,:]*np.sqrt(np.power(np.sqrt(nprim[:,None,:])/nprim[:,None,:],2) + np.power(np.sqrt(self.deta_phi['value'])/self.deta_phi['value'],2))
        self.deta_phi['value'] = self.deta_phi['value']/nprim[:,None,:]

        print self.dphi_eta['value'].shape
        #self.dphi_eta['error'] = np.sqrt(self.dphi_eta['value'])/nprim[:,None,:,:]

        self.dphi_eta['value'][:,:,7:9] = float('NaN')
        self.dphi_eta['error'][:,:,19:21] = float('NaN')
        #delta_phi_err[samples,vertexBin,:,etaBin] = delta_phi1[samples,vertexBin,:,etaBin]*np.sqrt(np.power(np.divide(np.sqrt(nprim[samples,vertexBin,etaBin]),nprim[samples,vertexBin,etaBin],where=nprim[samples,vertexBin,etaBin]>0),2) + np.power(np.divide(np.sqrt(delta_phi[samples,vertexBin,:,etaBin]),delta_phi[samples,vertexBin,:,etaBin],where=delta_phi[samples,vertexBin,:,etaBin]>0),2))#np.nanstd(delta_phi,axis=0)                    

        #delta_phi_err[samples,vertexBin,:,etaBin] = delta_phi1[samples,vertexBin,:,etaBin]*np.sqrt(np.power(np.divide(np.sqrt(nprim[samples,vertexBin,etaBin]),nprim[samples,vertexBin,etaBin],where=nprim[samples,vertexBin,etaBin]>0),2) + np.power(np.divide(np.sqrt(delta_phi[samples,vertexBin,:,etaBin]),delta_phi[samples,vertexBin,:,etaBin],where=delta_phi[samples,vertexBin,:,etaBin]>0),2))#np.nanstd(delta_phi,axis=0)                    


        self.cauchy = {'amp'          : np.zeros((vertexbins,etabins,centbins)),   \
                       'gamma'        : np.zeros((vertexbins,etabins,centbins)),   \
                       'gamma error'  : np.zeros((vertexbins,etabins,centbins)),   \
                       'const'        : np.zeros((vertexbins,etabins,centbins)),   \
                       'redchi'       : np.zeros((vertexbins,etabins,centbins)),   \
                       'corr'         : np.zeros((n,vertexbins,etabins,centbins)), \
                       'corr error'   : np.zeros((n,vertexbins,etabins,centbins))}

        self.two_cauchy = {'amp'          : np.zeros((vertexbins,etabins,centbins)),   \
                           'gamma1'   : np.zeros((vertexbins,etabins,centbins)),       \
                           'gamma1 error' : np.zeros((vertexbins,etabins,centbins)),   \
                           'gamma2'       : np.zeros((vertexbins,etabins,centbins)),   \
                           'gamma2 error' : np.zeros((vertexbins,etabins,centbins)),   \
                           'N'            : np.zeros((vertexbins,etabins,centbins)),   \
                           'redchi'       : np.zeros((vertexbins,etabins,centbins)),   \
                           'corr'         : np.zeros((n,vertexbins,etabins,centbins)), \
                           'corr error'   : np.zeros((n,vertexbins,etabins,centbins))}

        self.fft = {   'corr'         : np.zeros((n,vertexbins,etabins,centbins)), \
                       'corr error'   : np.zeros((n,vertexbins,etabins,centbins))}

        self.fft_const = {'corr'      : np.zeros((n,vertexbins,etabins,centbins)), \
                       'corr error'   : np.zeros((n,vertexbins,etabins,centbins))}


    def symmetrise(self,data,err):
        #data = data - data[0]
        for i in range(0,len(data)/2):
            mean = (data[i] + data[-(i+1)])/2
            data[i] = mean
            data[-(i+1)] = mean
            mean1 = (err[i] + err[-(i+1)])/2
            err[i] = mean1
            err[-(i+1)] = mean1        
        return data, err

    def makedeltaphi(self,method,dmode="dphi_eta",xlist=None,ylist=None):
        """
        Make dphi plot as a function of eta and do specified analysis.

        """
        data = {'value' : 0, 'error' : 0}
        if (dmode == "dphi_eta"):
            data['value'] = self.dphi_eta['value']
            data['error'] =  self.dphi_eta['error']

        if (dmode == "deta_phi"):
            data['value'] = self.deta_phi['value']
            data['error'] =  self.deta_phi['error']

        if (dmode == "dphi_phi"):
            data['value'] = self.dphi_phi['value']
            data['error'] =  self.dphi_phi['error']

        if (dmode == "deta_eta"):
            data['value'] = self.deta_eta['value']
            data['error'] =  self.deta_eta['error']

        for cbin in range(0,1):
            for vbin in range(0,self.vertexbins):

                for ybin in range(0,len(ylist)):

                    _data = data['value'][vbin,:,ybin]
                    _err  = data['error'][vbin,:,ybin]
                    #_data, _err = self.symmetrise(_data,_err)
                    #if (np.nansum(_data) < 0.01): #or np.nansum(_err) < 0.01):#or np.nansum(_err) < 0.01
                    #    continue
                    #if (np.any(_data == 0.0)):
                    #    continue

                    if (method == 'cauchy'):
                        fit_out = secfit.fitcauchy(xlist, _data, _err)

                        self.cauchy['amp'][vbin,ybin] = fit_out['amplitude']
                        self.cauchy['const'][vbin,ybin] = fit_out['constant']
                        self.cauchy['gamma'][vbin,ybin] = fit_out['gamma']
                        self.cauchy['gamma error'][vbin,ybin] = fit_out['gamma error']
                        self.cauchy['redchi'][vbin,ybin] = fit_out['reduced chi']

                    if (method == 'two_cauchy'):
                        fit_out = secfit.fitTwoCauchy_N(xlist, _data, _err, ybin)

                        self.two_cauchy['amp'][vbin,ybin] = fit_out['amplitude']
                        self.two_cauchy['N'][vbin,ybin] = fit_out['N_1']
                        self.two_cauchy['gamma1'][vbin,ybin] = fit_out['gamma_1']
                        self.two_cauchy['gamma1 error'][vbin,ybin] = fit_out['gamma_1 error']
                        self.two_cauchy['gamma2'][vbin,ybin] = fit_out['gamma_2']
                        self.two_cauchy['gamma2 error'][vbin,ybin] = fit_out['gamma_2 error']                        
                        self.two_cauchy['redchi'][vbin,ybin] = fit_out['reduced chi']

                    if (method == 'fft'):
                        fft_out = secfit.fftTransform(xlist, _data, _err, False)
                        self.fft['corr'][0,vbin,ybin] = fft_out['v_2 corr.']
                        self.fft['corr'][1,vbin,ybin] = fft_out['v_3 corr.']
                        self.fft['corr'][2,vbin,ybin] = fft_out['v_4 corr.']
                        self.fft['corr error'][0,vbin,ybin] = fft_out['v_2 error']
                        self.fft['corr error'][1,vbin,ybin] = fft_out['v_3 error']
                        self.fft['corr error'][2,vbin,ybin] = fft_out['v_4 error']

                    if (method == 'fft_const'):
                        fft_out = secfit.fftTransform(xlist, _data, _err, True)

                        self.fft_const['corr'][0,vbin,ybin] = fft_out['v_2 corr.']
                        self.fft_const['corr'][1,vbin,ybin] = fft_out['v_3 corr.']
                        self.fft_const['corr'][2,vbin,ybin] = fft_out['v_4 corr.']
                        self.fft_const['corr error'][0,vbin,ybin] = fft_out['v_2 error']
                        self.fft_const['corr error'][1,vbin,ybin] = fft_out['v_3 error']
                        self.fft_const['corr error'][2,vbin,ybin] = fft_out['v_4 error']
        if (method == 'cauchy'):
            for n in range(0,3):
                self.cauchy['corr'][n,...] = np.exp((n+2)*self.cauchy['gamma'])
                self.cauchy['corr error'][n,...] = (n+2)*np.exp((n+2)*self.cauchy['gamma'])*self.cauchy['gamma error']

        if (method == 'two_cauchy'):
            print self.two_cauchy['corr'].shape
            for n in range(0,3):

                self.two_cauchy['corr'][n,...] = self.two_cauchy['N']*np.exp((n+2)*self.two_cauchy['gamma1'])+(1-self.two_cauchy['N'])*np.exp((n+2)*self.two_cauchy['gamma2'])
                self.two_cauchy['corr error'][n,...] = np.sqrt( \
                                                np.power((n+2)*self.two_cauchy['N']*np.exp((n+2)*self.two_cauchy['gamma1'])*self.two_cauchy['gamma1 error'],2) \
                                              + np.power((n+2)*(1-self.two_cauchy['N'])*np.exp((n+2)*self.two_cauchy['gamma2'])*self.two_cauchy['gamma2 error'],2))


    def makeplot(self,singleplot=False,dmode="dphi_eta",method='cauchy',foldername='.',prefix='',xlist=None,ylist=None):
        load_matplotlib_defaults()
        plt.style.use('seaborn')
        sns.set_context('paper')
        sns.set(style='ticks')
        mpl.rcParams['lines.linewidth'] = 1
        mpl.rcParams['lines.markersize'] = 4.5

        color=plt.cm.inferno(np.linspace(0.0,1,9))
        vertexlist = np.linspace(-10,10,11)
        data = {'value' : 0, 'error' : 0}
        # xlist = 0
        if (dmode == "dphi_eta"):
            data['value'] = self.dphi_eta['value']
            data['error'] =  self.dphi_eta['error']

            miny = 3
            if (singleplot):
                maxy = miny + 1
            else:
                maxy = 11

        if (dmode == "deta_phi"):
            data['value'] = self.deta_phi['value']
            data['error'] =  self.deta_phi['error']

        if (dmode == "dphi_phi"):
            data['value'] = self.dphi_phi['value']
            data['error'] =  self.dphi_phi['error']

        if (dmode == "deta_eta"):
            data['value'] = self.deta_eta['value']
            data['error'] =  self.deta_eta['error']

        miny = 3
        if (singleplot):
            maxy = miny + 1
        else:
            maxy = 11

        fig = plt.figure(figsize=(10,7))
        axis = plt.gca()
        # fig.text(0.48,0.85,'Centrality %.0f' %(centBin*10) + '-%.0f' %(centBin*10 + 10) +'%',fontsize=14)
        #fig.text(0.535,0.85,'All Centralities',fontsize=14)  

        fig.text(0.14,0.84,'ALICE Simulation',fontsize=14)
        #fig.text(0.18,0.84,r'p--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
        #fig.text(0.18,0.8,'DPMJET',fontsize=14)
        fig.text(0.14,0.79,r'Pb--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
        fig.text(0.14,0.75,'AMPT w. String Melting',fontsize=14)
        fig.text(0.14,0.7,'FMD',fontsize=14)
        fig.text(0.4,0.84,"%.2f" %(vertexlist[0])  + ' cm $\geq \mathrm{IP}_z >$' + "%.2f" %(vertexlist[1]) +' cm',fontsize=14)
#        axis = [None]*2

        fig1 = plt.figure(figsize=(10,5))
        axis1 = plt.gca()
        fig1.text(0.14,0.83,'ALICE Simulation',fontsize=14)
        #fig.text(0.18,0.84,r'p--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
        #fig.text(0.18,0.8,'DPMJET',fontsize=14)
        fig1.text(0.14,0.78,r'Pb--Pb $\sqrt{s_{\rm{NN}}} = $ 5.02 TeV',fontsize=14)
        fig1.text(0.14,0.74,'AMPT w. String Melting',fontsize=14)
        fig1.text(0.14,0.69,'FMD',fontsize=14)
        fig1.text(0.4,0.83,"%.2f" %(vertexlist[0])  + ' cm $\geq \mathrm{IP}_z >$' + "%.2f" %(vertexlist[0+1]) +' cm',fontsize=14)
        first = True
        icolor = 0
        y_label = ''
        if (dmode == 'dphi_phi' or dmode == 'deta_phi'):
            y_label = r'\varphi'
        else:
            y_label = r'\eta'
        xfit = np.linspace(xlist[0],xlist[-1],1000)
        for cbin in range(0,1):
            for vbin in range(0,1):
                for ybin in range(miny,maxy):

                    if (method == 'two_cauchy'):
                        if (first):
                            axis.plot(xfit, np.full_like(xfit, float('NaN')), "-",c='black',label=r'$f = A \cdot (f_1 + f_2)$') # Plot of the data and the fit
                            axis.plot(xfit, np.full_like(xfit, float('NaN')), ":",c='black',alpha=0.5,label=r'$f_1 = N \cdot [\pi \cdot \gamma_1 \cdot (1+ (x/\gamma_1)^2)]^{-1}$') # Plot of the data and the fit
                            axis.plot(xfit, np.full_like(xfit, float('NaN')), "--",c='black',alpha=0.5,label=r'$f_2 = (1-N) \cdot [\pi \cdot \gamma_2 \cdot (1+ (x/\gamma_2)^2)]^{-1}$') # Plot of the data and the fit
                            first = False

                        fit1 = secfit.cauchy_values(xfit, self.two_cauchy['gamma1'][vbin,ybin],self.two_cauchy['amp'][vbin,ybin]*self.two_cauchy['N'][vbin,ybin],0.)
                        fit2 = secfit.cauchy_values(xfit, self.two_cauchy['gamma2'][vbin,ybin],self.two_cauchy['amp'][vbin,ybin]*(1-self.two_cauchy['N'][vbin,ybin]),0.)
                        fit12 = secfit.two_cauchy_values(xfit, self.two_cauchy['gamma1'][vbin,ybin],self.two_cauchy['gamma2'][vbin,ybin],self.two_cauchy['N'][vbin,ybin],self.two_cauchy['amp'][vbin,ybin],0.0)
                        residual = secfit.two_cauchy_residual(xlist, data['value'][vbin,:,ybin],data['error'][vbin,:,ybin], self.two_cauchy['N'][vbin,ybin], self.two_cauchy['amp'][vbin,ybin], self.two_cauchy['gamma1'][vbin,ybin], self.two_cauchy['gamma2'][vbin,ybin])

                        # Plot of the fit
                        axis.plot(xfit, fit1 + 0.4*icolor, ":",c=color[icolor],alpha=0.5)  # Plot of the data and the fit
                        axis.plot(xfit, fit2 + 0.4*icolor, "--",c=color[icolor],alpha=0.5) # Plot of the data and the fit
                        axis.plot(xfit, fit12 + 0.4*icolor, "-",c=color[icolor])           # Plot of the data and the fit
                        # Plot the residual
                        axis1.errorbar(xlist, residual,fmt='-o',c=color[icolor],label= "%.2f" %(ylist[ybin]-0.125)  + '$\leq '+y_label+' <$' + "%.2f" %(ylist[ybin+1]-0.125))

                    else: # single cauchy
                        fit = secfit.cauchy_values(xfit, self.cauchy['gamma'][vbin,ybin],self.cauchy['amp'][vbin,ybin],self.cauchy['const'][vbin,ybin])
                        _data, _err = self.symmetrise(data['value'][vbin,:,ybin],data['error'][vbin,:,ybin])

                        residual = secfit.cauchy_residual(xlist, _data,_err, self.cauchy['gamma'][vbin,ybin], self.cauchy['amp'][vbin,ybin],self.cauchy['const'][vbin,ybin])

                        if (first):
                            axis.plot(xfit, np.full_like(xfit, float('NaN')), "-",c='black',alpha=0.5,label=r'$f = [\pi \cdot \gamma \cdot (1+ (x/\gamma)^2)]^{-1}$') # Plot of the data and the fit
                            first = False

                        axis.plot(xfit, fit + 0.4*icolor, "-",c=color[icolor]) # Plot of the data and the fit
                        axis1.errorbar(xlist, residual,fmt='-o',c=color[icolor],label= "%.2f" %(ylist[ybin]-0.125)  + '$\leq '+y_label+' <$' + "%.2f" %(ylist[ybin+1]-0.125),markersize=4.5) # Plot of the data and the fit
                    axis.errorbar(xlist,data['value'][vbin,:,ybin] + 0.4*icolor,yerr=data['error'][vbin,:,ybin],label= "%.2f" %(ylist[ybin]-0.125)  + '$\leq '+y_label+' <$' + "%.2f" %(ylist[ybin+1]-0.125) + ", +" +str(icolor*0.4),fmt='o',c=color[icolor])
                    icolor = icolor + 1


        #axis.set_ylabel(r'$ \frac{1}{N_{\text{mother}}}\frac{\mathrm{d}N_{\text{secondaries}}}{\mathrm{d}(\Delta \varphi)}$',fontsize=20)
        axis.legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
        axis.tick_params(which='both',labelsize=14) #direction='in',top='on',right='on',

        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels, ncol=2, loc='upper right',columnspacing=7.5,handletextpad=0.1,fontsize=14)
        box = axis.get_position()

        axis.set_position([box.x0, box.y0, box.width * 0.65, box.height])

        # Put a legend to the right of the current axis
        axis.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize=13)
        #axis.set_ylim([0,5])
        # if (method == 'twoCauchy'):
        #     axis1.set_ylim([0.0,3.8])
        # if (method == 'cauchy'):
        #     axis1.set_ylim([0.0,11.5])
        if (dmode == 'dphi_eta' or dmode == 'dphi_phi'):
            axis1.set_xlabel(r"$\Delta \varphi$", fontsize=17)
            axis.set_xlabel(r"$\Delta \varphi$", fontsize=17)

        else:
            axis1.set_xlabel(r"$\Delta \eta$", fontsize=17)
            axis.set_xlabel(r"$\Delta \eta$", fontsize=17)

        
        axis1.set_ylabel(r'$ \frac{data - fit}{\sigma}$',fontsize=20)
        axis1.legend(loc="upper right", bbox_to_anchor=[0, 1],ncol=2)
        axis1.tick_params(which='both',labelsize=14)

        handles1, labels1 = axis.get_legend_handles_labels()
        axis1.legend(handles, labels, ncol=2, loc='upper right',columnspacing=7.5,handletextpad=0.1,fontsize=14)
        box1 = axis.get_position()
        axis1.set_position([box.x0, box.y0, box.width * 0.65, box.height])

        # Put a legend to the right of the current axis
        axis1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),fontsize=13)
        #axis.set_ylim([0,1.5])

        fig.savefig(foldername + '/' + method + prefix + '_' + str(ybin) + '_fits_deltaphi.pdf',bbox_inches='tight')
        fig.savefig(foldername + '/' + method + prefix + '_' + str(ybin) + '_fits_deltaphi.png',bbox_inches='tight',pdi=100)
        fig1.savefig(foldername + '/' + method + prefix + '_res_' + str(ybin) + '_fits_deltaphi_residual.pdf',bbox_inches='tight')
        fig1.savefig(foldername + '/' + method + prefix + '_res_' + str(ybin) + '_fits_deltaphi_residual.png',bbox_inches='tight')
        #print foldername + '/' +date+'_' + 'allc'+str(centBin) +'_z'+ str(vertexBin) +  '.eps'
        plt.show()
        fig.clf()
        plt.close()