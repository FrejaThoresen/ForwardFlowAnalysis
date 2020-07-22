
import time

import helpers.general as gen
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helpers.constants import *
from lmfit import Minimizer, Parameters
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


class Sys:

    def vnn_cent_sys_new(self,ratio,ratio_e,vn_upper,vn_upper_e,vn_lower,vn_lower_e,vn_upper_label,vn_lower_label,n_max,cum,foldername,name):
        etalist=self.etalist[9:19]
        ratio = ratio[:,9:19,:]
        ratio_e = ratio_e[:,9:19,:]
        vn_upper = vn_upper[:,9:19,:]
        vn_upper_e = vn_upper_e[:,9:19,:]
        vn_lower = vn_lower[:,9:19,:]
        vn_lower_e = vn_lower_e[:,9:19,:]


        barlow = np.zeros((n_max,10,9))
        sys = np.zeros((n_max,9))

        for n in range(0,n_max):
            ax = [None]*9
            f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5]),(ax[6],ax[7],ax[8])) = plt.subplots(3,3, sharex=True,sharey=True)
            c_endd = 7
            c_start = 0
            if (n==0):
                color=plt.cm.Blues(np.linspace(0.4,1.0,9))
            if (n==1):
                c_endd = 7
                color=plt.cm.Oranges(np.linspace(0.4,1.0,9))
            if (n==2):
                c_endd = 7
                color=plt.cm.Greens(np.linspace(0.4,1.0,9))        
            if (cum == 4):
                c_endd = 7
                c_start = 1

            allcolor=['#0000cc','#cc0000','#006600']

            sigma_x = (etalist[1] - etalist[0])/2
            centlist = np.array([0,5,10,20,30,40,50,60,70,80])

            fon=14
            allcolor = [C0,C1,C2]

            c_end = c_endd


            f.set_figwidth(8)
            f.set_figheight(4)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,7):
                if (c < c_end and c >= c_start):
                    value = ratio[n,:,c]
                    value, outliers = self.detect_outlier(value,3)
                    barlow[n,:,c]=np.divide(np.fabs(vn_lower[n,:,c]-vn_upper[n,:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[n,:,c],2)-np.power(vn_upper_e[n,:,c],2))))

                    error = ratio_e[n,:,c]
                    residual, fit_out = self.fitconst(etalist, value,error)

                    error[value == float('NaN')] = float('NaN')

                    sys[n,c] = gen.weightedStd(value,error,0)
                    rect1 = Rectangle((etalist[0]-sigma_x, 1.0-sys[n,c]), (-etalist[0]+etalist[-1])+sigma_x, 2*sys[n,c],facecolor='grey', edgecolor='grey',alpha=0.4)

                    ax[c].add_patch(rect1)  
                    ax[c].errorbar(etalist,value,yerr=error,c=color[6],ecolor=color[6],elinewidth=2,markersize=4,fmt='o')

                    if (fit_out['c error'] is not None):
                        ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.5,edgecolor=None,facecolor=color[6])
                        ax[c].text(-0.9,1.09,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                    ax[c].text(-0.9,1.05,r'$\sigma_w='+str(np.around(sys[n,c],4))+'$',fontsize=12)

                    sys[n,c] += fit_out['c']
                    sys[n,c] = np.fabs(sys[n,c]-1)*100

                    
                    ax[c].axhline(y=1.0,ls=":", c=".5")
                ax[c].set(xticks=np.arange(-1.1,1.1,0.5), yticks=np.arange(0.9,1.11,0.05),xlim=(-1.1, 1.1), ylim=(0.9, 1.11))

                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)

                ax[3].set_ylabel(r'$v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '/v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}$',fontsize=fon+4)

                ax[6].set_xlabel(r'$\eta$',fontsize=fon+2)
                ax[7].set_xlabel(r'$\eta$',fontsize=fon+2)
                ax[8].set_xlabel(r'$\eta$',fontsize=fon+2)

            plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.close()

            fig = plt.figure(figsize=(4.0,2.5))
            for c in range(c_start,c_end):
                plt.errorbar(etalist,barlow[n,:,c],c=color[c],fmt='o',label=str(centlist[c])+ ' - ' + str(centlist[c+1])+' %')
            plt.gca().set_ylabel(r'$\frac{|v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '- v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}|}{\sqrt{|\sigma^{' + vn_upper_label + ',2}' + '- \sigma^{' + vn_lower_label + ',2}|}} $',fontsize=22)
            plt.gca().tick_params(axis='both', which='major', labelsize=fon)
            plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
            plt.legend(fontsize=13,bbox_to_anchor=[1.,1.])
            plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')        
            plt.show()

        return np.array(sys), np.array(barlow)        

    def vnn_all_sys_new(self,ratio,ratio_e,vn_upper,vn_upper_e,vn_lower,vn_lower_e,vn_upper_label,vn_lower_label,n_max,cum,foldername,name):
        etalist=self.etalist[:]
        barlow = np.zeros((n_max,34,9))
        sys = np.zeros((n_max,9))

        for n in range(0,n_max):
            ax = [None]*9
            f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5]),(ax[6],ax[7],ax[8])) = plt.subplots(3,3, sharex=True,sharey=True)
            c_endd = 7
            c_start = 0
            if (n==0):
                color=plt.cm.Blues(np.linspace(0.4,1.0,9))
            if (n==1):
                c_endd = 7
                color=plt.cm.Oranges(np.linspace(0.4,1.0,9))
            if (n==2):
                c_endd = 7
                color=plt.cm.Greens(np.linspace(0.4,1.0,9))        
            if (cum == 4):
                c_endd = 7
                c_start = 1

            allcolor=['#0000cc','#cc0000','#006600']

            sigma_x = (etalist[1] - etalist[0])/2
            centlist = np.array([0,5,10,20,30,40,50,60,70,80])

            fon=14
            allcolor = [C0,C1,C2]

            c_end = c_endd


            f.set_figwidth(8)
            f.set_figheight(4)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,7):
                if (c < c_end and c >= c_start):
                    value = ratio[n,:,c]
                    value, outliers = self.detect_outlier(value,3)
                    barlow[n,:,c]=np.divide(np.fabs(vn_lower[n,:,c]-vn_upper[n,:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[n,:,c],2)-np.power(vn_upper_e[n,:,c],2))))

                    error = ratio_e[n,:,c]
                    residual, fit_out = self.fitconst(etalist, value,error)

                    error[value == float('NaN')] = float('NaN')

                    sys[n,c] = gen.weightedStd(value,error,0)
                    rect1 = Rectangle((etalist[0]-sigma_x, 1.0-sys[n,c]), (-etalist[0]+etalist[-1])+sigma_x, 2*sys[n,c],facecolor='grey', edgecolor='grey',alpha=0.4)

                    ax[c].add_patch(rect1)  
                    ax[c].errorbar(etalist,value,yerr=error,c=color[6],ecolor=color[6],elinewidth=2,markersize=4,fmt='o')

                    if (fit_out['c error'] is not None):
                        ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.5,edgecolor=None,facecolor=color[6])
                        ax[c].text(-0.9,1.09,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                    
                    ax[c].axhline(y=1.0,ls=":", c=".5")
                    ax[c].text(-0.9,1.05,r'$\sigma_w='+str(np.around(sys[n,c],4))+'$',fontsize=12)
                ax[c].set(xticks=np.arange(-3.5,5.0,2.), yticks=np.arange(0.9,1.11,0.05),xlim=(-3.5, 5.0), ylim=(0.9, 1.11))

                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)

                ax[3].set_ylabel(r'$v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '/v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}$',fontsize=fon+4)

                ax[6].set_xlabel(r'$\eta$',fontsize=fon+2)
                ax[7].set_xlabel(r'$\eta$',fontsize=fon+2)
                ax[8].set_xlabel(r'$\eta$',fontsize=fon+2)

            plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.close()

            fig = plt.figure(figsize=(4.0,2.5))
            for c in range(c_start,c_end):
                plt.errorbar(etalist,barlow[n,:,c],c=color[c],fmt='o',label=str(centlist[c])+ ' - ' + str(centlist[c+1])+' %')
            plt.gca().set_ylabel(r'$\frac{|v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '- v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}|}{\sqrt{|\sigma^{' + vn_upper_label + ',2}' + '- \sigma^{' + vn_lower_label + ',2}|}} $',fontsize=22)
            plt.gca().tick_params(axis='both', which='major', labelsize=fon)
            plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
            plt.legend(fontsize=13,bbox_to_anchor=[1.,1.])
            plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')        
            plt.show()

        return np.array(sys), np.array(barlow)        


    def vnn_fwd_sys_new(self,ratio,ratio_e,vn_upper,vn_upper_e,vn_lower,vn_lower_e,vn_upper_label,vn_lower_label,n,cum,foldername,name):
        etalist=self.etalist[:]
        ax = [None]*9
        f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5]),(ax[6],ax[7],ax[8])) = plt.subplots(3,3, sharex=True,sharey=True)
        c_endd = 7
        c_start = 0
        if (n==0):
            color=plt.cm.Blues(np.linspace(0.4,1.0,9))
        if (n==1):
            c_endd = 7
            color=plt.cm.Oranges(np.linspace(0.4,1.0,9))
        if (n==2):
            c_endd = 7
            color=plt.cm.Greens(np.linspace(0.4,1.0,9))        
        if (cum == 4):
            c_endd = 7
            c_start = 1

        allcolor=['#0000cc','#cc0000','#006600']

        sigma_x = (etalist[1] - etalist[0])/2
        centlist = np.array([0,5,10,20,30,40,50,60,70,80])

        fon=14
        allcolor = [C0,C1,C2]

        sys = np.zeros((9))
        c_end = c_endd
        barlow = np.zeros((34,9))


        f.set_figwidth(8)
        f.set_figheight(4)
        f.subplots_adjust(wspace=0.05,hspace=0.3)
        for c in range(0,7):
            if (c < c_end and c >= c_start):
                value = ratio[:,c]
                value[7:21] = float('NaN')
                value, outliers = self.detect_outlier(value,3)
                barlow[:,c]=np.divide(np.fabs(vn_lower[:,c]-vn_upper[:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[:,c],2)-np.power(vn_upper_e[:,c],2))))

                error = ratio_e[:,c]
                error[7:21] = float('NaN')

                residual, fit_out = self.fitconst(etalist, value,error)

                error[value == float('NaN')] = float('NaN')

                sys[c] = gen.weightedStd(value,error,0)
                rect1 = Rectangle((etalist[0]-sigma_x, 1.0-sys[c]), (-etalist[0]+etalist[-1])+sigma_x, 2*sys[c])

                errorboxes = []
                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          
                ax[c].errorbar(etalist,value,yerr=error,c=color[6],ecolor=color[6],elinewidth=2,markersize=4,fmt='o')

                if (fit_out['c error'] is not None):
                    ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.5,edgecolor=None,facecolor=color[6])
                    ax[c].text(-0.9,1.09,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                
                ax[c].axhline(y=1.0,ls=":", c=".5")
                ax[c].text(-0.9,1.05,r'$\sigma_w='+str(np.around(sys[c],4))+'$',fontsize=12)
            ax[c].set(xticks=np.arange(-3.5,5.,2.), yticks=np.arange(0.9,1.11,0.05),xlim=(-3.5, 5.), ylim=(0.9, 1.11))

            ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
            ax[c].spines['right'].set_visible(False)
            ax[c].spines['top'].set_visible(False)

            ax[c].tick_params(axis='both', which='major', labelsize=fon)
            ax[c].tick_params(axis='both', which='minor', labelsize=fon)

            ax[3].set_ylabel(r'$v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '/v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}$',fontsize=fon+4)

            ax[6].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[7].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[8].set_xlabel(r'$\eta$',fontsize=fon+2)

        plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        fig = plt.figure(figsize=(4.0,2.5))
        for c in range(c_start,c_end):
            plt.errorbar(etalist,barlow[:,c],c=color[c],fmt='o',label=str(centlist[c])+ ' - ' + str(centlist[c+1])+' %')
        plt.gca().set_ylabel(r'$\frac{|v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '- v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}|}{\sqrt{|\sigma^{' + vn_upper_label + ',2}' + '- \sigma^{' + vn_lower_label + ',2}|}} $',fontsize=22)
        plt.gca().tick_params(axis='both', which='major', labelsize=fon)
        plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
        plt.legend(fontsize=13,bbox_to_anchor=[1.,1.])
        plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')        
        plt.show()
        return np.array(sys), np.array(barlow)        


    def vnn_cent_sys(self,vn_upper,vn_upper_e,vn_upper_label,vn_lower,vn_lower_e,vn_lower_label,n,cum,foldername,name):
        etalist=self.etalist[9:19]
        ax = [None]*9
        f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5]),(ax[6],ax[7],ax[8])) = plt.subplots(3,3, sharex=True,sharey=True)
        c_endd = 9
        c_start = 0
        if (n==0):
            color=plt.cm.Blues(np.linspace(0.4,1.0,9))
        if (n==1):
            c_endd = 8
            color=plt.cm.Oranges(np.linspace(0.4,1.0,9))
        if (n==2):
            c_endd = 7
            color=plt.cm.Greens(np.linspace(0.4,1.0,9))        
        if (cum == 4):
            c_endd = 8
            c_start = 1

        allcolor=['#0000cc','#cc0000','#006600']

        sigma_x = (etalist[1] - etalist[0])/2
        centlist = np.array([0,5,10,20,30,40,50,60,70,80])

        fon=14
        allcolor = [C0,C1,C2]

        sys = np.zeros((9))
        c_end = c_endd
        barlow = np.zeros((10,9))


        f.set_figwidth(8)
        f.set_figheight(4)
        f.subplots_adjust(wspace=0.05,hspace=0.3)
        for c in range(0,7):
            if (c < c_end and c >= c_start):
                value = np.divide(vn_upper[:,c],vn_lower[:,c],where=vn_lower[:,c]>0)

                value, outliers = self.detect_outlier(value,3)
                barlow[:,c]=np.divide(np.fabs(vn_lower[:,c]-vn_upper[:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[:,c],2)-np.power(vn_upper_e[:,c],2))))

                errorboxes = []
                error = np.divide(vn_upper_e[:,c],vn_lower[:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                residual, fit_out = self.fitconst(etalist, value,error)

                error[value == float('NaN')] = float('NaN')

                weights = np.divide(1, np.float_power(error,2))
                sumweights = np.nansum(weights)
                wx = np.nansum(weights*value)
                weightedmean = np.divide(wx,sumweights)
                upper = np.nansum(weights*np.power(value - weightedmean,2))
                lower = (len(weights) -1)*sumweights/len(weights)
                height = np.divide(upper,lower,where=np.fabs(lower)>0)

                #stds[c] = np.nanstd()
                sys[c] = np.nanstd(value)#np.around(fit_out['c'],3)#height
                rect1 = Rectangle((etalist[0]-sigma_x, 1.0-height/2), etalist[-1]+sigma_x, height)

                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          
                ax[c].errorbar(etalist,value,yerr=error/2,c=color[6],ecolor=color[6],elinewidth=2,markersize=4,fmt='o')

                if (fit_out['c error'] is not None):
                    ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.5,edgecolor=None,facecolor=color[6])
                    ax[c].text(-0.9,1.09,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                
                ax[c].axhline(y=1.0,ls=":", c=".5")
                ax[c].text(-0.9,1.05,r'$\sigma_w='+str(np.around(sys[c],4))+'$',fontsize=12)
            ax[c].set(xticks=np.arange(-1.1,1.1,0.5), yticks=np.arange(0.9,1.11,0.05),xlim=(-1.1, 1.1), ylim=(0.9, 1.11))

            ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
            ax[c].spines['right'].set_visible(False)
            ax[c].spines['top'].set_visible(False)

            ax[c].tick_params(axis='both', which='major', labelsize=fon)
            ax[c].tick_params(axis='both', which='minor', labelsize=fon)

            ax[3].set_ylabel(r'$v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '/v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}$',fontsize=fon+4)

            ax[6].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[7].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[8].set_xlabel(r'$\eta$',fontsize=fon+2)

        plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        fig = plt.figure(figsize=(4.0,2.5))
        for c in range(c_start,c_end):
            plt.errorbar(etalist,barlow[:,c],c=color[c],fmt='o',label=str(centlist[c])+ ' - ' + str(centlist[c+1])+' %')
        plt.gca().set_ylabel(r'$\frac{|v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '- v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}|}{\sqrt{|\sigma^{' + vn_upper_label + ',2}' + '- \sigma^{' + vn_lower_label + ',2}|}} $',fontsize=22)
        plt.gca().tick_params(axis='both', which='major', labelsize=fon)
        plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
        plt.legend(fontsize=13,bbox_to_anchor=[1.,1.])
        plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'.png',dpi=100,bbox_inches='tight')        
        plt.show()
        return np.array(sys), np.array(barlow)        


    def vnn_fwd_sys(self,vn_upper,vn_upper_e,vn_upper_label,vn_lower,vn_lower_e,vn_lower_label,n,cum,foldername,name):
        etalist=self.etalist
        ax = [None]*9
        f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5]),(ax[6],ax[7],ax[8])) = plt.subplots(3,3, sharex=True,sharey=True)
        c_endd = 9
        c_start = 0
        if (n==0):
            color=plt.cm.Blues(np.linspace(0.4,1.0,9))
        if (n==1):
            c_endd = 8
            color=plt.cm.Oranges(np.linspace(0.4,1.0,9))
        if (n==2):
            c_endd = 7
            color=plt.cm.Greens(np.linspace(0.4,1.0,9))        
        if (cum == 4):
            c_endd = 8
            c_start = 1

        allcolor=['#0000cc','#cc0000','#006600']

        sigma_x = (etalist[1] - etalist[0])/2
        centlist = np.array([0,5,10,20,30,40,50,60,70,80])
        vn_upper[7:21,:] = float('NaN')
        vn_upper_e[7:21,:] = float('NaN')
        vn_lower[7:21,:] = float('NaN')
        vn_lower_e[7:21,:] = float('NaN')
        fon=14
        allcolor = [C0,C1,C2]

        sys = np.zeros((9))
        c_end = c_endd
        barlow = np.zeros((9,34))


        f.set_figwidth(8)
        f.set_figheight(4)
        f.subplots_adjust(wspace=0.05,hspace=0.3)
        for c in range(0,9):
            if (c < c_end and c >= c_start):
                value = np.divide(vn_upper[:,c],vn_lower[:,c],where=vn_lower[:,c]>0)

                value, outliers = self.detect_outlier(value,3)
                barlow[c,:]=(np.divide(np.fabs(vn_lower[:,c]-vn_upper[:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[:,c],2)-np.power(vn_upper_e[:,c],2)))))
                value[7:21] = float('NaN')

                errorboxes = []
                error = np.divide(vn_upper_e[:,c],vn_lower[:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                residual, fit_out = self.fitconst(etalist, value,error)
                error[7:21] = float('NaN')

                error[value == float('NaN')] = float('NaN')

                # weights = np.divide(1, np.float_power(error,2))
                # sumweights = np.nansum(weights)
                # wx = np.nansum(weights*value)
                # weightedmean = np.divide(wx,sumweights)
                # upper = np.nansum(weights*np.power(value - weightedmean,2))
                # lower = (len(weights) -1)*sumweights/len(weights)
                # height = np.divide(upper,lower,where=np.fabs(lower)>0)


                sys[c] = np.nanstd(value)
                rect1 = Rectangle((etalist[0]-sigma_x, 1.0-sys[c]/2), (-etalist[0]+etalist[-1]+sigma_x), sys[c])

                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          
                ax[c].errorbar(etalist,value,yerr=error/2,c=color[6],ecolor=color[6],elinewidth=2,markersize=4,fmt='o')

                if (fit_out['c error'] is not None):
                    ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.5,edgecolor=None,facecolor=color[6])
                    ax[c].text(-0.9,1.09,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                
                ax[c].axhline(y=1.0,ls=":", c=".5")
                ax[c].text(-0.9,1.05,r'$\sigma_w='+str(np.around(sys[c],4))+'$',fontsize=12)
            ax[c].set(xticks=np.arange(-3.5,5.0,2.0), yticks=np.arange(0.9,1.11,0.05),xlim=(-3.5, 5.), ylim=(0.9, 1.11))

            ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
            ax[c].spines['right'].set_visible(False)
            ax[c].spines['top'].set_visible(False)

            ax[c].tick_params(axis='both', which='major', labelsize=fon)
            ax[c].tick_params(axis='both', which='minor', labelsize=fon)

            ax[3].set_ylabel(r'$v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '/v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}$',fontsize=fon+4)

            ax[6].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[7].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[8].set_xlabel(r'$\eta$',fontsize=fon+2)

        plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'_fwd.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/'+name+'sys_'+str(cum)+'n'+str(n)+'_fwd.png',dpi=100,bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        fig = plt.figure(figsize=(4.0,2.5))
        for c in range(0,c_end-c_start):
            plt.errorbar(etalist,barlow[c],c=color[c],fmt='o',label=str(centlist[c])+ ' - ' + str(centlist[c+1])+' %')
        plt.gca().set_ylabel(r'$\frac{|v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_upper_label + '}' + '- v_'+str(n+2)+'\{'+str(cum)+'\}^{' + vn_lower_label + '}|}{\sqrt{|\sigma^{' + vn_upper_label + ',2}' + '- \sigma^{' + vn_lower_label + ',2}|}} $',fontsize=22)
        plt.gca().tick_params(axis='both', which='major', labelsize=fon)
        plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
        plt.legend(fontsize=13,bbox_to_anchor=[1.,1.])
        plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'_fwd.pdf',bbox_inches='tight')
        plt.savefig(foldername +'/'+name+'barlow_'+str(cum)+'n'+str(n)+'_fwd.png',dpi=100,bbox_inches='tight')        
        plt.show()

        return sys, np.array(barlow)                

    # def rnnsys(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername,name):

    #     sigma_x = (etalist[1] - etalist[0])/2
    #     centlist = np.linspace(0,100,11)

    #     color=plt.cm.viridis(np.linspace(0,1,12))
    #     color = sns.color_palette("GnBu_d",120)
    #     fon=14
    #     color = sns.color_palette()

    #     sys = np.zeros((6))

    #     for n in range(0,1):
    #         ax = [None]*6
    #         f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5])) = plt.subplots(2,3, sharex=True,sharey=True)

    #         f.set_figwidth(8)
    #         f.set_figheight(4)
    #         f.subplots_adjust(wspace=0.05,hspace=0.3)
    #         for c in range(0,6):
    #             value = np.divide(vn_upper[:,c],vn_lower[:,c],where=vn_lower[:,c]>0)
    #             value[value > 1.2] = float('NaN')
    #             value[value < 0.8] = float('NaN')
    #             #value, outliers = detect_outlier(value)
    #             print value
    #             errorboxes = []
    #             error = np.divide(vn_upper_e[:,c],vn_lower[:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
    #             error[value == float('NaN')] = float('NaN')
                
    #             print 'value',value
    #             print 'error',error
    #             weights = np.divide(1, np.float_power(error,2))
    #             sumweights = np.nansum(weights)
    #             wx = np.nansum(weights*value)
    #             weightedmean = np.divide(wx,sumweights)
    #             upper = np.nansum(weights*np.power(value - weightedmean,2))
    #             lower = (len(weights) -1)*sumweights/len(weights)
    #             height = np.divide(upper,lower,where=np.fabs(lower)>0)
                
                
    #             #height = gen.weightedStd(value,error,0)

    #             sys[c] = height
    #             rect1 = Rectangle((etalist[0]-sigma_x, 1.0-height/2), etalist[-1]+sigma_x, height)

    #             errorboxes.append(rect1)
    #             pc = PatchCollection(errorboxes, facecolor='grey',
    #                              edgecolor='grey',alpha=0.4)
    #             ax[c].add_collection(pc)          


    #             ax[c].errorbar(etalist,value,yerr=error/2,c=color[n],ecolor=color[n],elinewidth=2,markersize=4,fmt='o')

    #             ax[c].set(xticks=np.arange(0.0,1.0,0.2), yticks=np.arange(0.9,1.11,0.05),xlim=(0.0, 1.0), ylim=(0.9, 1.11))

    #             ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
    #             ax[c].spines['right'].set_visible(False)
    #             ax[c].spines['top'].set_visible(False)
    #             ax[c].axhline(y=1.0,ls=":", c=".5")

    #             ax[c].tick_params(axis='both', which='major', labelsize=fon)
    #             ax[c].tick_params(axis='both', which='minor', labelsize=fon)

    #         ax[0].set_ylabel(r'$r_{2,2}' + extra_label +'^{' + vn_upper_label + '}' + '/r_{2,2}' + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
    #         ax[3].set_ylabel(r'$r_{2,2}' + extra_label +'^{' + vn_upper_label + '}' + '/r_{2,2}' + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)

    #         ax[3].set_xlabel(r'$\eta$',fontsize=fon+2)
    #         ax[4].set_xlabel(r'$\eta$',fontsize=fon+2)
    #         ax[5].set_xlabel(r'$\eta$',fontsize=fon+2)

    #         plt.savefig(foldername +'sys_'+name+'.pdf',bbox_inches='tight')
    #         plt.savefig(foldername +'sys_'+name+'.png',dpi=100,bbox_inches='tight')
    #         plt.show()
    #         plt.clf()
    #         plt.close()
    #     return sys




    def sysalleta(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername):
        sys = np.zeros((3,3,6))
        color=plt.cm.viridis(np.linspace(0,1,12))
        color = sns.color_palette("GnBu_d",120)
        fon=14
        color = sns.color_palette()
        sigma_x = (etalist[19] - etalist[18])/2
        centlist = np.linspace(0,100,11)
        date = time.strftime("%d-%m-%Y")

        for n in range(0,3):
            ax = [None]*9
            f, ((ax[0], ax[1]),(ax[2],ax[3]),(ax[4],ax[5])) = plt.subplots(3,2, sharex=True,sharey=True)

            f.set_figwidth(8)
            f.set_figheight(7)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,6):
                value = np.divide(vn_upper[n][:,c],vn_lower[n][:,c],where=vn_lower[n][:,c]>0)
                value,outliers = detect_outlier(value)
                print outliers

                errorboxes = []

                height = np.nanstd(value[4:12])
                sys[n,0,c] = height
                rect = Rectangle((etalist[4]-sigma_x, 1.0-height/2), -etalist[4]+etalist[12], height)

                height = np.nanstd(value[14:26])
                sys[n,1,c] = height
                rect1 = Rectangle((etalist[14]-sigma_x, 1.0-height/2), -etalist[14]+etalist[26], height)

                height = np.nanstd(value[28:45])
                sys[n,2,c] = height
                rect2 = Rectangle((etalist[28]-sigma_x, 1.0-height/2), -etalist[28]+etalist[45], height)

                errorboxes.append(rect)
                errorboxes.append(rect1)
                errorboxes.append(rect2)
                
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)                    
                error = np.divide(vn_upper_e[n][:,c],vn_lower[n][:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))

                #error = value*np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                error[value == np.nan] = float('NaN')
                ax[c].errorbar(etalist,value,yerr=error,c=color[n],ecolor=color[n],elinewidth=2,markersize=4,fmt='o')

                if (n == 0):
                    ax[c].set(xticks=np.arange(-4,6.1,2), yticks=np.arange(0.8,1.21,0.1),xlim=(-4, 6), ylim=(0.8, 1.21))
                if (n == 1):
                    ax[c].set(xticks=np.arange(-4,6.1,2), yticks=np.arange(0.4,1.6,0.2),xlim=(-4, 6), ylim=(0.4, 1.61))
                if (n==2):
                    ax[c].set(xticks=np.arange(-4,6.1,2), yticks=np.arange(0.0,2.1,0.5),xlim=(-4, 6), ylim=(0.0, 2.1))
                
                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' \%',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].axhline(y=1.0,ls=":", c=".5")

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)
            ax[0].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
            ax[2].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
            ax[4].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)


            ax[4].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[5].set_xlabel(r'$\eta$',fontsize=fon+2)

            plt.savefig(foldername + date+'sys_n'+str(n)+'.pdf')
            plt.savefig(foldername + date+'sys_n'+str(n)+'.png',dpi=100)
            plt.show()
        return sys



    def syscentraleta(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername):
        sigma_x = (etalist[19] - etalist[18])/2
        centlist = np.linspace(0,100,11)
        date = time.strftime("%d-%m-%Y")

        color=plt.cm.viridis(np.linspace(0,1,12))
        color = sns.color_palette("GnBu_d",120)
        fon=14
        color = sns.color_palette()

        sys = np.zeros((3,3,6))

        for n in range(0,3):
            ax = [None]*9
            f, ((ax[0], ax[1]),(ax[2],ax[3]),(ax[4],ax[5])) = plt.subplots(3,2, sharex=True,sharey=True)

            f.set_figwidth(8)
            f.set_figheight(7)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,6):
                value = np.divide(vn_upper[n][:,c],vn_lower[n][:,c],where=vn_lower[n][:,c]>0)
                print np.array(value[14:26]).shape

                value, outliers = detect_outlier(value)
                print np.array(value[14:26]).shape
                
                
                errorboxes = []
                error = np.divide(vn_upper_e[n][:,c],vn_lower[n][:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                error[value == float('NaN')] = float('NaN')
                height = np.nanstd(value[14:26])

                sys[n,1,c] = height
                rect1 = Rectangle((etalist[14]-sigma_x, 1.0-height/2), -etalist[14]+etalist[26], height)

                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          
                
                print etalist[14:26].shape
                print np.array(value[14:26]).shape
                print np.array(error[14:26]).shape

                ax[c].errorbar(etalist[14:26],value[14:26],yerr=error[14:26],c=color[n],ecolor=color[n],elinewidth=2,markersize=4,fmt='o')

                if (n == 0):
                    ax[c].set(xticks=np.arange(-1.5,1.6,0.5), yticks=np.arange(0.8,1.21,0.1),xlim=(-1.5, 1.5), ylim=(0.9, 1.11))
                if (n == 1):
                    ax[c].set(xticks=np.arange(-1.5,1.6,0.5), yticks=np.arange(0.4,1.6,0.2),xlim=(-1.5, 1.5), ylim=(0.8, 1.21))
                if (n==2):
                    ax[c].set(xticks=np.arange(-1.5,1.6,0.5), yticks=np.arange(0.0,2.1,0.5),xlim=(-1.5, 1.5), ylim=(0.5, 1.51))
                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' \%',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].axhline(y=1.0,ls=":", c=".5")

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)
                
            ax[0].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
            ax[2].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
            ax[4].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)

            ax[4].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[5].set_xlabel(r'$\eta$',fontsize=fon+2)

            plt.savefig(foldername +date+'sys'+str(n)+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +date+'sys'+str(n)+'.png',dpi=100,bbox_inches='tight')
            plt.show()
        return sys
        
        
        
        

    def detect_outlier(self,data_1,threshold):
        outliers=[]

        for y in range (0,len(data_1)):
            mean_1 = np.nanmean(data_1)
            std_1 =np.nanstd(data_1)
            if (data_1[y] != np.nan):
                z_score= (data_1[y] - mean_1)/std_1 
                if np.abs(z_score) > threshold:
                    outliers.append(y)
                    data_1[y] = float('NaN')
                    print 'found outlier'

        return data_1,outliers



    def syscentraleta_sym(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername):
        #sys_symint = syscentraleta(r'\{2\}',r'normal',r'sym.',vn_tpc,vn_tpc_e,vn_tpc_sym,vn_tpc_sym_e,eta,'../plots/systematics/')
        sigma_x = (etalist[19] - etalist[18])/2
        centlist = np.linspace(0,100,11)
        date = time.strftime("%d-%m-%Y")

        color=plt.cm.viridis(np.linspace(0,1,12))
        color = sns.color_palette("GnBu_d",120)
        fon=14
        color = sns.color_palette()

        sys = np.zeros((3,3,6))

        for n in range(0,3):
            ax = [None]*9
            f, ((ax[0], ax[1]),(ax[2],ax[3])) = plt.subplots(2,2, sharex=True,sharey=True)

            f.set_figwidth(8)
            f.set_figheight(6)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,4):
                value = np.divide(vn_upper[n][:,c],vn_lower[n][:,c],where=vn_lower[n][:,c]>0)
                
                errorboxes = []
                error = np.divide(vn_upper_e[n][:,c],vn_lower[n][:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                error[value == float('NaN')] = float('NaN')
                error[value > 1.2] = float('NaN')
                value[value > 1.2] = float('NaN')
                error[value < 0.9] = float('NaN')
                value[value < 0.9] = float('NaN')            
                value, outliers = detect_outlier(value)

                
                height = np.nanstd(value[14:26])

                sys[n,1,c] = height
                rect1 = Rectangle((etalist[14]-sigma_x, 1.0-height/2), -etalist[14]+etalist[26], height)

                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          
                

                ax[c].errorbar(etalist[14:26],value[14:26],yerr=error[14:26],c=color[n],ecolor=color[n],elinewidth=2,markersize=4,fmt='o')

                if (n == 0):
                    ax[c].set(xticks=np.arange(-1.5,1.6,0.5), yticks=np.arange(0.9,1.11,0.05),xlim=(-1.5, 1.5), ylim=(0.9, 1.11))
                if (n == 1):
                    ax[c].set(xticks=np.arange(-1.5,1.6,0.5), yticks=np.arange(0.8,1.21,0.1),xlim=(-1.5, 1.5), ylim=(0.8, 1.21))
                if (n==2):
                    ax[c].set(xticks=np.arange(-1.5,1.6,0.5), yticks=np.arange(0.5,1.51,0.25),xlim=(-1.5, 1.5), ylim=(0.5, 1.51))
                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' \%',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].axhline(y=1.0,ls=":", c=".5")

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)
                
            ax[0].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
            ax[2].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)

            ax[2].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[3].set_xlabel(r'$\eta$',fontsize=fon+2)

            plt.savefig(foldername +date+'sys'+str(n)+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +date+'sys'+str(n)+'.png',dpi=100,bbox_inches='tight')
            plt.show()
        return sys
        
    def sysalleta_sym(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername):
        #sys_symint = sysalleta(r'\{2\}',r'normal',r'sym.',vn_fmd,vn_fmd_e,vn_fmd_sym,vn_fmd_sym_e,eta,'../plots/systematics/')
        sys = np.zeros((3,3,6))
        color=plt.cm.viridis(np.linspace(0,1,12))
        color = sns.color_palette("GnBu_d",120)
        fon=14
        color = sns.color_palette()
        sigma_x = (etalist[19] - etalist[18])/2
        centlist = np.linspace(0,100,11)
        date = time.strftime("%d-%m-%Y")

        for n in range(0,3):
            ax = [None]*4
            f, ((ax[0], ax[1]),(ax[2],ax[3])) = plt.subplots(2,2, sharex=True,sharey=True)

            f.set_figwidth(8.5)
            f.set_figheight(5.5)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,4):
                value = np.divide(vn_upper[n][:,c],vn_lower[n][:,c],where=vn_lower[n][:,c]>0)
                value,outliers = detect_outlier(value)

                errorboxes = []

                height = np.nanstd(value[4:12])
                sys[n,0,c] = height
                rect = Rectangle((etalist[4]-sigma_x, 1.0-height/2), -etalist[4]+etalist[12], height)

                height = np.nanstd(value[14:26])
                sys[n,1,c] = height
                rect1 = Rectangle((etalist[14]-sigma_x, 1.0-height/2), -etalist[14]+etalist[26], height)

                height = np.nanstd(value[28:45])
                sys[n,2,c] = height
                rect2 = Rectangle((etalist[28]-sigma_x, 1.0-height/2), -etalist[28]+etalist[45], height)

                errorboxes.append(rect)
                errorboxes.append(rect1)
                errorboxes.append(rect2)
                
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)                    
                error = np.divide(vn_upper_e[n][:,c],vn_lower[n][:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))

                #error = value*np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                error[value == np.nan] = float('NaN')
                ax[c].errorbar(etalist,value,yerr=error,c=color[n],ecolor=color[n],elinewidth=2,markersize=4,fmt='o')

                if (n == 0):
                    ax[c].set(xticks=np.arange(-4,6.1,2), yticks=np.arange(0.8,1.21,0.1),xlim=(-4, 6), ylim=(0.8, 1.21))
                if (n == 1):
                    ax[c].set(xticks=np.arange(-4,6.1,2), yticks=np.arange(0.4,1.6,0.2),xlim=(-4, 6), ylim=(0.4, 1.61))
                if (n==2):
                    ax[c].set(xticks=np.arange(-4,6.1,2), yticks=np.arange(0.0,2.1,0.5),xlim=(-4, 6), ylim=(0.0, 2.1))
                
                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' \%',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].axhline(y=1.0,ls=":", c=".5")

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)
            ax[0].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)
            ax[2].set_ylabel(r'$v_' + str(n+2) + extra_label +'^{' + vn_upper_label + '}' + '/v_' + str(n+2) + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+2)

            ax[2].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[3].set_xlabel(r'$\eta$',fontsize=fon+2)

            plt.savefig(foldername + date+'sys_n'+str(n)+'.pdf')
            plt.savefig(foldername + date+'sys_n'+str(n)+'.png',dpi=100)
            plt.show()
        return sys



    def fitconst(self,x, data, err):
        params = Parameters()
        params.add('c', value=1.0, min=0.95,max=1.95)
        minner = Minimizer(self.const, params, fcn_args=(data,err),nan_policy='omit')
        result = minner.minimize()

        # calculate final result
        xfit = np.linspace(x[0],x[-1],1000)
        
        residual = np.divide(np.fabs(data - result.params['c'].value),err,where=err>0,out=np.zeros_like(data))
        
        fit_out = {'c' : result.params['c'].value,'c error' : result.params['c'].stderr, \
                   'reduced chi' : result.redchi}
        #report_fit(result)
        return residual, fit_out


    def const(self,params, data,err):
        return np.power((params['c'] - data)/err,2)

    def rnnsys(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername,name,mode,n):
        sys = 0
        if (n==0):
            sys = self.r22sys(extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername,name,mode)
        if (n==1):
            sys=self.r33sys(extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername,name,mode)
        return sys

    def r33sys(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername,name,mode):
        #syscolor=['#999999','#ffcccc','#9999ff','#99cc99']
        allcolor=['#000000','#0000cc','#cc0000','#006600']
        
        sigma_x = (etalist[1] - etalist[0])/2
        centlist = np.linspace(0,100,11)

        fon=14

            
        sys = np.zeros((3))
        barlow = []
        for n in range(0,1):
            ax = [None]*6
            f, ((ax[0], ax[1],ax[2])) = plt.subplots(1,3, sharex=True,sharey=True)

            f.set_figwidth(8)
            f.set_figheight(2)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,3):
                value = np.divide(vn_upper[:,c+1],vn_lower[:,c+1],where=vn_lower[:,c+1]>0)
                
                #print 'barlow', barlow
                value, outliers = self.detect_outlier(value,3)

                vn_lower[:,c][value<0.8] = float('NaN')
                vn_upper[:,c][value<0.8] = float('NaN')
                vn_lower_e[:,c][value<0.8] = float('NaN')
                vn_upper_e[:,c][value<0.8] = float('NaN')       
                
                vn_lower[:,c][value>1.2] = float('NaN')
                vn_upper[:,c][value>1.2] = float('NaN')
                vn_lower_e[:,c][value>1.2] = float('NaN')
                vn_upper_e[:,c][value>1.2] = float('NaN')      

                barlow.append(np.divide(np.fabs(vn_lower[:,c]-vn_upper[:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[:,c],2)-np.power(vn_upper[:,c],2)))))
                
                #value[value > 1.2] = float('NaN')
                #value[value < 0.8] = float('NaN')
                #value, outliers = detect_outlier(value)
                errorboxes = []
                error = np.divide(vn_upper_e[:,c],vn_lower[:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                residual, fit_out = self.fitconst(etalist, value,error)
                
                error[value == float('NaN')] = float('NaN')
                
                #print 'value',value
                #print 'error',error
                weights = np.divide(1, np.float_power(error,2))
                sumweights = np.nansum(weights)
                wx = np.nansum(weights*value)
                weightedmean = np.divide(wx,sumweights)
                upper = np.nansum(weights*np.power(value - weightedmean,2))
                lower = (len(weights) -1)*sumweights/len(weights)
                height = np.divide(upper,lower,where=np.fabs(lower)>0)
                
                
                #height = gen.weightedStd(value,error,0)

                sys[c] = height
                rect1 = Rectangle((etalist[0]-sigma_x, 1.0-height/2), etalist[-1]+sigma_x, height)

                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          


                ax[c].errorbar(etalist,value,yerr=error/2,c=allcolor[mode],ecolor=allcolor[mode],elinewidth=2,markersize=4,fmt='o')
                
                #ax[c].errorbar(etalist,barlow,c='black',ecolor=color[n],elinewidth=2,markersize=4,fmt='o')
                print fit_out['c error']
                if (fit_out['c'] == None):
                    fit_out['c'] = 1
                if (fit_out['c error'] == None):
                    fit_out['c error'] = 0                   
                ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.3,edgecolor=None,facecolor=allcolor[mode])

                ax[c].set(xticks=np.arange(0.0,1.0,0.2), yticks=np.arange(0.9,1.11,0.05),xlim=(0.0, 1.0), ylim=(0.9, 1.11))

                ax[c].set_title(str(centlist[c+1]) + ' - ' + str(centlist[c+2]) + ' %',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].axhline(y=1.0,ls=":", c=".5")

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)
                ax[c].text(0.2,1.1,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                ax[c].text(0.2,1.07,r'$\chi^2_\nu='+str(np.around(fit_out['reduced chi'],3))+'$',fontsize=12)
                ax[c].text(0.2,1.04,r'$\sigma_w='+str(np.around(sys[c],4))+'$',fontsize=12)

                
            ax[0].set_ylabel(r'$r_{3,3}' + extra_label +'^{' + vn_upper_label + '}' + '/r_{3,3}' + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+4)

            ax[0].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[1].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[2].set_xlabel(r'$\eta$',fontsize=fon+2)
            
            if (mode==0):
                color=plt.cm.Greys(np.linspace(0.4,1.0,6))
                mytitle = r'$2.0 < \eta_{ref} < 3.5$'
            if (mode==1):
                color=plt.cm.Blues(np.linspace(0.4,1.0,6))
                mytitle = r'$2.0 < \eta_{ref} < 2.5$'
                
            if (mode==2):
                color=plt.cm.Reds(np.linspace(0.4,1.0,6))
                mytitle = r'$2.5 < \eta_{ref} < 3.0$'
            if (mode==3):
                color=plt.cm.Greens(np.linspace(0.4,1.0,6))        
                mytitle = r'$3.0 < \eta_{ref} < 3.5$'
            
            plt.gcf().suptitle(mytitle,y=1.25,fontsize=fon+4)
            plt.savefig(foldername +'sys_'+name+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'sys_'+name+'.png',dpi=100,bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.close()
            
            #syscolor=['#999999','#ffcccc','#9999ff','#99cc99']
            #color=['#000000','#cc0000','#0000cc', '#006600']
            #color = plt.colormaps

            

            fig = plt.figure(figsize=(4.0,2.5))
            plt.gca().set_title(mytitle,fontsize=fon)
            for c in range(0,3):
                plt.errorbar(etalist,barlow[c],c=color[c],fmt='o',label=str(centlist[c+1])+ ' - ' + str(centlist[c+2])+' %')
            plt.gca().set_ylabel(r'$\frac{|r_{3,3}' + extra_label +'^{' + vn_upper_label + '}' + '- r_{3,3}' + extra_label +'^{' + vn_lower_label + '}|}{\sqrt{|\sigma' + extra_label +'^{' + vn_upper_label + ',2}' + '- \sigma' + extra_label +'^{' + vn_lower_label + ',2}|}} $',fontsize=22)
            plt.gca().tick_params(axis='both', which='major', labelsize=fon)
            plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
            plt.legend(fontsize=12)
            plt.savefig(foldername +'barlow_'+name+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'barlow_'+name+'.png',dpi=100,bbox_inches='tight')        
            plt.show()
            
            
        return sys, np.array(barlow)


    def r22sys(self,extra_label,vn_upper_label,vn_lower_label,vn_upper,vn_upper_e,vn_lower,vn_lower_e,etalist,foldername,name,mode):
        #syscolor=['#999999','#ffcccc','#9999ff','#99cc99']
        allcolor=['#000000','#0000cc','#cc0000','#006600']
        
        sigma_x = (etalist[1] - etalist[0])/2
        centlist = np.linspace(0,100,11)

        fon=14

            
        sys = np.zeros((6))
        barlow = []
        for n in range(0,1):
            ax = [None]*6
            f, ((ax[0], ax[1],ax[2]),(ax[3],ax[4],ax[5])) = plt.subplots(2,3, sharex=True,sharey=True)

            f.set_figwidth(8)
            f.set_figheight(4)
            f.subplots_adjust(wspace=0.05,hspace=0.3)
            for c in range(0,6):
                value = np.divide(vn_upper[:,c],vn_lower[:,c],where=vn_lower[:,c]>0)
                
                vn_lower[:,c][value<0.8] = float('NaN')
                vn_upper[:,c][value<0.8] = float('NaN')
                vn_lower_e[:,c][value<0.8] = float('NaN')
                vn_upper_e[:,c][value<0.8] = float('NaN')       
                
                vn_lower[:,c][value>1.2] = float('NaN')
                vn_upper[:,c][value>1.2] = float('NaN')
                vn_lower_e[:,c][value>1.2] = float('NaN')
                vn_upper_e[:,c][value>1.2] = float('NaN')                      
                value[value > 1.2] = float('NaN')
                value[value < 0.8] = float('NaN')
                value, outliers = self.detect_outlier(value,3)
                barlow.append(np.divide(np.fabs(vn_lower[:,c]-vn_upper[:,c]),np.sqrt(np.fabs(np.power(vn_lower_e[:,c],2)-np.power(vn_upper[:,c],2)))))

                errorboxes = []
                error = np.divide(vn_upper_e[:,c],vn_lower[:,c])#np.sqrt(np.power(np.divide(vn_upper_e[n][:,c],vn_upper[n][:,c]),2) + np.power(np.divide(vn_lower_e[n][:,c],vn_lower[n][:,c]),2))
                residual, fit_out = self.fitconst(etalist, value,error)
                
                error[value == float('NaN')] = float('NaN')
                
                #print 'value',value
                #print 'error',error
                weights = np.divide(1, np.float_power(error,2))
                sumweights = np.nansum(weights)
                wx = np.nansum(weights*value)
                weightedmean = np.divide(wx,sumweights)
                upper = np.nansum(weights*np.power(value - weightedmean,2))
                lower = (len(weights) -1)*sumweights/len(weights)
                height = np.divide(upper,lower,where=np.fabs(lower)>0)
                
                
                #height = gen.weightedStd(value,error,0)

                sys[c] = height
                rect1 = Rectangle((etalist[0]-sigma_x, 1.0-height/2), etalist[-1]+sigma_x, height)

                errorboxes.append(rect1)
                pc = PatchCollection(errorboxes, facecolor='grey',
                                 edgecolor='grey',alpha=0.4)
                ax[c].add_collection(pc)          


                ax[c].errorbar(etalist,value,yerr=error/2,c=allcolor[mode],ecolor=allcolor[mode],elinewidth=2,markersize=4,fmt='o')
                
                #ax[c].errorbar(etalist,barlow,c='black',ecolor=color[n],elinewidth=2,markersize=4,fmt='o')
                
                ax[c].fill_between(etalist,np.repeat(fit_out['c']+fit_out['c error'],len(etalist)),np.repeat(fit_out['c']-fit_out['c error'],len(etalist)),alpha=0.3,edgecolor=None,facecolor=allcolor[mode])

                ax[c].set(xticks=np.arange(0.0,1.0,0.2), yticks=np.arange(0.9,1.11,0.05),xlim=(0.0, 1.0), ylim=(0.9, 1.11))

                ax[c].set_title(str(centlist[c]) + ' - ' + str(centlist[c+1]) + ' %',fontsize=fon)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].axhline(y=1.0,ls=":", c=".5")

                ax[c].tick_params(axis='both', which='major', labelsize=fon)
                ax[c].tick_params(axis='both', which='minor', labelsize=fon)
                ax[c].text(0.2,1.1,r'$y='+str(np.around(fit_out['c'],3))+'\pm'+str(np.around(fit_out['c error'],3)) +'$',fontsize=12)
                ax[c].text(0.2,1.07,r'$\chi^2_\nu='+str(np.around(fit_out['reduced chi'],3))+'$',fontsize=12)
                ax[c].text(0.2,1.04,r'$\sigma_w='+str(np.around(sys[c],4))+'$',fontsize=12)

                
            ax[0].set_ylabel(r'$r_{2,2}' + extra_label +'^{' + vn_upper_label + '}' + '/r_{2,2}' + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+4)
            ax[3].set_ylabel(r'$r_{2,2}' + extra_label +'^{' + vn_upper_label + '}' + '/r_{2,2}' + extra_label +'^{' + vn_lower_label + '}$',fontsize=fon+4)

            ax[3].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[4].set_xlabel(r'$\eta$',fontsize=fon+2)
            ax[5].set_xlabel(r'$\eta$',fontsize=fon+2)
            
            if (mode==0):
                color=plt.cm.Greys(np.linspace(0.4,1.0,6))
                mytitle = r'$2.0 < \eta_{ref} < 3.5$'
            if (mode==1):
                color=plt.cm.Blues(np.linspace(0.4,1.0,6))
                mytitle = r'$2.0 < \eta_{ref} < 2.5$'
                
            if (mode==2):
                color=plt.cm.Reds(np.linspace(0.4,1.0,6))
                mytitle = r'$2.5 < \eta_{ref} < 3.0$'
            if (mode==3):
                color=plt.cm.Greens(np.linspace(0.4,1.0,6))        
                mytitle = r'$3.0 < \eta_{ref} < 3.5$'
            
            plt.gcf().suptitle(mytitle,y=1.1,fontsize=fon+4)
            plt.savefig(foldername +'sys_'+name+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'sys_'+name+'.png',dpi=100,bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.close()
            
            #syscolor=['#999999','#ffcccc','#9999ff','#99cc99']
            #color=['#000000','#cc0000','#0000cc', '#006600']
            #color = plt.colormaps

            

            fig = plt.figure(figsize=(4.0,2.5))
            plt.gca().set_title(mytitle,fontsize=fon)
            for c in range(0,6):
                plt.errorbar(etalist,barlow[c],c=color[c],fmt='o',label=str(centlist[c])+ ' - ' + str(centlist[c+1])+' %')
            plt.gca().set_ylabel(r'$\frac{|r_{2,2}' + extra_label +'^{' + vn_upper_label + '}' + '- r_{2,2}' + extra_label +'^{' + vn_lower_label + '}|}{\sqrt{|\sigma' + extra_label +'^{' + vn_upper_label + ',2}' + '- \sigma' + extra_label +'^{' + vn_lower_label + ',2}|}} $',fontsize=22)
            plt.gca().tick_params(axis='both', which='major', labelsize=fon)
            plt.gca().set_xlabel(r'$\eta$',fontsize=fon)
            plt.legend(fontsize=12)
            plt.savefig(foldername +'barlow_'+name+'.pdf',bbox_inches='tight')
            plt.savefig(foldername +'barlow_'+name+'.png',dpi=100,bbox_inches='tight')        
            plt.show()
            
            
        return sys, np.array(barlow)



    def calcrnn_forwardside(self,n):


        dif = self.differential[n,...]

        rnn = np.zeros_like(self.differential[n,:,:,:,14:20,:,dW2TwoA])
        N = dif[:,:,:,14:20,:,dW2TwoB]/dif[:,:,:,14:20,:,dW2B]
        D = dif[:,:,:,20:26,:,dW2TwoA]/dif[:,:,:,20:26,:,dW2A]
        rnn = np.divide(np.flip(N,3),D,where=D!=0)


        rnn_sigma1 = np.nanstd(rnn, axis=2)
        rnn_mu1 = np.nanmean(rnn, axis=2)
        #print rnn_mu1.shape
        rnn_mu1,rnn_sigma1 = gen.weighted_rebin(rnn_mu1, rnn_sigma1,(1,1,6,10))
        #rnn_mu,rnn_sigma = gen.weighted_rebin(rnn_mu1[0,...], rnn_sigma1[0,...],(1,6,6))
        rnn_mu = rnn_mu1[0,0,...]
        rnn_sigma = rnn_sigma1[0,0,...]

        rnnA = rnn_mu[0:5]
        rnnA_e = rnn_sigma[0:5]


        return rnnA, rnnA_e




    def calcrnn_backwardside(self,n):

        dif = self.differential[n,...]

        rnn = np.zeros_like(self.differential[n,:,:,:,14:20,:,dW2TwoA])
        N = dif[:,:,:,14:20,:,dW2TwoA]/dif[:,:,:,14:20,:,dW2B]
        D = dif[:,:,:,20:26,:,dW2TwoB]/dif[:,:,:,20:26,:,dW2A]
        rnn = np.divide(np.flip(N,3),D,where=D!=0)

        rnn_sigma1 = np.nanstd(rnn, axis=2)
        rnn_mu1 = np.nanmean(rnn, axis=2)
        #print rnn_mu1.shape
        rnn_mu1,rnn_sigma1 = gen.weighted_rebin(rnn_mu1, rnn_sigma1,(1,1,6,10))
        #rnn_mu,rnn_sigma = gen.weighted_rebin(rnn_mu1[0,...], rnn_sigma1[0,...],(1,6,6))
        rnn_mu = rnn_mu1[0,0,...]
        rnn_sigma = rnn_sigma1[0,0,...]

        rnnB = 1/rnn_mu[0:5]
        rnnB_e = rnn_sigma[0:5]

        return rnnB, rnnB_e
