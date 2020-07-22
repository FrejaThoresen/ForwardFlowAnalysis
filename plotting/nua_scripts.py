import helpers.general as gen
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helpers.constants import *
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from plottingScripts import *
from root_numpy import hist2array
from rootpy.io import root_open


def nua_vertex(vertex):

	vertex = vertex.rebinned(10,axis=0)
	test = hist2array(vertex,return_edges=True)

	norm = vertex.Integral()

	y = test[0]
	z=y[10:20]

	x_boundary = test[1][0][1:]
	x_corr = (x_boundary[2] - x_boundary[1])/2
	x_center = x_boundary[:] - x_corr
	x_width = x_boundary[2] - x_boundary[1];

	fig = plt.figure(figsize=(6, 4))
	axes = plt.axes()
	plt.bar(x_center,y,width=x_width,color=C1,label='Vertex Position')

	axes.legend(loc="upper left", columnspacing=-0.1,handletextpad=0.1)
	axes.set_xlabel("Vertex Position [cm]",fontsize=18)
	axes.set_ylabel("Count",fontsize=18)
	#axes.yaxis.set_major_locator(MultipleLocator(50000))
	#axes.yaxis.set_minor_locator(AutoMinorLocator())
	axes.xaxis.set_major_locator(MultipleLocator(2))
	#axes.xaxis.set_minor_locator(AutoMinorLocator())
	axes.tick_params(which='both',direction='in',top='on',right='on')
	#axes.set_ylim([0,200000])
	axes.set_xlim([-10.0,10.0])
	axes.set_title('Vertex distribution after selection',fontsize=20)
	entries = vertex.GetEntries()
	mu = vertex.GetMean()
	sigma = vertex.GetStdDev()
	textstr = '$\mathrm{Entries}=%.d$\n$\mu=%.2f$\n$\sigma=%.2f$'%(entries, mu, sigma)

	# these are matplotlib.patch.Patch properties
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

	# place a text box in upper left in axes coords
	axes.text(0.65, 0.95, textstr, transform=axes.transAxes, fontsize=14,
	        verticalalignment='top', bbox=props)


	fig.savefig('../../plots/nua/vertex_nua.png')
	return z

def nua_raw(awesome,hist):
	nuahist = awesome.FindObject(hist)

	Zinit = hist2array(nuahist)
	Zinit[Zinit == 0] = np.nan

	return Zinit

def nua_norm(Zinit,z):
	Zinit[np.isnan(Zinit)] = 0
	Znorm =Zinit/z

	Znorm[Znorm == 0] = np.nan
	Zinit[Zinit == 0] = np.nan
	return Znorm

def nua_mean(Znorm):
	Znorm[np.isnan(Znorm)] = 0
	Zmean = np.zeros_like(Znorm)

	for i in range(0, len(Znorm)):
	    for j in range(0,10):
	        Zmean[i,:,j] = Znorm[i,:,j].mean()
	Znorm[Znorm == 0] = np.nan
	Zmean[Zmean == 0] = np.nan
	return Zmean

def nua_meanSum(Znorm):
	Znorm[np.isnan(Znorm)] = 0
	Zmean = np.zeros_like(Znorm)

	for i in range(0, len(Znorm)):
		Zmean[i,:] = Znorm[i,:].mean()
	Znorm[Znorm == 0] = np.nan
	Zmean[Zmean == 0] = np.nan
	return Zmean

def nua_frac(Zmean,Znorm):
	Znorm[np.isnan(Znorm)] = 0
	Zmean[np.isnan(Zmean)] = 0

	Znew=np.divide(Zmean,Znorm,out=np.zeros_like(Znorm),where=Znorm>0)
	Znew[Znew == 0.0] = np.nan
	Znew[Znew > 10] = np.nan
	return Znew


def nua_result(Znew, Znorm):
	Znorm[np.isnan(Znorm)] = 0
	Znew[np.isnan(Znew)] = 0

	Zres = Znew*Znorm
	Zres[Zres == 0.0] = np.nan
	return Zres

def myplot(Z, savename, mytitle, ztitle,equation,Zbin,detector):
	ylen = len(Z[0,:])
	xlen = len(Z[:,0])

	if (detector == "TPC"):
		xmin = -1.2
		xmax = 1.2
	elif (detector == "SPD"):
		xmin = -2.5
		xmax = 2.5
	else: #(detector == "ITS" or "FMD):
		xmin = -4.0
		xmax = 6.0 
	xlist = np.linspace(xmin, xmax, xlen+1)
	x_corr = -(abs(xlist[2]) - abs(xlist[1]))/2
	x_center = xlist[:] - x_corr
	#xlist = x_center[1:xlen+1]

	ylist = np.linspace(0, 2*np.pi, ylen+1)
	y_corr = -(abs(xlist[2]) - abs(xlist[1]))/2
	y_center = ylist[:] - y_corr
	#ylist = y_center[1:ylen+1]
	#ylist = makephilist(0,2*np.pi,ylen)

	X,Y = np.mgrid[xlist[0]:xlist[xlen]:complex(0,xlen+1), ylist[0]:ylist[ylen]:complex(0,ylen+1)]

	fig = plt.figure(figsize=(8, 5))
	axes = plt.axes()
	#axes.set_xlim([xmin,xmax])
	#axes.set_ylim([0.0,2*np.pi])
    
	plt.tight_layout()
	#plt.subplots_adjust(right=0.9, top=0.9)
	#if ('frac' in savename):
	#	pcm = axes.pcolor(X, Y, Z,vmin=np.nanmin(Z), vmax=np.nanmax(Z),cmap='summer',norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
	#else:
	pcm = axes.pcolor(X, Y, Z,vmin=np.nanmin(Z), vmax=np.nanmax(Z),cmap='summer')
	
	pcm.cmap.set_under('white')

	axes.set_xlabel("$\eta$")
	axes.set_ylabel(r"$\varphi$")
	axes.yaxis.set_major_locator(MultipleLocator(1))
	axes.yaxis.set_minor_locator(AutoMinorLocator())
	if (detector == 'TPC'):
		axes.xaxis.set_major_locator(MultipleLocator(0.5))
		axes.xaxis.set_minor_locator(AutoMinorLocator())	
	else:
		axes.xaxis.set_major_locator(MultipleLocator(1))
		axes.xaxis.set_minor_locator(AutoMinorLocator())
	#axes.tick_params(which='both',direction='in',top='on',right='on')
	#plt.title(mytitle)
	vertexlist = np.linspace(-10,10,11)
	if (Zbin < 0):
		zztitle = "%.2f" %(vertexlist[0])  + ' cm $\leq \mathrm{IP}_z <$' + "%.2f" %(vertexlist[10]) +' cm'
	else:
		zztitle = "%.2f" %(vertexlist[Zbin])  + ' cm $\leq \mathrm{IP}_z <$' + "%.2f" %(vertexlist[Zbin+1]) +' cm'
	
	fig.text(0.11,0.8,mytitle,fontsize=18)
	fig.text(0.11,0.16,zztitle,fontsize=18)
	cbar = fig.colorbar(pcm, ax=axes, extend='max')
	#if (equation):
	#	cbar.set_label(ztitle, rotation=270,labelpad=40,fontsize=24)
	#else:
	cbar.set_label(ztitle)#, rotation=270)
	plt.grid(which='major')
	fig.text(0.11,0.86,detector,fontsize=18)
	        
	fig.savefig('../../plots/nua/' + savename + '_zbin_' +str(Zbin)+'.png',bbox_inches='tight')
	plt.close(fig)	

    
def makeVertexPlot(awesome,mydata):
    Zinit = hist2array(awesome.FindObject('NUA_cen'))

    z = hist2array(awesome.FindObject('EventInfo').FindObject('Vertex'))

    norm = np.nansum(z)
    x_center = np.linspace(-10,10,11)
    x_center = x_center -1
    x_center = x_center[1:]
    deltax = x_center[1] - x_center[0]
    fig = plt.figure(figsize=(6, 4))
    axes = plt.axes()

    plt.bar(x_center,z/norm,width=2,label='Vertex Position')

    axes.legend(loc="upper left", columnspacing=-0.1,handletextpad=0.1)
    axes.set_xlabel(r'$\mathrm{IP}_z$ [cm]',fontsize=18)
    axes.set_ylabel(r'$\mathrm{d}N/\mathrm{d} z$',fontsize=18)

    axes.set_title('Vertex distribution after selection',fontsize=20)
    plt.savefig('../../plots/nua/' + mydata + '/central/vertex_nua.png',bbox_inches='tight')
    plt.show()
    
    
def makeSumPlot(Zinit,mydata,detector):
    
    Zsum = np.nansum(Zinit, axis=2)
    Zsum[Zsum == 0.] = np.nan
    if (detector == 'FMD'):
        savename = 'forward'
    else:
        savename = 'central'

    myplot(Zsum,mydata + '/' + savename +'/nua_total','', r'$\mathrm{d}N_{ch} / \mathrm{d}\eta \mathrm{d}\varphi$',False,-1,detector)
    plt.show()
    
    
def makeCentralPlots(awesome,Zinit,mydata,cendetector,doedges, doplt):
    histname = 'NUA_cen'
    zbins = 10
    z = hist2array(awesome.FindObject('EventInfo').FindObject('Vertex'))

    Znorm = nua_norm(Zinit,z)
    if (doplt):
        for Zbin in range(0,zbins):
            myplot(Znorm[:,:,Zbin],mydata + '/central/nua_norm','', r'$dN / d\varphi d\eta$',False,Zbin,cendetector)

    for Zbin in range(0,zbins): #10
        priornan = False
        donan = False
        for etabin in range(0,len(Znorm[:,0,0])):
            foundnan = 0
            foundnumber = False
            for phibin in range(0,len(Zinit[0,:,0])):
                if np.isnan(Znorm[etabin,phibin,Zbin]):
                    foundnan = True
                if Znorm[etabin,phibin,Zbin] > 0:
                    foundnumber = True

            if (doedges):
                for phibin in range(0,len(Zinit[0,:,0])):
                    if np.isnan(Znorm[etabin,phibin,Zbin]):
                        foundnan = foundnan + 1
                    if Znorm[etabin,phibin,Zbin] > 0:
                        foundnumber = True
                if ((foundnan > 5) & foundnumber):
                    Znorm[etabin,:,Zbin] = np.nan;
                #     if (priornan == True ):
                #         Znorm[etabin-1,:,Zbin] = np.nan;
                #     if (etabin < 200):
                #         Znorm[etabin-1,:,Zbin] = np.nan;
                #         Znorm[etabin-2,:,Zbin] = np.nan;
                #         Znorm[etabin-3,:,Zbin] = np.nan;

                #     if (etabin>200):
                #         donan = True
                #     priornan = True
                # if (etabin > 200):
                #     priornan = False
                # if (donan):
                #     Znorm[etabin,:,Zbin] = np.nan;
        if (doplt):
            myplot(Znorm[:,:,Zbin],mydata + '/central/nua_norm_wfill','w. cuts', r'$dN_{ch} / d\eta d\varphi$',False,Zbin,cendetector)


    Zmean = nua_mean(Znorm)
    Znew = nua_frac(Zmean,Znorm)
    Zmean[Zmean == 0] = np.nan
    if (doplt):
        for Zbin in range(0,10):
            myplot(Zmean[:,:,Zbin],mydata + '/central/nua_mean_wfill','', r'$\langle N_{ch} \rangle / d\eta$',False,Zbin,cendetector)
            
            myplot(Znew[:,:,Zbin],mydata + '/central/nua_frac','', r'$\langle N \rangle / N$',False,Zbin,cendetector)
        plt.show()
    Zcentral = Znew
    return Zcentral

    
def makeForwardPlots(awesome,Zinit,mydata,doedges,dofill):
    histname = 'NUAforward'
    z = hist2array(awesome.FindObject('EventInfo').FindObject('Vertex'))
    zbins=10

    Znorm = nua_norm(Zinit,z)

    for Zbin in range(0,10):
        myplot(Znorm[:,:,Zbin],mydata + '/forward/nua_norm','', r'$dN / d\varphi d\eta$',False,Zbin,'FMD')

    for Zbin in range(0,10): #10
        for etabin in range(0,200):
            if (histname == 'NUAforward'):
                foundnan = False
                foundnumber = False

                if (dofill):
                    if np.isnan(Znorm[etabin,16,Zbin]):
                        foundnan = True
                    if Znorm[etabin,18:19,Zbin] > 0:
                        foundnumber = True
                    if (foundnan & foundnumber):
                        Znorm[etabin,16,Zbin] = 1; 

                    foundnan = False
                    foundnumber = False
                    if np.isnan(Znorm[etabin,17,Zbin]):
                        foundnan = True
                    if Znorm[etabin,18,Zbin] > 0:
                        foundnumber = True
                    if (foundnan & foundnumber):
                        Znorm[etabin,17,Zbin] = 1;

                    foundnan = False
                    foundnumber = False
                    if np.isnan(Znorm[etabin,13,Zbin]):
                        foundnan = True
                    if Znorm[etabin,14,Zbin] > 0:
                        foundnumber = True
                    if (foundnan & foundnumber):
                        Znorm[etabin,13,Zbin] = 1;

            if (doedges):
                foundnan = False
                foundnumber = False
                for phibin in range(0,len(Zinit[0,:,0])):
                    if (phibin == 16 or phibin == 17 or phibin == 13 or phibin == 14):
                        continue
                    if np.isnan(Znorm[etabin,phibin,Zbin]):
                        foundnan = True
                    if Znorm[etabin,phibin,Zbin] > 0:
                        foundnumber = True
                if (foundnan & foundnumber):
                    Znorm[etabin,:,Zbin] = np.nan;

        myplot(Znorm[:,:,Zbin],mydata + '/forward/nua_norm_wfill','w. fills', r'$dN_{ch} / d\eta d\varphi$',False,Zbin,'FMD')

    Zmean = nua_mean(Znorm)
    Znew = nua_frac(Zmean,Znorm)
    Zmean[Zmean == 0] = np.nan

    for Zbin in range(0,10):
        myplot(Zmean[:,:,Zbin],mydata + '/forward/nua_mean_wfill','', r'$\langle N_{ch} \rangle / d\eta$',False,Zbin,'FMD')
        myplot(Znew[:,:,Zbin],mydata + '/forward/nua_frac','', r'$\langle N \rangle / N$',False,Zbin,'FMD')
    plt.show()

    Zres = nua_result(Znew,Znorm)
    Zforward = Znew

    for Zbin in range(0,10):
        myplot(Zres[:,:,Zbin],mydata + '/forward/nua_res','', r'$\langle N \rangle / N$',False,Zbin,'FMD')
        
    return Zforward




    
def makeForwardPlotsSum(awesome,Zinit,mydata,doedges,dofill):
    histname = 'NUAforward'
    z = hist2array(awesome.FindObject('EventInfo').FindObject('Vertex'))
    zbins=10
    Zinit = np.nansum(Zinit,axis=2)
    z = np.nansum(z)

    Znorm = nua_norm(Zinit,z)

    myplot(Znorm[:,:],mydata + '/forward/nua_norm','', r'$dN / d\varphi d\eta$',False,-1,'FMD')
    myplot(Znorm[:,:],mydata + '/forward/nua_norm_wfill','w. fills', r'$dN_{ch} / d\eta d\varphi$',False,-1,'FMD')

    Zmean = nua_meanSum(Znorm)
    Znew = nua_frac(Zmean,Znorm)
    Zmean[Zmean == 0] = np.nan

    myplot(Zmean[:,:],mydata + '/forward/nua_mean_wfill','', r'$\langle N_{ch} \rangle / d\eta$',False,-1,'FMD')
    myplot(Znew[:,:],mydata + '/forward/nua_frac','', r'$\langle N \rangle / N$',False,-1,'FMD')
    plt.show()

    Zres = nua_result(Znew,Znorm)
    Zforward = Znew

    myplot(Zres[:,:],mydata + '/forward/nua_res','', r'$\langle N \rangle / N$',False,-1,'FMD')
        
    return Zforward



def makeCentralPlotsSum(awesome,Zinit,mydata,doedges, doplt,detector):
    histname = 'NUA_cen'
    zbins = 10
    z = hist2array(awesome.FindObject('EventInfo').FindObject('Vertex'))
    Zinit = np.nansum(Zinit,axis=2)
    z = np.nansum(z)
    
    Znorm = nua_norm(Zinit,z)
    if (doplt):
        myplot(Znorm[:,:],mydata + '/central/nua_norm','', r'$dN / d\varphi d\eta$',False,-1,detector)



    priornan = False
    donan = False
    if (doedges):

        for etabin in range(0,len(Znorm[:,0])):
            foundnan = False
            foundnumber = False

            for phibin in range(0,len(Zinit[0,:])):
                if np.isnan(Znorm[etabin,phibin]):
                    foundnan = True
                if Znorm[etabin,phibin] > 0:
                    foundnumber = True
            if (foundnan & foundnumber):
                Znorm[etabin,:] = float('NaN');


    if (doplt):
        myplot(Znorm[:,:],mydata + '/central/nua_norm_wfill','w. cuts', r'$dN_{ch} / d\eta d\varphi$',False,-1,detector)


    Zmean = nua_meanSum(Znorm)
    Znew = nua_frac(Zmean,Znorm)
    Zmean[Zmean == 0] = np.nan
    if (doplt):
        myplot(Zmean[:,:],mydata + '/central/nua_mean_wfill','', r'$\langle N_{ch} \rangle / d\eta$',False,-1,detector)
        myplot(Znew[:,:],mydata + '/central/nua_frac','', r'$\langle N \rangle / N$',False,-1,detector)
        
        plt.show()
    Zcentral = Znew
    return Zcentral


def nua_per_run(run_list,title,save_dir,inputfile,branch,forward,y_low,y_up,yx_low,yx_up):
    hist_list = []
    colors = sns.color_palette("Paired",20)

    if (forward):
        xlist = gen.makelist(0,2*np.pi,20)
        histname = 'NUA_fwd'
        detector = 'FMD'
    else:
        xlist = gen.makelist(0,2*np.pi,300)
        histname = 'NUA_cen'
        detector = 'TPC'

    
    fig = plt.figure(figsize=(7, 4))
    axes = plt.axes()
    
    for i in range(0,len(run_list)):
        input = inputfile + 'output/' + run_list[i] + '.root'
        myfile = root_open(input, 'read')
        
        awesome = myfile.V0M_globaltracks
        Zinit = nua_raw(awesome,histname)
        hist_list.append(np.array(np.nansum(np.nansum(Zinit, axis=0),axis=1)))
        print hist_list[i].shape
        hist_list[i] = hist_list[i]/np.nansum(hist_list[i])
        axes.plot(xlist,hist_list[i],label=run_list[i],c=colors[i],linewidth=1)

    plt.ylabel(r'Normalized count',fontsize=16)
    plt.xlabel(r'$\varphi$',fontsize=16)
    plt.ylim([y_low,y_up])

    plt.title(r'Runs from '+title,fontsize=14)
    plt.figtext(0.87,0.19,detector,fontsize=16)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, fancybox=True, shadow=True,fontsize=14)
    axis = plt.axes()
    axis.tick_params(which='both',direction='in',labelsize=14)
    plt.savefig(save_dir + 'allruns_'+detector+'.pdf',bbox_inches='tight')
    plt.savefig(save_dir + 'allruns_'+detector+'.png',bbox_inches='tight')
    plt.show()


    fig = plt.figure(figsize=(7, 4))
    axes = plt.axes()
    for i in range(0,len(run_list)):
        axes.plot(xlist,np.divide(hist_list[i],hist_list[0]),label=run_list[i],c=colors[i],linewidth=1)
    plt.ylim([yx_low,yx_up])

    plt.ylabel(r'first run/X',fontsize=16)
    plt.xlabel(r'$\varphi$',fontsize=16)

    axis = plt.axes()
    axis.tick_params(which='both',direction='in',labelsize=14)

    plt.figtext(0.87,0.19,detector,fontsize=16)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, fancybox=True, shadow=True,fontsize=14)
    plt.title(r'Runs from LHC15o\_pass1\_highIR',fontsize=14)
    plt.savefig(save_dir + 'allrunsX_'+detector+'.pdf',bbox_inches='tight')
    plt.savefig(save_dir + 'allrunsX_'+detector+'.png',bbox_inches='tight')

    plt.show()