
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np



def cauchy(params, x, data,err):
	amp = params['amp']
	const = params['const']
	gamma_1 = params['gamma_1']
    
	model = const +  amp* 1/(np.pi*gamma_1*(1+np.power(x/gamma_1,2)))  # + amp
	return np.power((model - data)/err,2)




def cauchy_no_const(params, x, data,err):
    amp = params['amp']
    gamma = params['gamma_1']
    
    model = amp* 1/(np.pi*gamma*(1+np.power(x/gamma,2)))  # + amp
    return np.power((model - data)/err,2)

def fitcauchy(x, data, err):
    params = Parameters()
    params.add('amp', value=0.2, min=0.01,max=1.0)
    params.add('const', value=0.01, min=0.0,max=1.0)
    params.add('gamma_1',value=0.1,min=0.0)
    minner = Minimizer(cauchy, params, fcn_args=(x, data,err),nan_policy='omit')
    result = minner.minimize()
    
    fit_out = {'amplitude' : result.params['amp'].value, 'constant' : result.params['const'].value, 'gamma' : result.params['gamma_1'].value, 'gamma error' : result.params['gamma_1'].stderr, 'reduced chi' : result.redchi}
    return fit_out

def cauchy_values(x, gamma_1,amp,const):
    model = const +  amp* 1/(np.pi*gamma_1*(1+np.power(x/gamma_1,2)))  # + amp
    return (model)


def twoCauchy(params, x, data,err,gamma_2,amp2):
    amp = params['amp']
    N = params['nonamp']
    gamma_1 = params['gamma_1']

    model = (amp*1/(np.pi*gamma_1*(1+np.power(x/gamma_1,2))) + N*amp2*1/(np.pi*gamma_2*(1+np.power(x/gamma_2,2))))
    return np.power((model - data)/err,2)

def twoCauchy_N(params, x, data,err,gamma_2):
    amp = params['amp']
    N = params['N']
    gamma_1 = params['gamma_1']

    model = amp*(N*1/(np.pi*gamma_1*(1+np.power(x/gamma_1,2))) + (1-N)*1/(np.pi*gamma_2*(1+np.power(x/gamma_2,2))))
    return np.power((model - data)/err,2)


def fitTwoCauchy_N(x, data, err,xbin):   
    data1 = np.copy(data)#[9,10,11]
    #data1[len(data)/2-2] = float('NaN')
    data1[len(data)/2-1] = float('NaN')
    data1[len(data)/2] = float('NaN')
    data1[len(data)/2+1] = float('NaN')
    #data1[len(data)/2+2] = float('NaN')


    params1 = Parameters()
    params1.add('amp', value=0.3, min=0.1)
    params1.add('gamma_1',value=0.4,min=0.2,max=0.5)
    
    minner1 = Minimizer(cauchy_no_const, params1, fcn_args=(x, data1,err),nan_policy='omit')
    result1 = minner1.minimize()
    gamma_2 = result1.params['gamma_1']
    amp2 = result1.params['amp']
    
    params = Parameters()
    params.add('amp', value=0.3, min=0.1,max=0.5)
    params.add('N', value=0.6,min=0.15,max=1.0)
    params.add('gamma_1',value=0.04, min=0.025,max=0.2)
    
    minner = Minimizer(twoCauchy_N, params, fcn_args=(x, data,err,gamma_2),nan_policy='omit')
    result = minner.minimize()
    #report_fit(result)

    a = result.params['amp'].value
    n = result.params['N'].value
    fit_out = {'amplitude' : a, 'constant' : 0.0, 'gamma_1' : result.params['gamma_1'].value, 'gamma_1 error' : result.params['gamma_1'].stderr,  'gamma_2' : result1.params['gamma_1'].value, 'gamma_2 error' : result1.params['gamma_1'].stderr, 'reduced chi' : result.redchi,'N_1' : n}
    
    return fit_out


def cauchy_residual(x, data, err, gamma, amp,const):

    fit = cauchy_values(x,gamma,amp,const)
    
    residual = np.divide(np.fabs(data - fit),err,where=err>0,out=np.zeros_like(data))    
    return residual

def fitTwoCauchy(x, data, err,xbin):   
    data1 = np.copy(data)#[9,10,11]
    #data1[len(data)/2-2] = float('NaN')
    data1[len(data)/2-1] = float('NaN')
    data1[len(data)/2] = float('NaN')
    data1[len(data)/2+1] = float('NaN')
    #data1[len(data)/2+2] = float('NaN')


    params1 = Parameters()
    params1.add('amp', value=0.3, min=0.1)
    params1.add('gamma_1',value=0.4,min=0.3,max=1.0)
    
    minner1 = Minimizer(cauchy_no_const, params1, fcn_args=(x, data1,err),nan_policy='omit')
    result1 = minner1.minimize()
    gamma_2 = result1.params['gamma_1']
    amp2 = result1.params['amp']
    
    params = Parameters()
    params.add('amp', value=0.3, min=0.1,max=0.5)
    params.add('nonamp', value=0.6,min=0.0,max=1.0)
    params.add('gamma_1',value=0.04, min=0.02,max=0.2)
    
    minner = Minimizer(twoCauchy, params, fcn_args=(x, data,err,gamma_2,amp2),nan_policy='omit')
    result = minner.minimize()
    #report_fit(result)

    a = result1.params['amp'].value*result.params['nonamp'].value*result.params['amp'].value + result.params['amp'].value
    n = result.params['amp'].value/a
    fit_out = {'amplitude' : a, 'constant' : 0.0, 'gamma_1' : result.params['gamma_1'].value, 'gamma_1 error' : result.params['gamma_1'].stderr,  'gamma_2' : result1.params['gamma_1'].value, 'gamma_2 error' : result1.params['gamma_1'].stderr, 'reduced chi' : result.redchi,'N_1' : n}
    
    return fit_out


def two_cauchy_residual(x, data,err, N, amp, gamma1, gamma2):
    # calculate final result
    
    fit1 = cauchy_values(x,gamma1,amp*N,0.0)
    fit2 = cauchy_values(x,gamma2,amp*(1-N),0.0)
    fit12 = fit1 + fit2
    
    residual = np.divide(np.fabs(data - fit12),err,where=err>0,out=np.zeros_like(data))
    return residual


def two_cauchy_values(x, gamma_1,gamma_2,N_1,amp,const):

    model = const + amp*1/np.pi*(N_1*gamma_1/(np.power(x,2) + np.power(gamma_1,2)) + (1-N_1)*gamma_2/(np.power(x,2) + np.power(gamma_2,2)))
    return model


def fftTransform(x, data, err,const):
    if (const == False):
        constant = 0.0
    else:
        fit_out = fitcauchy(x, data, err)
        constant = fit_out['constant']

    myn = len(x)
    # if (len(x) > 100):
    #     myn=100
    y= np.abs( np.fft.fft(data-constant,n=myn,norm=None))
    myy = 1.0/(y*1.0/y[0])
    samples = np.zeros([5,myn],dtype=float)
    
    for i in range(0,myn):
        samples[:,i] = np.random.choice([data[i]-constant - err[i],data[i]-constant + err[i]], 5)
    ally = np.zeros([5,3],dtype=float)
    for i in range(0,5):
        y= np.abs( np.fft.fft(samples[i],n=myn,norm=None))
        y = 1.0/(y*1.0/y[0])#1.0/y/y[0] #*1.0/y[0])
        
        ally[i,0] = y[2]
        ally[i,1] = y[3]
        ally[i,2] = y[4]
        #print 'y',y[2:4]

    #print 'myy',myy[2:4]
    fft_out = {'v_2 corr.' : np.nanmean(ally[:,0]), 'v_3 corr.' : np.nanmean(ally[:,1]), 'v_4 corr.' : np.nanmean(ally[:,2]), 'v_2 error' : np.nanstd(ally[:,0]), 'v_3 error' : np.nanstd(ally[:,1]), 'v_4 error' : np.nanstd(ally[:,2])}

    return fft_out
