#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:51:06 2019

@author: jaymz
"""

import numpy as np
import scipy
from scipy import stats
import scipy.signal
import scipy.interpolate
import numexpr

def replaceByRegression(xInput):
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(xInput)),xInput)
    return [intercept + slope*i for i in range(0,len(xInput))]

def highPassFilter(y,freq=0.0005,order=3):
    transferNumerator, transferDenominator = scipy.signal.butter(order,freq,'highpass')
    y = scipy.signal.filtfilt(transferNumerator,transferDenominator,y)
    return y

def lowPassFilter(y,freq=0.002,order=3):
    transferNumerator, transferDenominator = scipy.signal.bessel(order,freq,'lowpass')
    y = scipy.signal.filtfilt(transferNumerator,transferDenominator,y)
    return y

def replaceInfs(y, replaceNans=True): #Needed for new Tektonix scope apparently?!?
    mask = np.isinf(y) | np.isnan(y)
    if(replaceNans):
        mask= mask | np.isnan(y)
    mask_not = mask!=True
    minVal, maxVal = np.min(y[mask_not]), np.max(y[mask_not])
    
    y[mask] = np.where(y[mask]<0,minVal,maxVal)
    return y

def getWindow(N, window = 'none'):
    windowFunction = range(0,N)
    if(window == 'none'):
        windowFunction = np.array([1.0]*N)
    elif(window == 'Hann'):
        windowFunction = 0.5*(1.0-np.cos(2*np.pi*np.array(windowFunction)/(N-1)))
    elif(window == 'Hamming'):
        windowFunction = 0.54-0.46*np.cos(2*np.pi*np.array(windowFunction)/(N-1))
    elif(window == 'Welch'):
        windowFunction = 1.0-((np.array(windowFunction)-(N-1)/2)/((N-1)/2))**2
    elif(window == 'Blackman'):
        windowFunction = 0.42  - 0.5*np.cos(2*np.pi*np.array(windowFunction)/(N-1)) + 0.08*np.cos(4*np.pi*np.array(windowFunction)/(N-1))
    elif(window == 'Nuttall'):
        a0, a1, a2, a3 = 0.355768,0.487396,0.144232,0.012604
        windowFunction = np.array(windowFunction)
        windowFunction = a0 - a1*np.cos(2*np.pi*windowFunction/(N-1))+a2*np.cos(4*np.pi*windowFunction/(N-1))-a3*np.cos(6*np.pi*windowFunction/(N-1))
    return windowFunction

def resample(x,y,scan):
    if(x[0]>x[-1]):
        x = x[::-1]
        y = y[::-1]
    sR = abs(x[-1]-x[0])/len(x)
    n = int(scan/sR)
    xNew = np.linspace(-scan/2,scan/2,n)
    yNew = np.interp(xNew,x,y)
    zerosF = np.greater(xNew,x[0])
    zerosL = np.less(xNew,x[-1])
    yNew = numexpr.evaluate('zerosF*yNew*zerosL')

    return {'x' : xNew , 'y' : yNew}

def resampleUniform(x,y):
    if(x[0]>x[-1]):
        x = x[::-1]
        y = y[::-1]
    sR = abs(x[-1]-x[0])/len(x)
    xNew = np.linspace(x[0],x[-1],len(x))
    yNew = np.interp(xNew,x,y)

    return {'x' : xNew , 'y' : yNew}

def fft(datX,datY):
    rangeX=datX[-1]-datX[0]
    ft = np.fft.fft(datY)
    ft = ft[1::]
    i = np.arange(1,len(datX))[::-1]
    return {'f' : 1000*rangeX/i, 'y' : ft[::-1]}

def trimSpectrum(f,y, wI_uM, wF_uM):
    mask = f > wI_uM
    mask = mask* (f<wF_uM)
    return {'f' : f[mask], 'y' : y[mask]}

def findMaxFrequency(ft):
    fVal, outI = -1, -1
    for i in range(0,len(ft['f'])):
        y = np.abs(ft['y'][i])
        if(y > outI):
            outI = y
            fVal = ft['f'][i]
    return fVal

def subtractBaseline(y):
    baseline = np.mean([np.mean(y[1:int(len(y)/4)]),np.mean(y[int(len(y)*3/4):-1])])
    return y - baseline

def getZeroCrossings(x,y):
    xi = np.array(x[0:-1])
    xi1 = np.array(x[1::])
    yi = np.array(y[0:-1])
    yi1= np.array(y[1::])
    mask = yi*yi1 < 0
    xzero = -yi[mask]*(xi[mask] - xi1[mask])/(yi[mask]-yi1[mask]) + xi[mask]
    return xzero

def getZeroCrossings_p(x,y):
    xi = np.array(x[0:-1])
    xi1 = np.array(x[1::])
    yi = np.array(y[0:-1])
    yi1= np.array(y[1::])
    mask = yi*yi1 < 0
    mask = mask & ((yi1-yi) > 0)
    xzero = -yi[mask]*(xi[mask] - xi1[mask])/(yi[mask]-yi1[mask]) + xi[mask]
    return xzero

def getZeroCrossings2(x,y):
    inversionPoints = []
    for i in range(1,len(x)):
        if(y[i-1]*y[i]<0):
            inversionPoints.append((x[i]*y[i-1]-x[i-1]*y[i])/(y[i-1]-y[i]))
    return inversionPoints

def wavemeter(ref, sig, refWavelength_nm=632.991, resample_factor=None):
    if(resample_factor is not None):
        t=np.arange(len(ref))
        ri = scipy.interpolate.interp1d(t, ref, kind='cubic', fill_value='extrapolate')
        si = scipy.interpolate.interp1d(t, sig, kind='cubic', fill_value='extrapolate')
        ti = np.linspace(0,len(ref),int(len(ref)*resample_factor))
        r = ri(ti)
        s = si(ti)
    else:
        r = ref
        s = sig
    t=np.arange(len(r))
    zc_ref = getZeroCrossings(t,r)
    zc_sig = getZeroCrossings(t,s)
    n_ref = np.arange(len(zc_ref))
    n_sig = np.arange(len(zc_sig))
    ref_interp = scipy.interpolate.interp1d(zc_ref,n_ref)
    slope = np.polyfit(ref_interp(zc_sig[20:-20]),n_sig[20:-20],1)[0]
    return [refWavelength_nm/slope,len(zc_ref),len(zc_sig)]

def constructPhase(zeroCrossings, x, deriv=0, smooth=0):
    #phase = [np.pi*i for i in range(0,len(zeroCrossings))]
    phase = np.pi*np.arange(len(zeroCrossings))
    interpFunc =  scipy.interpolate.splrep(zeroCrossings, phase, s=smooth)
    return scipy.interpolate.splev(x, interpFunc, der=deriv)

def reconstructDistance(x,y,refWavelength,cutoffInterval = 30, SAVGOL_WINDOW_SIZE = 7, HPF_FREQ = None, LPF_FREQ = None):
    if(x[0]>x[-1]):
        x = x[::-1]
        y = y[::-1]
    #yDat = y - np.mean(y)
    yDat = subtractBaseline(y)
    if(LPF_FREQ is not None):
        yDat = lowPassFilter(yDat,LPF_FREQ,4)
    if(HPF_FREQ is not None):
        yDat = highPassFilter(yDat,HPF_FREQ,4)
    #yDat = scipy.signal.savgol_filter(yDat,SAVGOL_WINDOW_SIZE,3)
    zc = getZeroCrossings(x,yDat)
    #print("Found %d zero crossings" % (len(zc)))
    phi = constructPhase(zc,x[cutoffInterval:-cutoffInterval])
    distOut = phi/(2*np.pi)*refWavelength
    distOut = distOut - distOut[int(len(distOut)/2)]
    return {'distance' : distOut, 'x' : x[cutoffInterval:-cutoffInterval], 'y' : y[cutoffInterval:-cutoffInterval], 'zc' : zc}

def stitchDR(lowGain, highGain, crossoverWidth=100, invert=False, range_ratio=1e2, max_consecutive=1000):
    maxY = np.max(highGain)
    minY = np.min(highGain)
    rangeY = maxY-minY
    lowerWindow, upperWindow = None, None
    consec=0
    for i in range(0,len(lowGain)):
        if( maxY - highGain[i] < rangeY/range_ratio or highGain[i] - minY < rangeY/range_ratio):
            consec=0
            if(lowerWindow is None):
                lowerWindow = i
            else:
                upperWindow = i
        else:
            consec+=1
            if(consec > max_consecutive and lowerWindow is not None):
                break
    print("Low gain window: %d to %d" % (lowerWindow, upperWindow))
    iArr = np.arange(len(lowGain))
    weight = 1.0/(1.0+np.exp(-(iArr-lowerWindow+crossoverWidth*10)/crossoverWidth))/(1.0+np.exp((iArr-upperWindow-crossoverWidth*10)/crossoverWidth))
    if(not invert):
        out = weight*lowGain + (1-weight)*highGain
    else:
        out = weight*lowGain - (1-weight)*highGain
    return out, (lowerWindow, upperWindow)

def stitchDR_N(lowGain, highGain, crossoverWidth=100, invert=False, range_ratio=1e2, max_consecutive=1000):
    maxY = np.max(highGain)
    minY = np.min(highGain)
    rangeY = maxY-minY

    lowerWindow, upperWindow = None, None
    windows = []
    consec=0
    for i in range(0,len(lowGain)):
        if( maxY - highGain[i] < rangeY/range_ratio or highGain[i] - minY < rangeY/range_ratio):
            consec=0
            if(lowerWindow is None):
                lowerWindow = i
            else:
                upperWindow = i
        else:
            consec+=1
            if(consec > max_consecutive and lowerWindow is not None):
                windows.append([lowerWindow, upperWindow])
                lowerWindow, upperWindow = None, None
                consec=0

    print("Detected %d clip windows" % len(windows))
    N = len(lowGain)
    weight = np.ones(N)
    for l,u in windows:
        print("Low gain window: %d to %d" % (l, u))
        iArr = np.arange(len(lowGain))
        weight *= 1.0 - 1.0/(1.0+np.exp(-(iArr-l+crossoverWidth*10)/crossoverWidth))/(1.0+np.exp((iArr-u-crossoverWidth*10)/crossoverWidth))
    if(not invert):
        out = (1-weight)*lowGain + weight*highGain
    else:
        out = (1-weight)*lowGain - weight*highGain
    return out, windows

def findWindows(x,lowGain,highGain, windowWidth = 500):
    maxY = max(highGain)
    minY = min(highGain)
    rangeY = maxY-minY
    windows = []
    while True:
        winStart, winEnd = None, None
        startIndex = 0
        if(len(windows) > 0):
            startIndex = windows[-1][1] + 1
        for i in range(startIndex,len(highGain)):
            if(maxY - highGain[i] < rangeY/10 or highGain[i] - minY < rangeY/10):
                if(winStart is None):
                    print("window %d start: %d" % (len(windows) +1, i))
                    winStart = i
                else:
                    print("window %d end: %d" % (len(windows) +1, i))
                    winEnd = i
            if(winEnd is not None and (i - winEnd) > windowWidth):
                windows.append([winStart,winEnd])
                break
        if(i >= len(x) - 1):
            break

    return windows

def stitchWindows(x,lowGain, highGain, windows, crossoverWidth = 100, invert = False):
    import scipy.interpolate
    out = np.array(highGain)
    weight = np.ones(len(x))
    xInterp = scipy.interpolate.interp1d(x,range(len(x)))
    for w in windows:
        lowerWindow = xInterp(min(w))
        upperWindow = xInterp(max(w))
        #print("Lower window: %d\tUpper window: %d" % (lowerWindow,upperWindow))
        iArr = np.array(range(len(x)))
        weight *= 1.0 - 1.0/(1.0+np.exp(-(iArr-lowerWindow+crossoverWidth*10)/crossoverWidth))/(1.0+np.exp((iArr-upperWindow-crossoverWidth*10)/crossoverWidth))
    if(not invert):
        out = out*weight + (1.0 - weight)*lowGain
    else:
        out = out*weight - (1.0 - weight)*lowGain
    return out

def zeroPadBefore(x,y,numZeros):
    zPad = [0.0]*numZeros
    xInterval = (x[-1] - x[0])/(len(x) - 1)
    xStart = x[0] - numZeros*xInterval
    xPad = np.linspace(xStart,x[0] - xInterval,numZeros).tolist()
    return np.array(xPad + x.tolist()), np.array(zPad + y.tolist())

def zeroPad(x,y,numZeros):
    zPad = [0.0]*numZeros
    xInterval = (x[-1] - x[0])/(len(x) - 1)
    xStart = x[0] - numZeros*xInterval
    xPad_before = np.linspace(xStart,x[0] - xInterval,numZeros).tolist()
    xEnd = x[-1] + numZeros*xInterval
    xPad_after = np.linspace(x[-1] + xInterval,xEnd,numZeros).tolist()
    return np.array(xPad_before + x.tolist()+xPad_after), np.array(zPad + y.tolist()+ zPad)
