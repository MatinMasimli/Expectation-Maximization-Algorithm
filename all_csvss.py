import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from linear_regression_mixtures import *
from fpdf import FPDF
from scipy import stats


dire = os.path.dirname(os.path.realpath(__file__)) #current path
folder = (dire + '/classifier_data') #folder containing all of the .csv files

files = glob.glob(folder+'/*.csv') #list of all files in folder
num = len(files) #number of files in folder

CC = np.array(['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'gray'], dtype = object)
NUM_COMPONENTS = 3 #for all runs

pdf = FPDF('L', 'in', 'Letter')

for i in range(num): #for each file in folder

    fname = files[i].split('/')[-1] #filename.csv
    #fname = files[i].split('\\')[-1]  # filename.csv
    title = fname.split('.')[0] #name of csv file without .csv
    rawdata = np.genfromtxt(files[i], dtype = float, delimiter = ',', skip_header = 1) #data in shape (_, 2)
    x, y = rawdata[:, 0], rawdata[:, 1]
    keys = np.arange(rawdata.shape[0])
    print('File Number: {}'.format(i)) #just to watch the progress
    print('File Name: {}'.format(title))
    print('Number of datapoints: {}'.format(keys.size))

    xlow = np.percentile(x, 1) * 0.5  # 1st percentile limit
    ylow = np.percentile(y, 1) * 0.5
    xhigh = np.percentile(x, 99) * 2  #+ (.5*np.percentile(x, 1)) # 99th percentile limit
    yhigh = np.percentile(y, 99) * 2  #+ np.percentile(y, 1)

    #instead of your xborder, yborder
    #here I just gave .75 from the source i got, you could change some shit
    '''if keys.size < 2000: 
        xlow = .75 * np.percentile(x, 1)
        xhigh = np.percentile(x, 99) + (.25 * np.percentile(x, 1))   '''
        #ylow = .75 * np.percentile(y, 1)
        #yhigh = np.percentile(y, 99) + (.25 * np.percentile(y, 1))

    xLowBorder = xlow + (xlow * 0.75)
    yLowBorder = ylow + (ylow * 0.75)

    xHighBorder = (xhigh / 1.9)
    yHighBorder = (yhigh / 1.9)

    print('Number of datapoints outside x bounds: {}'.format(np.sum(x > xhigh) + np.sum(x < xlow)))
    print('Number of datapoints outside y bounds: {}'.format(np.sum(y > yhigh) + np.sum(y < ylow)))
    data = {}
    for j in range(keys.size): 
        data[j] = (np.array([[1, x[j]]]), np.array([y[j]]))

    #EM clustering:
    results = fit_with_restarts(data, keys, NUM_COMPONENTS, 1000, 20, stochastic=False, verbose=False, num_workers=2)
    statResult = returnResult()
    #print("num_restart in all_csvss: ", statResult[0])
    #print("goodnes-fit in all_csvss: ", statResult[1])

    #print("results in all_csvss: ", results)
    #noise = np.random.randn(y.shape[0], ) #random noise from standard normal dist
    
    #plot data points with small noise:
    plt.figure(1, figsize=(20, 20))
    plt.xlim([xLowBorder, xHighBorder])
    plt.ylim([yLowBorder, yHighBorder])
    plt.scatter(x, y, s = 8, color=CC[results.best.assignments.astype(int)])

    #plot lines:
    for _, coefficients in enumerate(results.best.coefficients):
        xl = np.linspace(xlow, xhigh, 20)
        features = np.ones((20, 2))
        features[:, 1] = xl
        yl = features.dot(coefficients)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        plt.plot(xl, yl, color='black')

    plt.xlim([xLowBorder, xHighBorder])
    plt.ylim([yLowBorder, yHighBorder])
    plt.xlabel('EWT', fontsize = 60)
    plt.ylabel('Watts', fontsize = 60)
    plt.savefig('all_figures/cem_results_test_' + title + '.png')
    plt.clf()

    # below is all for the scatter plot of raw data with histograms
    nullfmt = NullFormatter()

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.01

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(20, 20))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    #plt.xlabel('EWT', fontsize = 30)
    #plt.ylabel('Watts', fontsize = 30)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, s = 8)
    axScatter.set_xlim([xLowBorder, xHighBorder])   # xlow, xhigh
    axScatter.set_ylim([yLowBorder, yHighBorder])   # ylow, yhigh

    #may need to tune number of bins below
    axHistx.hist(x, bins=500, range = (xLowBorder, xHighBorder))        # xlow, xhigh
    axHisty.hist(y, bins=500, range = (yLowBorder, yHighBorder), orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.savefig('all_figures/raw_data_test_' + title + '.png')
    plt.clf()
####################### PDF ##############################################
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.image('all_figures/raw_data_test_' + title + '.png', x = 0, y = 0, w = 5.8, type = 'PNG')
    pdf.image('all_figures/cem_results_test_' + title + '.png', x = 5.6, y = 0.8, w = 5.0, type='PNG')
    pdf.set_y(5.5)
    plt.xlabel('EWT')
    plt.ylabel('Watts')
    pdf.cell(3, h = 1, txt = title)


pdf.output(name = 'all_figures/all_plots.pdf')