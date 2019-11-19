from flask import Flask, render_template, request
# import sqlite3 as sql
# import math
from datetime import datetime, timedelta

import numpy as np
# import uuid
from matplotlib import pylab
from numpy import vstack,array
from scipy.cluster.vq import *
from pylab import *

application = Flask(__name__)

import csv 
import sys
import os

coloumn_names = ["pclass","survived","name","sex","age","sibsp","parch","ticket","fare","cabin", "embarked", "boat", "body", "home.dest"]
myfile = open("titanic3.csv","r")
csv_reader = csv.DictReader(myfile, fieldnames=coloumn_names)
next(csv_reader)       


@application.route('/')
def index():
  return render_template("index.html")


collist = []
@application.route('/kmeans', methods=['GET', 'POST'])
def kmeans():
        col1 = request.form['col1']
        col2 = request.form['col2']
        clusters = request.form['clusters']
        K_clusters = int(clusters)
        collist = getdata(col1,col2)
        data = []
        cdist=[]
        data = array(collist)
        cent, pts = kmeans2(data,K_clusters)
        disCluster = []
        for i in range(len(cent)):
            x1 = cent[i][0]#latitude
            y1 = cent[i][1]#longitude
            x1 = float("{0:.3f}".format(x1))
            y1 = float("{0:.3f}".format(y1))

            for j in range(i+1,len(cent)):
                dc = {}
                x2 = cent[j][0]
                y2 = cent[j][1]
                x2 = float("{0:.3f}".format(x2))
                y2 = float("{0:.3f}".format(y2))
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # print(dist)
                cdist.append(dist)
                dc['dist'] = "Distance between cluster " + str(i) + " and cluster " + str(j) + " is: " + str(dist)
                disCluster.append(dc)
                # print (disCluster)
                # print ("Distance between cluster " + str(i) + " and cluster " + str(j) + " is: " + str(dist))
        clr = ([1, 1, 0.0],[0.2,1,0.2],[1,0.2,0.2],[0.3,0.3,1],[0.0,1.0,1.0],[0.6, 0.6,0.1],[1.0,0.5,0.0],[1.0,   0.0, 1.0],[0.6,   0.2, 0.2],[0.1,0.6,0.6],[0.0,0.0,0.0],[0.8,1.0,1.0],[0.70,0.50,0.50],[0.5,0.5,0.5],[0.77,0.70,0.00])
        colors = ([(clr)[i] for i in pts])
        clr_dict = {"yellow":0,"green":0,"red":0,"blue":0,"cyan":0,"deepolive":0,"orange":0,"magenta":0,"ruby":0,"deepteal":0,"black":0,"palecyan":0,"dirtyviolet":0,"gray":0,"olive":0}
        pdict=[]
        for x in colors:
            if str(x) == "[1, 1, 0.0]":
                clr_dict["yellow"] += 1
            if str(x) == "[0.2, 1, 0.2]":
                clr_dict["green"] += 1
            if str(x) == "[1, 0.2, 0.2]":
                clr_dict["red"] += 1
            if str(x) == "[0.3, 0.3, 1]":
                clr_dict["blue"] += 1
            if str(x) == "[0, 1.0, 1.0]":
                clr_dict["cyan"] += 1
            if str(x) == "[0.6, 0.6,0.1]":
                clr_dict["deepolive"] += 1
            if str(x) == "[1.0,  0.5, 0.0]":
                clr_dict["orange"] += 1
            if str(x) == "[1.0,  0.0, 1.0]":
                clr_dict["magenta"] += 1
            if str(x) == "[0.6,  0.2, 0.2]":
                clr_dict["ruby"] += 1
            if str(x) == "[0.1,  0.6, 0.6]":
                clr_dict["deepteal"] += 1
            if str(x) == "[0.0,  0.0, 0.0]":
                clr_dict["black"] += 1
            if str(x) == "[0.8,  1.0, 1.0]":
                clr_dict["palecyan"] += 1
            if str(x) == "[0.70, 0.50, 0.50]":
                clr_dict["dirtyviolet"] += 1
            if str(x) == "[0.5,  0.5, 0.5]":
                clr_dict["gray"] += 1
            if str(x) == "[0.77, 0.70, 0.00]":
                clr_dict["olive"] += 1

        f_write='Cluster,Count\r\n'
        cnt=0
        # print (clr_dict)
        for i in clr_dict:
            if clr_dict[i] == 0:
                continue
            string = str(cnt) + " : " + str(clr_dict[i])
            pdict.append(string)
            # print ("No of points in cluster with " + str(i) + " is: " + str(clr_dict[i]))
            f_write+= str(cnt)+','+str(clr_dict[i])+'\r\n'
            cnt += 1
        with open("static/d3chart.csv",'wb') as nfile:
            nfile.write(f_write.encode("utf-8"))
        pylab.scatter(data[:,0],data[:,1], c=colors)
        pylab.scatter(cent[:,0],cent[:,1], marker='o', s = 400, linewidths=3, c='none')
        pylab.scatter(cent[:,0],cent[:,1], marker='*', s = 400, linewidths=3)

        pylab.savefig("static/kmeans6.png")
        return render_template('kmeans.html',cdist=cdist,pdict=pdict, disCluster = disCluster)

def getdata(c1,c2):
    c = 0
    for row in csv_reader:
        c += 1
        if c == 5000:
            break
        pair = []
        if row[c1] == "":
            row[c1] = 0
        if row[c2] == "":
            row[c2] = 0
        x = float(row[c1])
        y = float(row[c2])
        pair.append(x)
        pair.append(y)
        collist.append(pair)
    return collist


if __name__ == '__main__':
    application.debug = True
    application.run()