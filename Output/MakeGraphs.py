import matplotlib.pyplot as plt
import numpy as np
pref = "RDDFashion-28032020 - 0.01 - 32 - 0.9 - 0/"
fapref = "FAfashion-28032020 - 0.01 - 32 - 0.9 - 0/"

correct_1= np.load(pref + "correct_1.npy")
correct_2= np.load(pref + "correct_2.npy")
correct_3= np.load(pref + "correct_3.npy")
#plt.title("Correct"+pref)
cor1, cor2, cor3 = [],[],[]
length = int(len(correct_1)/10)
def drawSignAlignment():

    for i in range(length):
        cor1.append(correct_1[10*i])
        cor2.append(correct_2[10*i])
        cor3.append(correct_3[10*i])

    facorrect_1= np.load(fapref + "correct_1.npy")
    facorrect_2= np.load(fapref + "correct_2.npy")
    facorrect_3= np.load(fapref + "correct_3.npy")
    plt.title("FC1")
    plt.ylim(0,100)
    plt.xlim(0,length)
    plt.plot(facorrect_1[:length])
    plt.plot(cor1)
    plt.legend(["FA","RDD"])
    plt.show()
    plt.title("FC2")
    plt.ylim(0,100)
    plt.xlim(0,length)
    plt.plot(facorrect_2[:length])
    plt.plot(cor2)
    plt.legend(["FA","RDD"])
    plt.show()
    plt.title("FC3")
    plt.ylim(0,100)
    plt.xlim(0,length)
    plt.plot(facorrect_3[:length])
    plt.plot(cor3)
    plt.legend(["FA","RDD"])
    plt.show()

#plt.legend(handles=[Layer1, Layer2, Layer3])
#plt.show()
def drawWeights():
    weight_1= np.load(pref + "weight_1.npy")
    weight_2= np.load(pref + "weight_2.npy")
    weight_3= np.load(pref + "weight_3.npy")

    faweight_1= np.load(fapref + "weight_1.npy")
    faweight_2= np.load(fapref + "weight_2.npy")
    faweight_3= np.load(fapref + "weight_3.npy")

    fb_weight_1= np.load(pref + "fb_weight_1.npy").T
    fb_weight_2= np.load(pref + "fb_weight_2.npy").T
    fb_weight_3= np.load(pref + "fb_weight_3.npy").T

    fafb_weight_1= np.load(fapref + "fb_weight_1.npy").T
    fafb_weight_2= np.load(fapref + "fb_weight_2.npy").T
    fafb_weight_3= np.load(fapref + "fb_weight_3.npy").T

    flatweight_1=weight_1.flatten()
    flatweight_2=weight_2.flatten()
    flatweight_3=weight_3.flatten()
    flatfaweight_1=faweight_1.flatten()
    flatfaweight_2=faweight_2.flatten()
    flatfaweight_3=faweight_3.flatten()
    flatfb_weight_1=fb_weight_1.flatten()
    flatfb_weight_2=fb_weight_2.flatten()
    flatfb_weight_3=fb_weight_3.flatten()
    flatfafb_weight_1=fafb_weight_1.flatten()
    flatfafb_weight_2=fafb_weight_2.flatten()
    flatfafb_weight_3=fafb_weight_3.flatten()

    plt.title("FC1")
    plt.xlabel("weight_1")
    plt.xlim()
    plt.ylabel("fb_weight_1")
    plt.scatter(flatweight_1,flatfb_weight_1,s=1)
    plt.show()
    plt.title("FC2")
    plt.xlabel("weight_2")
    plt.ylabel("fb_weight_2")
    plt.scatter(flatweight_2,flatfb_weight_2,s=1)
    plt.show()
    plt.title("FC3")
    plt.xlabel("weight_3")
    plt.ylabel("fb_weight_3")
    plt.scatter(flatweight_3,flatfb_weight_3,s=1)
    plt.show()
    plt.title("FC1")
    plt.xlabel("weight_1")
    plt.ylabel("fb_weight_1")
    plt.xlim(-.2,.2)
    plt.ylim(-0.02,0.02)
    plt.scatter(flatfaweight_1,flatfafb_weight_1,s=1, color='r')
    plt.show()
    plt.title("FC2")
    plt.xlabel("weight_2")
    plt.xlim(-.2,.2)
    plt.ylim(-0.05,0.05)
    plt.ylabel("fb_weight_2")
    plt.scatter(flatfaweight_2,flatfafb_weight_2,s=1, color='r')
    plt.show()
    plt.title("FC3")
    #plt.xlim(-.2,.2)
    #plt.ylim(-0.05,0.05)
    plt.xlabel("weight_3")
    plt.ylabel("fb_weight_3")
    plt.scatter(flatfaweight_3,flatfafb_weight_3,s=1, color='r')
    plt.show()
#decay_1 = np.load(pref + "decay_1.npy")
#decay_2 = np.load(pref + "decay_2.npy")
#decay_3 = np.load(pref + "decay_3.npy")
#plt.title("decay"+pref)
#Layer1,=plt.plot(decay_1)
#Layer2,=plt.plot(decay_2)
#Layer3,=plt.plot(decay_3)
##plt.legend(handles=[Layer1, Layer2,Layer3])
#plt.show()
#sparse_1= np.load(pref + "sparse_1.npy")
#sparse_2= np.load(pref +"sparse_2.npy")
#sparse_3= np.load(pref + "sparse_3.npy")
#plt.title("sparse"+pref)
#Layer1,=plt.plot(sparse_1)
#Layer2,=plt.plot(sparse_2)
#Layer3,=plt.plot(sparse_3)
#plt.legend(handles=[Layer1, Layer2, Layer3])
#plt.show()


def drawSelf():
    faself_1 = np.load(fapref + "self_1.npy")
    faself_2= np.load(fapref + "self_2.npy")
    faself_3 = np.load(fapref + "self_3.npy")

    self_1 = np.load(pref + "self_1.npy")
    self_2= np.load(pref + "self_2.npy")
    self_3 = np.load(pref + "self_3.npy")

    tself1 = []
    tself2 = []
    tself3 = []
    for i in range(length):
        tself1.append(self_1[10*i])
        tself2.append(self_2[10*i])
        tself3.append(self_3[10*i])

    plt.title("Self FC1")
    Layer1=plt.plot(faself_1,color="r", linestyle="--")
    Layer1=plt.plot(tself1,color="r")
    plt.show()

    plt.title("Self FC2")
    Layer2=plt.plot(faself_2,color="r", linestyle="--")
    Layer2=plt.plot(tself2,color="r")
    plt.show()

    plt.title("Self FC3")
    Layer3=plt.plot(faself_3,color="r", linestyle="--")
    Layer3=plt.plot(tself3,color="r")
    #plt.legend(handles=[Layer1, Layer2, Layer3])
    plt.show()


#amp_1 = np.load(pref+ "amp_1.npy")
#amp_2 = np.load(pref+ "amp_2.npy")
#amp_3 = np.load(pref+ "amp_3.npy")
#plt.title("amp"+pref)
#Layer1=plt.plot(amp_1)
#Layer2=plt.plot(amp_2)
#Layer3=plt.plot(amp_3)
##plt.legend(handles=[Layer1, Layer2,Layer3])
#plt.show()
#info_1 = np.load(pref+ "info_1.npy"  )
#info_2 = np.load(pref+ "info_2.npy"  )
#info_3 = np.load(pref+ "info_3.npy"  )
#plt.title("info"+pref)
#Layer1=plt.plot(info_1)
#Layer2=plt.plot(info_2)
#Layer3=plt.plot(info_3)
#plt.legend([Layer1, Layer2,Layer3])
#plt.show()
def printInfo():
    sym1= np.load(pref + "correct_1.npy")
    sym2= np.load(pref + "correct_2.npy")
    sym3= np.load(pref + "correct_3.npy")
    sim1 = []
    sim2 = []
    sim3 = []
    for i in range(length):
        sim1.append(sym1[10*i])
        sim2.append(sym2[10*i])
        sim3.append(sym3[10*i])
    plt.title("SymmLoss")
    plt.xlabel("epoch")
    plt.xlim()
    plt.ylabel("Percent")
    plt.plot(sim1)
    plt.plot(sim2)
    plt.plot(sim3)
    plt.show()

def drawErr():
    RDDerr = np.load(pref + "train_err.npy")
    faerr = np.load(fapref + "train_err.npy")

    plt.title("error "+ pref)
    Layer1,=plt.plot(RDDerr[1:101],color="m")
    Layer2,=plt.plot(faerr[1:101],color="b")

    #plt.legend(handles=[Layer1, Layer2,Layer3])
    plt.show()

#drawSignAlignment()
#drawWeights()
#drawSelf()
#printInfo()
drawErr()
