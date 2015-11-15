#from __future__ import division

from melpy.core.Classifier import Classifier
from melpy.validation.ClassPrediction import ClassPrediction
from melpy.kernels.LinearKernel import LinearKernel
from melpy.kernels.RBFKernel import RBFKernel
import numpy as np
from scipy.stats import norm
import sklearn.cluster

class StructNoiseSparseGPClassifier(Classifier):
    
    learned_model=0
    kernels=0
    inducing_kernel=0
    isindep_sub_inducing_points=False
    
    # model params
    R=-1      # num latent dimensions
    beta=0.5   # observation noise precision 
    P=20      # Number of inducing points
    
    max_iter=1
    Xind=0    # Observed inducing points
    
    # variational params
    Z=0      # inducing points, size = P x R
    m=0      # mean of q(u), size = P x 1
    S=0      # covariance of q(u), size = P x P

    A=0     # mean of q(X), size = N x R
    B=0     # variance of q(X), size = N x R
            
    def __init__(self,inducing_kernel,kernels,num_inducing, max_iter, learnhyper=1, learn_sub_gps=1, learning_rate_start=1., isindep_sub_inducing_points=True):
        Classifier.__init__(self,'Struct Noise Gaussian Process Classifier')
        
        self.R=len(kernels)
       
        self.kernels=kernels         
        
        self.P = num_inducing
                
        self.inducing_kernel=inducing_kernel
        
        self.max_iter = max_iter
        
        self.learnhyper = learnhyper
        
        self.learn_sub_gps=learn_sub_gps
        
        self.learning_rate_start=learning_rate_start
        
        self.isindep_sub_inducing_points=isindep_sub_inducing_points
        
    def train(self,X,y):
        
        Ntr = X.shape[0]
        t = np.float64(y.ravel())
        tsign=-1+2*t
        
        eps=0.0001 #0.1

        learningrate_m=self.learning_rate_start
        learningrate_S=0.0 
        learningrate_Z=self.learning_rate_start
        learningrate_hyper=self.learning_rate_start
        learningrate_C=self.learning_rate_start
        
        if  isinstance(self.inducing_kernel,LinearKernel):
            learningrate_Z *= 0.1 
            learningrate_C *= 0.0000001 
            
        self.Xind = list()  # choose this more carefully (e.g. k-means)
        self.yind = list() 
        self.Kzz=list()
        self.Kzz_inv=list()
        self.Kzx=list()
        self.kxx=list()
                
        if self.isindep_sub_inducing_points==False:
             for vv in range(20):            
                permInd=np.random.permutation(Ntr)
                         
                tsubset = tsign[permInd[0:self.P]]
                
                if sum(tsubset==1)> self.P*0.3:
                    break     
                        
        for rr in range(self.R):
            if self.isindep_sub_inducing_points == True:
                for vv in range(20):            
                    permInd=np.random.permutation(Ntr)
                             
                    tsubset = tsign[permInd[0:self.P]]
                    
                    if sum(tsubset==1)> self.P*0.3:
                        break     
                
            self.Xind.append(X[permInd[0:self.P],:])            
            self.Kzz.append(self.kernels[rr].compute(self.Xind[rr],self.Xind[rr])) 
            self.Kzz_inv.append(np.linalg.inv(self.Kzz[rr]+np.identity(self.P)*0.001))    
            self.Kzx.append(self.kernels[rr].compute(self.Xind[rr],X))            
            self.kxx.append(self.kernels[rr].computeSelfDistance(X))
            Q=self.Kzx[rr].T.dot(self.Kzz_inv[rr])
            yind_rr=np.linalg.lstsq(Q,tsign)       
            yind_rr=yind_rr[0]
            self.yind.append(yind_rr)

        C=np.zeros([self.P,self.R])        
        Z=np.zeros([self.P,self.R])
        A=np.zeros([Ntr,self.R])
        
        self.kpred=list()
        for rr in range(self.R):
            self.kpred.append(self.kernels[rr].compute(self.Xind[rr],X).T.dot(self.Kzz_inv[rr]))
            A[:,rr]=self.kpred[rr].dot(self.yind[rr])
            C[:,rr]=self.yind[rr]                   

        km=sklearn.cluster.KMeans(n_clusters=self.P,max_iter=400)        
        km.fit(A)
        Z=km.cluster_centers_

        for vv in range(20):            
            permInd=np.random.permutation(Ntr)
                     
            tsubset = tsign[permInd[0:self.P]]
            
            if sum(tsubset==1)> self.P*0.3:
                break 
            

        S = np.identity(self.P)*0.0000001   
        
        B=np.ones([Ntr, self.R])*0.01  
               
        Vzz = self.inducing_kernel.compute(Z,Z)+np.identity(self.P)*eps
        if  isinstance(self.inducing_kernel,LinearKernel) == 1:
            self.inducing_kernel.scale= np.max(Vzz)            
            Vzz /= self.inducing_kernel.scale      
        elif  isinstance(self.inducing_kernel,RBFKernel) == 1: 
            for ii in range(20):
              if np.mean(Vzz) > 0.8:
                 self.inducing_kernel.length_scale *= 0.5
                 Vzz = self.inducing_kernel.compute(Z,Z)+np.identity(self.P)*eps
                 print "Length scale updated to: " + str(self.inducing_kernel.length_scale) 
              elif np.mean(Vzz) < 0.2:
                 self.inducing_kernel.length_scale *= 2.
                 Vzz = self.inducing_kernel.compute(Z,Z)+np.identity(self.P)*eps                 
                 print "Length scale updated to: " + str(self.inducing_kernel.length_scale) 
              else:
                 break
            
        Vzz_inv = np.linalg.inv(Vzz)      
        EVzxVzxT_list=self.inducing_kernel.EVzxVzxT(Z,A,B)        
        EVzxVzxT_this=np.sum(EVzxVzxT_list,axis=0)     
        EVzx_this=self.inducing_kernel.EVzx(Z,A,B) 
        
        Q=EVzx_this.T.dot(Vzz_inv)
        m=np.linalg.lstsq(Q,tsign)       
        m=m[0]
        
        Lthis=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
             
        # variational iterations
        for tt in range(self.max_iter):
                      
           # Update q(u)
                        
           dM = self.gradLowerBound_by_m(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)

           mtry=m+learningrate_m*dM
                              
           L=self.lowerBound(X,t,mtry,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
               
           if L>Lthis:
               m=mtry.copy()
           else:
               learningrate_m=learningrate_m*0.9  
                   
           Lthis=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)

           print  "Iter: " + str(tt) + " ELBO after m:" + np.str(Lthis)     

#           dS = self.gradLowerBound_by_S(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
#
#           Stry=S+learningrate_S*dS
#                          
#           L=self.lowerBound(X,t,m,Stry,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
#           
#           if L>Lthis:
#               S=Stry
#           else:
#               learningrate_S=learningrate_S*0.9
#                   
#           Lthis=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
#
#           print "ELBO after S:" + np.str(Lthis)    
                      
           myFunct =  self.gradLowerBound_by_Z_closure(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
           mapped = np.array(map(myFunct, range(self.P*self.R)))  
           dZ = np.reshape(mapped,[self.P,self.R]) 
             
           Ztry=Z+learningrate_Z*dZ
           
           EVzxVzxT_list_try=self.inducing_kernel.EVzxVzxT(Ztry,A,B)        
           EVzxVzxT_this_try=np.sum(EVzxVzxT_list_try,axis=0)
           EVzx_this_try=self.inducing_kernel.EVzx(Ztry,A,B)   
           Vzz_try = self.inducing_kernel.compute(Ztry,Ztry)+np.identity(self.P)*0.0001
           Vzz_inv_try = np.linalg.inv(Vzz_try)                 
           
           L=self.lowerBound(X,t,m,S,Ztry,Vzz_try,Vzz_inv_try,A,B,C,EVzxVzxT_list_try,EVzxVzxT_this_try,EVzx_this_try)
               
           if L>Lthis:
               Z=Ztry.copy()
               EVzxVzxT_list=EVzxVzxT_list_try.copy()
               EVzxVzxT_this=EVzxVzxT_this_try.copy()
               EVzx_this = EVzx_this_try.copy()
               Vzz=Vzz_try.copy()
               Vzz_inv=Vzz_inv_try.copy()
           else:
               learningrate_Z=learningrate_Z*0.9     
                          
           Lthis=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
                      
           print  "Iter: " + str(tt) + " ELBO after Z:" + np.str(Lthis)  
           
           # Update the GP inducing outputs a_r! -----------------------------
           if self.learn_sub_gps==1:
               
               myFunct2 =  self.gradLowerBound_by_C_closure(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
               mapped2 = np.array(map(myFunct2, range(self.P*self.R)))  
               gC = np.reshape(mapped2,[self.P,self.R])                
                                        
#               for aa in range(5):
               Ctry = C + learningrate_C*gC
                       
               Atry = A.copy()
               for rr in range(self.R):
                   Atry[:,rr]=self.kpred[rr].dot(Ctry[:,rr])
                   
               EVzxVzxT_list_try=self.inducing_kernel.EVzxVzxT(Z,Atry,B)        
               EVzxVzxT_this_try=np.sum(EVzxVzxT_list_try,axis=0)
               EVzx_this_try=self.inducing_kernel.EVzx(Z,Atry,B)            
                                  
               L=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,Atry,B,Ctry,EVzxVzxT_list_try,EVzxVzxT_this_try,EVzx_this_try)
               
               if L>Lthis:
                   A=Atry.copy()
                   C=Ctry.copy()
                   EVzxVzxT_list=EVzxVzxT_list_try.copy()
                   EVzxVzxT_this=EVzxVzxT_this_try.copy()
                   EVzx_this=EVzx_this_try.copy()
                   #break
               else:
                   learningrate_C=learningrate_C*0.9
           
               Lthis=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
               print "Iter: " + str(tt) + " ELBO after C:" + np.str(Lthis)           
                   
           # Update the length scale!        
           if self.learnhyper==1 & self.inducing_kernel.num_hyperparams>0:
               gSigma = np.zeros([self.inducing_kernel.num_hyperparams,1]).ravel()
               for hh in range(self.inducing_kernel.num_hyperparams):
                   gSigma[hh]=self.gradLowerBound_by_hyper(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,hh)                                                        
               
               trykernel=self.inducing_kernel.clone()
               
               #for aa in range(5):
               trykernel.length_scale = self.inducing_kernel.length_scale+learningrate_hyper*gSigma
               
               EVzxVzxT_list_try=trykernel.EVzxVzxT(Z,A,B)        
               EVzxVzxT_this_try=np.sum(EVzxVzxT_list_try,axis=0)
               EVzx_this_try=trykernel.EVzx(Z,A,B)   
               Vzz_try = trykernel.selfCompute(Z)+np.identity(self.P)*eps
               Vzz_inv_try = np.linalg.inv(Vzz_try)                 
                   
               L=self.lowerBound(X,t,m,S,Z,Vzz_try,Vzz_inv_try,A,B,C,EVzxVzxT_list_try,EVzxVzxT_this_try,EVzx_this_try)
                   
               if L>Lthis:
                   self.inducing_kernel.length_scale=trykernel.length_scale
                   EVzxVzxT_list=EVzxVzxT_list_try.copy()     
                   EVzxVzxT_this=EVzxVzxT_this_try.copy()
                   EVzx_this=EVzx_this_try.copy()                             
                   Vzz = Vzz_try
                   Vzz_inv = Vzz_inv_try                    
                   #break
               else:
                   learningrate_hyper=learningrate_hyper*0.9       
               
               Lthis=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
                          
               print "Iter: " + str(tt) + " ELBO after H:" + np.str(Lthis)  + " Length scale : " + np.str(self.inducing_kernel.length_scale)
                                
        Vzz = self.inducing_kernel.compute(Z,Z)+np.identity(self.P)*eps
        Vzz_inv = np.linalg.inv(Vzz)
        self.m = m
        self.S = S
        self.Z = Z
        self.A = A
        self.B = B     
        self.C = C
        self.Vzz_inv=Vzz_inv
        self.Vzz=Vzz
        
    def gradLowerBound_by_Z_closure(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):               
        
        def funct(idx):
            return self.gradLowerBound_by_Z(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,idx)
            
        return funct 
        
    def gradLowerBound_by_Z_closure_diff(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):               
        
        def funct(idx):
            return self.gradLowerBound_by_Z_diff(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,idx)
            
        return funct

    def gradLowerBound_by_C_closure(self,X,y,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):               
        
        def funct(idx):
            return self.gradLowerBound_by_c(X,y,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,idx)
            
        return funct          
          
    def lowerBound(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):
        val = 0
        Ntr = np.float64(X.shape[0])
        
        t=t.ravel()*1
        m=m.ravel()*1
        
        sqBetaPlusOne=np.sqrt(1/self.beta+1)
        
        mmtS = np.outer(m,m)+S
        Minterm = Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv)
        
        phi = EVzx_this.T.dot(Vzz_inv).dot(m) / sqBetaPlusOne
        
        M2=m.dot(Vzz_inv).dot(EVzx_this)
        
        normCdfPhi=norm.cdf(phi)
        
        smallElements=normCdfPhi < 0.000001
        
        normCdfPhi[smallElements]=1.0
        
        logNormCdfPhi=np.log(normCdfPhi)    
        
        likTerm1 =  0.5*self.beta*np.power(M2,2) + logNormCdfPhi 
        likTerm0 = np.log(np.sqrt(2*np.pi/self.beta))- logNormCdfPhi
        
        tinv=np.abs(t-1)
            
        (temp,logdetS)=np.linalg.slogdet(S)
        (temp,logdetVzz)=np.linalg.slogdet(Vzz)
        
        val += 0.5*Ntr*np.log(self.beta) \
              -0.5*logdetVzz \
              +0.5*logdetS \
              -0.5*m.dot(Vzz_inv).dot(m) \
              -0.5*Vzz_inv.dot(S).trace() \
              +0.5*self.beta*Vzz_inv.dot(EVzxVzxT_this).trace() \
              -0.5*self.beta*Minterm.dot(mmtS).trace()\
              + (likTerm1*t).sum() + (likTerm0*tinv).sum()
              
        for rr in range(self.R):
            val += -0.5*C[:,rr].dot(self.Kzz_inv[rr]).dot(C[:,rr])
            
        return val
        
    def gradLowerBound_by_m(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):

        P=Z.shape[0]
        val = -self.beta*Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv).dot(m)-Vzz_inv.dot(m)
        
        ttiled=np.tile(t,[P,1])
        tinv_tiled=np.tile(np.abs(t-1),[P,1]) 
        sqBetaPlusOne=np.sqrt(1/self.beta+1)  
        
        Mm2= EVzx_this.T.dot(Vzz_inv).dot(m)  
        Mm2tiled =  np.tile(Mm2,[P,1])
        phiTiled= norm.cdf(Mm2tiled/sqBetaPlusOne)
        dLogPhi= norm.pdf(Mm2tiled/sqBetaPlusOne,0,1)*(1/phiTiled)*Vzz_inv.dot(EVzx_this)/sqBetaPlusOne 
        
        gradLik1 = self.beta*Mm2tiled*Vzz_inv.dot(EVzx_this) + dLogPhi
        gradLik0 = -dLogPhi
        
        val += (gradLik1*ttiled).sum(axis=1) + (gradLik0*tinv_tiled).sum(axis=1)
       
        return val  
        
    def gradLowerBound_by_S(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):

        Sinv=np.linalg.inv(S)+np.identity(Z.shape[0])*0.0001
        val = -0.5*Vzz_inv + 0.5*Sinv.T - 0.5*self.beta*Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv)
               
        return val
        
    def gradLowerBound_by_Z(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,idx):
 
        xx=idx/self.R       
        yy=np.mod(idx,self.R)
        
        tinv=np.abs(t-1)        
        
        # global terms  
        grad_Vzz=self.inducing_kernel.grad_K_by_Z(Vzz,Z,xx,yy)        
        grad_Vzz_inv = -Vzz_inv.dot(grad_Vzz).dot(Vzz_inv)
        grad_EVzxVzxT_by_Z_this=self.inducing_kernel.grad_EVzxVzxT_by_Z(EVzxVzxT_list,Z,A,B,xx,yy)        
        grad_EVzx=self.inducing_kernel.grad_EVzx_by_Z(EVzx_this,Z,A,B,xx,yy)
        
        mmtS = np.outer(m,m)+S  

        sqBetaPlusOne=np.sqrt(1/self.beta+1)              
                
        Mm1= grad_EVzx.T.dot(Vzz_inv).dot(m)+EVzx_this.T.dot(grad_Vzz_inv).dot(m)
        Mm2= EVzx_this.T.dot(Vzz_inv).dot(m)   
        dLogPhi= norm.pdf(Mm2/sqBetaPlusOne)*(1/norm.cdf(Mm2/sqBetaPlusOne))*Mm1/sqBetaPlusOne
        
        gradLik1=self.beta*Mm2*Mm1+dLogPhi
        gradLik0=-dLogPhi
        
        T1 = grad_Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv)
        T2 = Vzz_inv.dot(grad_EVzxVzxT_by_Z_this).dot(Vzz_inv)
        T3 = Vzz_inv.dot(EVzxVzxT_this).dot(grad_Vzz_inv)
        Ttot = T1+T2+T3 
        
        val = -0.5*Vzz_inv.dot(grad_Vzz).trace() \
               -0.5*m.dot(grad_Vzz_inv).dot(m) \
               -0.5*grad_Vzz_inv.dot(S).trace() \
               +0.5*self.beta*(grad_Vzz_inv.dot(EVzxVzxT_this).trace() \
               +Vzz_inv.dot(grad_EVzxVzxT_by_Z_this).trace()) \
               -0.5*self.beta*Ttot.dot(mmtS).trace() \
               +(gradLik1*t).sum() + (gradLik0*tinv).sum()
 
        return val  
        
    # Actually the same as the above function!    
    def gradLowerBound_by_hyper(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,hyperno):
        val=0 

        grad_Vzz=self.inducing_kernel.grad_K_by_hyper(Vzz,Z,hyperno)       
        grad_Vzz_inv = -Vzz_inv.dot(grad_Vzz).dot(Vzz_inv) 
        grad_EVzxVzxT_by_Z_this=self.inducing_kernel.grad_EVzxVzxT_by_hyper(EVzxVzxT_list,Z,A,B,hyperno)          
        grad_EVzx=self.inducing_kernel.grad_EVzx_by_hyper(EVzx_this,Z,A,B,hyperno) 
        
        tinv=np.abs(t-1)        
        
        mmtS = np.outer(m,m)+S  

        sqBetaPlusOne=np.sqrt(1/self.beta+1)              
                
        Mm1= grad_EVzx.T.dot(Vzz_inv).dot(m)+EVzx_this.T.dot(grad_Vzz_inv).dot(m)
        Mm2= EVzx_this.T.dot(Vzz_inv).dot(m)   
        dLogPhi= norm.pdf(Mm2/sqBetaPlusOne)*(1/norm.cdf(Mm2/sqBetaPlusOne))*Mm1/sqBetaPlusOne
        
        gradLik1=self.beta*Mm2*Mm1+dLogPhi
        gradLik0=-dLogPhi
        
        T1 = grad_Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv)
        T2 = Vzz_inv.dot(grad_EVzxVzxT_by_Z_this).dot(Vzz_inv)
        T3 = Vzz_inv.dot(EVzxVzxT_this).dot(grad_Vzz_inv)
        Ttot = T1+T2+T3 
        
        val += -0.5*Vzz_inv.dot(grad_Vzz).trace()
        val += -0.5*m.dot(grad_Vzz_inv).dot(m)
        val += -0.5*grad_Vzz_inv.dot(S).trace()
        val += 0.5*self.beta*(grad_Vzz_inv.dot(EVzxVzxT_this).trace() +Vzz_inv.dot(grad_EVzxVzxT_by_Z_this).trace())
        val += -0.5*self.beta*Ttot.dot(mmtS).trace()
        val += (gradLik1*t).sum() + (gradLik0*tinv).sum()
 
        return val 
        
    def gradLowerBound_by_c(self,X,y,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,idx):
        xx=idx/self.R       
        yy=np.mod(idx,self.R)
        
        mmtS = np.outer(m,m)+S  
                
        val = (-self.Kzz_inv[yy].dot(C[:,yy]))[xx]

        grad_EVzx=self.inducing_kernel.grad_EVzx_by_c(EVzx_this,Z,A,B,C,self.kpred,xx,yy)
           
        grad_EVzxVzxT=self.inducing_kernel.grad_EVzxVzxT_by_c(EVzxVzxT_list,Z,A,B,C,self.kpred,xx,yy)
        
        val += self.beta*y.dot(grad_EVzx.T).dot(Vzz_inv).dot(m) \
              -0.5*self.beta*Vzz_inv.dot(grad_EVzxVzxT).dot(Vzz_inv).dot(mmtS).trace()\
              +0.5*self.beta*Vzz_inv.dot(grad_EVzxVzxT).trace()\
              -0.5*self.beta*self.inducing_kernel.grad_EVxx_by_c(self.kpred,A,B,C,xx,yy)
            
        return val            
        
    def gradLowerBound_by_hyper_diff(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,idx):
        val=0
 
        L=self.lowerBound(X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this)
         
        trykernel2=self.inducing_kernel.clone()
        trykernel2.length_scale += 0.0001
        
        Vzz_new = trykernel2.compute(Z,Z)+np.identity(self.P)*0.0001
        Vzz_inv_new = np.linalg.inv(Vzz_new)        
        
        EVzxVzxT_list_try=trykernel2.EVzxVzxT(Z,A,B)        
        EVzxVzxT_this_try=np.sum(EVzxVzxT_list_try,axis=0)
        EVzx_this_try=trykernel2.EVzx(Z,A,B)  
               
        Ldelta=self.lowerBound(X,t,m,S,Z,Vzz_new,Vzz_inv_new,A,B,C,EVzxVzxT_list_try,EVzxVzxT_this_try,EVzx_this_try)
        
        val=(Ldelta-L)/ 0.0001
        
        return val        
      
        
    def predict(self,Xts):
      
      Xtest = np.zeros([Xts.shape[0],self.R])
      
      for rr in range(0,len(self.kernels)):
          Kzx=self.kernels[rr].compute(self.Xind[rr],Xts)
          Xtest[:,rr] = Kzx.T.dot(self.Kzz_inv[rr]).dot(self.C[:,rr])
                
      Vzx = self.inducing_kernel.compute(self.Z,Xtest)
      
      probabilities = norm.cdf(Vzx.T.dot(self.Vzz_inv).dot(self.m))
      #probabilities = norm.cdf(Xtest[:,2])
      classes = (probabilities>0.5)*1
        
      class_prediction=ClassPrediction(classes,probabilities)        
      
      return class_prediction       
