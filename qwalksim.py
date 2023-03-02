import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

class pauli_operator: 
    def __init__(self):
        self.x = npm.array([[0,1],[1,0]])
        self.y = npm.array([[0,-1j],[1j,0]])
        self.z = npm.array([[1,0],[0,-1]])

class Hilbert_space:
    def __init__(self,dim,labels):
        self.dim = dim
        self.labels = labels
        self.basis = self.construct_basis()
        self.id = npm.eye(self.dim)

    def construct_basis(self):
        basis = {}
        for i in range(self.dim):
            psi = npm.zeros((self.dim,1))
            psi[[i]]=1
            basis.update({self.labels[i]:psi})       
        return basis

    def projector(self,i,j):
        return np.kron(self.basis[i].H,self.basis[j])

class quantum_walk:
    def __init__(self,tmax,evolution,initial_state,walker,coin):
        self.tmax = tmax
        self.evolution = evolution
        self.state = initial_state
        self.walker = walker
        self.coin = coin
        self.data = {}
        self.representation = "position"
        self.dt = 1

    def simulation(self):
        self.data.update({0:self.state})
        if self.dt > 1:
            self.evolution = np.linalg.matrix_power(self.evolution,self.dt)
        for t in range(self.dt,self.tmax+1,self.dt):
            print(t)
            self.state = self.evolution*self.state
            self.data.update({t:self.state})

    def get_position_probability_amplitudes(self):
        prob_amps_dict = {}
        for t in range(self.dt,self.tmax+1,self.dt):
            rho = np.kron(self.data[t],self.data[t].H)
            prob_amps = []
            for i in self.walker.labels:
                p = 0
                for s in self.coin.labels:
                    bket = np.kron(self.walker.basis[i],self.coin.basis[s])
                    p = p + abs(((bket.H)*rho*bket)[0,0])
                prob_amps.append(p)  
            prob_amps_dict.update({t:prob_amps})
        return prob_amps_dict

    def plot_probability_amplitudes(self):

        data = np.array(list(self.get_position_probability_amplitudes().values()))
        minval = np.min(data[np.nonzero(data)])         
        cmap = copy.copy(mpl.cm.get_cmap("YlOrRd"))
        plt.imshow(data,cmap=cmap,vmin=minval,aspect='auto')
        cmap.set_under("white")
        ax = plt.gca()
        ticks = [0,int((self.walker.dim-1)/2),self.walker.dim-1]
        ticklabels = [i-max(self.walker.labels) for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        plt.xlabel(r"Site")
        plt.ylabel(r"Time step")
        cbar = plt.colorbar()
        cbar.set_label(r"Probabilty")
        plt.show()
        plt.close()

    def get_reduced_rho(self):

        reduced_rho_dict ={}
        for t in range(self.dt,self.tmax+1,self.dt):
            rho = np.kron(self.data[t],self.data[t].H)

            #coin reduced density matrix 
            rho_c = npm.zeros(shape=(self.coin.dim,self.coin.dim))
            for i in self.walker.labels:
                ket = np.kron(self.walker.basis[i],self.coin.id)
                bra = (ket.H)  
                rho_c = rho_c + bra*rho*ket

            reduced_rho_dict.update({t:rho_c})

        return reduced_rho_dict

    def plot_entanglement_entropy(self,normalized=False):

        reduced_rho = self.get_reduced_rho()
        entropies = []
        for t in range(self.dt,self.tmax+1,self.dt):
            rho_c = reduced_rho[t]
            if normalized:
                rho_c = rho_c/np.trace(rho_c)
            eigvals = np.linalg.eigvals(rho_c)

            if abs(eigvals[0]) < 1e-8 or abs(eigvals[1]) < 1e-8:
                entropies.append(0)
            else:
                entropies.append(-np.sum([v*np.log(v) for v in eigvals]))

        entropies_real = [val.real for val in entropies]
        entropies_imag = [val.imag for val in entropies] #in-case with imaginary
        if np.any([np.abs(v) > 1e-8 for v in entropies_imag]):
            print("Note: imaginary compenents of entropies that were discarded in plot")
            print(entropies_imag)
        plt.plot(list(reduced_rho.keys()),entropies_real)
        plt.ylabel(r"Entanglement entropy")
        plt.xlabel(r"Time step")
        plt.show()
        plt.close()

def plot_complex_eigenvalues(array,unit_circle=False):

    eigvals = np.linalg.eigvals(array)
    x = [v.real for v in eigvals]
    y = [v.imag for v in eigvals]
    plt.axes().set_aspect('equal')
    plt.grid()
    plt.scatter(x,y,marker="x")
    if unit_circle:
        circ_x = np.linspace(-1,1,100)
        circ_y = [np.sqrt(1-x**2) for x in circ_x]
        plt.plot(circ_x,circ_y,c="k")
        circ_y = [-np.sqrt(1-x**2) for x in circ_x]
        plt.plot(circ_x,circ_y,c="k")    
    plt.xlabel(r"Re($\lambda$)")
    plt.ylabel(r"Im($\lambda$)")
    plt.show()
    plt.close()

#######################################################################
if __name__ == "__main__":

    #Construct coin space and coin operator
    qubit = Hilbert_space(2,range(2))
    hadamard_flip = (qubit.projector(0,0)+qubit.projector(0,1)
                    +qubit.projector(1,0)-qubit.projector(1,1))/np.sqrt(2)
    #another way is to construct explicitly
    #hadamard_flip = np.matrix([[1,1],[1,-1]])/np.sqrt(2)

    #Construct walker space and the shift operator that acts on the full space
    xmax = 10
    tmax = xmax
    walker = Hilbert_space(2*xmax+1,range(-xmax,xmax+1)) #walker space
    full_space_dim = walker.dim*qubit.dim
    shift = npm.zeros((full_space_dim,full_space_dim))
    pairs = [(i,j) for i,j in zip(walker.labels,walker.labels[1:])]
    for i,j in pairs:
        shift += np.kron(walker.projector(j,i),qubit.projector(1,1)) #to the right
        shift += np.kron(walker.projector(i,j),qubit.projector(0,0)) #to the left
    
    #periodic boundaries ensure unitarity of the 
    #shift operator in this trunctated walker space
    shift += np.kron(walker.projector(-xmax,xmax),qubit.projector(1,1))
    shift += np.kron(walker.projector(xmax,-xmax),qubit.projector(0,0))
     
    #Construct evolution operator that acts on full space
    evolution = shift*np.kron(walker.id,hadamard_flip)

    #Construct initial state
    init_qubit = (qubit.basis[0]-1j*qubit.basis[1])/np.sqrt(2)
    initial_state = np.kron(walker.basis[0],init_qubit)

    #Run quantum walk simulation
    qwalk = quantum_walk(tmax,evolution,initial_state,walker,qubit)
    qwalk.simulation()
    #qwalk.plot_probability_amplitudes()
    qwalk.plot_entanglement_entropy()






        