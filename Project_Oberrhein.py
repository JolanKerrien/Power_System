import pandapower as pp
import pandapower.networks as pn
import pandapower.topology
import pandapower.plotting
import pandapower.converter
import pandapower.estimation
from pandapower.plotting import cmap_continuous, cmap_discrete, create_bus_collection, create_line_collection, draw_collections,create_trafo_collection
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#-------------FUNCTIONS-----------

#Initialisation
def Init(pmw_gen=10):
    net = pn.mv_oberrhein("generation")
    #Calculate the length of the lines
    for i,length in zip(net.line.index,net.line['length_km']):
        x1,y1 = float(net.bus_geodata['x'][net.line['from_bus'][i]]), float(net.bus_geodata['y'][net.line['from_bus'][i]])
        x2,y2 = float(net.bus_geodata['x'][net.line['to_bus'][i]]), float(net.bus_geodata['y'][net.line['to_bus'][i]])
        net.line['length_km'][i] = np.sqrt((x1-x2)**2+(y1-y2)**2)/300 #factor for convergence
    #Move position of external grid in order to be able to see the transformer
    net.bus_geodata['x'][318], net.bus_geodata['y'][318] = 3408302.042,5367780.101
    net.bus_geodata['x'][58], net.bus_geodata['y'][58] = 3420000.660 , 5369458.703
    #Drop all already in static generators
    #for i,_ in enumerate(net.sgen['p_mw']):
    #    net.sgen = net.sgen.drop(i)
    #Creating the generator
    x_gen, y_gen = 3410832.464,5368594.558 #Determine on QGis
    pp.create_bus(net,geodata=(x_gen,y_gen),vn_kv=20.0,min_vm_pu=0.9,max_vm_pu=1.1,name='Bus 179',index=320)
    #Create generator at the created node (200)
    pp.create_gen(net,320,p_mw=pmw_gen) #Change the p_mw, max and min value?
    return net

#Technical violation functions
    #Visual plot
def Plot(net,saveName = None): #Plot map with in red the problematic(s) line(s),node(s) or transformator(s)
    # plot network with color codes
    cmap_list_node = [((0,0.9), "orangered"), ((0.9,1.1), "blue"), ((1.1,2), "red")]
    cmap_n, norm_n = cmap_discrete(cmap_list_node)
    bc = create_bus_collection(net, size=70, cmap=cmap_n, norm=norm_n);
    
    cmap_list_line = [((0, 95), "green"),((95,100),"yellow"), ((100, 200), "red")]
    cmap_l, norm_l = cmap_discrete(cmap_list_line)
    lc = create_line_collection(net, cmap=cmap_l, norm=norm_l,use_bus_geodata=True)
    
    cmap_list_trafo = [((0, 95), "lightgreen"),((95,100),"gold"), ((100, 200), "tomato")]
    cmap_t, norm_t = cmap_discrete(cmap_list_trafo)
    tc = create_trafo_collection(net,cmap=cmap_t, norm=norm_t)
    
    draw_collections([bc,tc,lc])
    plt.savefig(saveName) if saveName !=None else None

    # check constraints violation - voltage - bus
def isVoltageViolation(net):
    vMin,vMax, Node = 0.9, 1.1, [] #Init
    for k,v in enumerate(net.res_bus['vm_pu']):
        if v>vMax or v<vMin: #if violation
            Node.append(k)  #save the bus index
    return Node

    # check constraints violation - current - line
def isCurrentViolation(net):
    iMax, Line = net.line['max_i_ka'], [] #Init
    for k,i in zip(net.line.index,net.res_line['i_ka']):
        if i>iMax[k]: #if violation
            Line.append(k) #save the line index
    return Line

    # check constraints violation - load - transformer
def isTransformerViolation(net):
    loadmax, Trafo = 100, []
    for k, t in enumerate(net.res_trafo['loading_percent']):
        if t>loadmax: #if violation
            Trafo.append(k) #save the trafo index
    return Trafo
    
    #Return boolean of "is there violation?" with the value of bus, line and transfo that the limiting factor.
def isThereViolation(net,plot=False,saveName=None):
    if plot:
        Plot(net,saveName)
    N,L,T = isVoltageViolation(net),  isCurrentViolation(net),  isTransformerViolation(net)
    if N!=[] or L!=[] or T!=[]:
        return [True, N, L , T]
    return [False, N, L, T]

#Function of connexion

def CreateLine20kV(net,nod1,nod2,plot=True,saveName=None):
    x1,y1 = float(net.bus_geodata['x'][net.bus_geodata.index==nod1]), float(net.bus_geodata['y'][net.bus_geodata.index==nod1])
    x2,y2 = float(net.bus_geodata['x'][net.bus_geodata.index==nod2]), float(net.bus_geodata['y'][net.bus_geodata.index==nod2])
    len_con = np.sqrt((x2-x1)**2+(y2-y1)**2)/300 #factor for convergence
    pp.create_line(net,nod1,nod2,length_km=len_con,std_type="243-AL1/39-ST1A 20.0")
    pp.runpp(net)
    return isThereViolation(net,plot=plot,saveName=None),len_con

    #Connect the generator to the network at bus "node". plot the line
def ConnexionGen(net, node, plot=True,saveName=None):
    x_nod = float(net.bus_geodata['x'][net.bus_geodata.index==node])
    y_nod = float(net.bus_geodata['y'][net.bus_geodata.index==node])
    x_gen, y_gen = 3410832.464,5368594.558 #Determine on QGis
    len_con = np.sqrt((x_gen-x_nod)**2+(y_gen-y_nod)**2)/300 #factor for convergence
    if network.bus.vn_kv[node] == network.bus.vn_kv[320]:
        pp.create_line(net,320,node,length_km=len_con,std_type="243-AL1/39-ST1A 20.0") #generator at bus 320
        T = False
    else:
        pp.create_bus(net,geodata=(x_gen+0.1,y_gen+0.1),vn_kv=110.0,min_vm_pu=0.9,max_vm_pu=1.1,name='Bus 180',index=321)
        pp.create_transformer(net,321,320,std_type="63 MVA 110/20 kV")
        pp.create_line(net,321,node,length_km=len_con,std_type="149-AL1/24-ST1A 110.0") #generator at bus 320
        T = True #say if a transformer was needed or not
    pp.runpp(net)
    return isThereViolation(net,plot=plot,saveName=saveName),len_con,T

def DoubleCon(net,node1,node2,plot=True,saveName=None):
    _,lent1,T1 = ConnexionGen(net,node1,plot=False)
    PB,lent2,T2 = ConnexionGen(net,node2,plot=plot,saveName=saveName)
    return PB,[lent1,lent2],np.max([T1,T2])
    

    #Change the problematic component (for now just lines)
def ChangeComp(net,id_node,id_line,id_trafo, pmw_gen=10,plot=True,saveName=None):
    net.gen.p_mw[0] = pmw_gen
    for l in id_line:
        fromb,tob,lenb = network.line['from_bus'][l],network.line['to_bus'][l],network.line['length_km'][l] #extract data
        network.line = network.line.drop(l) #drop the problematic line
        pp.create_line(net,fromb,tob,length_km=lenb,std_type="243-AL1/39-ST1A 20.0",index=l) #replace it
    Tr = [114,142,143]
    for t in id_trafo:
        fromb,tob = network.trafo['hv_bus'][Tr[t]],network.trafo['lv_bus'][Tr[t]] #extract data
        network.trafo = network.trafo.drop(Tr[t]) #drop the problematic trafo
        pp.create_transformer(net,fromb,tob,std_type="63 MVA 110/20 kV",index=Tr[t]) #replace it
    # for nodes: analyse the lines around the nodes to change them
    # Add the prices of change
    pp.runpp(net)
    return isThereViolation(network,plot=plot,saveName=saveName)

    #return max power of the added generator for the given network organisation
def MaxPow(net, p_min = 0, p_max = 30,N=100): 
    i=0
    while i<N:
        p_mw = (p_min+p_max)/2
        net.gen['p_mw'][0] = p_mw
        pp.runpp(net)
        if not isThereViolation(net)[0]:
            p_min = p_mw
        else:
            p_max = p_mw
        i+=1
    return net.gen['p_mw'][0]


#"Optimisation" functions
    #Function to calculate the max power achievable for all node connexions
    # Create a simple line, not a transformer
def MaxPowWithoutProb(net,p_max=25):
    MAX = []
    for j,i in enumerate(net.bus.index[:179]): #avoid doing the calculation with gen connected to himself
        net = Init()
        _,lenl,T = ConnexionGen(net,i,plot=False)
        MAX.append([i,MaxPow(net,p_max=p_max),lenl,T])
        if j%int(len(net.bus.index)/4)==0: #plot the advancement of the function every 25%
            print(f'{j/len(net.bus.index)*100}%')
    while [(maxi[0],i) for i,maxi in enumerate(MAX) if maxi[1]==p_max] !=[]: #divide loop to avoid divergence for low gen capacity node connexions
        for k,i in [(i,maxi[0]) for i,maxi in enumerate(MAX) if maxi[1]==p_max]:
            net = Init()
            ConnexionGen(net,i,plot=False)
            MAX[k][1]=MaxPow(net,p_max=p_max+10)
        p_max+=10
    return MAX

def MaxPowerWithoutProbDouble(net,node1,p_max=20): #Plot Max Power for a connected node1 to all the other in the network
    MAX = []
    for j,i in enumerate(net.bus.index[:179]): #avoid doing the calculation with gen connected to himself
        net = Init()
        _,LENGTH, T = DoubleCon(net,319,i,plot=False,saveName=None)
        MAX.append([i,MaxPow(net,p_max=p_max),LENGTH,T])
        # if j%int(len(net.bus.index)/4)==0: #plot the advancement of the function every 25%
        #     print(f'{j/len(net.bus.index)*100}%')
    while [(maxi[0],i) for i,maxi in enumerate(MAX) if maxi[1]==p_max] !=[]: #divide loop to avoid divergence for low gen capacity node connexions
        for k,i in [(i,maxi[0]) for i,maxi in enumerate(MAX) if maxi[1]==p_max]:
            net = Init()
            DoubleCon(net,319,i,plot=False,saveName=None)
            MAX[k][1]=MaxPow(net,p_max=p_max+10)
        p_max+=10
    return MAX

#-------------END FUNCTIONS-----------

#-----Initial network: Illinois200 case study with a generator to connect-----

#set the network
network = Init()
pp.runpp(network)
print(network)
#pandapower.to_excel(network, "AC_datas.xlsx")
pandapower.plotting.simple_plot(network);

# Exemple Case1
network = Init(pmw_gen = 30)
pp.runpp(network)
PB,length,T = ConnexionGen(network,319,plot=True,saveName=None)
print(f'Length = {length}km; Pb_bus={PB[1]} ; Pb_line={PB[2]} ; Pb_trafo={PB[3]} ; Transfo installed? = {T}' )
maxg = MaxPow(network,p_max=100); print(f'Max gen power = {maxg}')

#Exemple of change trafo
PB = ChangeComp(network,PB[1],PB[2],PB[3],pmw_gen=30,saveName=None);print(f'Pb_bus={PB[1]} ; Pb_line={PB[2]} ; Pb_trafo={PB[3]}')
maxg = MaxPow(network,p_max = 35); print(f'Max gen power = {maxg}')

#Exemple DoubleConnexion
network = Init(pmw_gen = 60)
pp.runpp(network)
PB,LENGTH, T = DoubleCon(network,318,30,saveName=None)
print(f'Length = {LENGTH[0]}km and {LENGTH[1]}km ; Pb_bus={PB[1]} ; Pb_line={PB[2]} ; Pb_trafo={PB[3]} ; Transfo installed? = {T}' )
maxg = MaxPow(network,p_max=70); print(f'Max gen power = {maxg}')

#Exemple other line in network
network = Init(pmw_gen=30)
pp.runpp(network)
_,length,T = ConnexionGen(network,110,plot=False,saveName=None)
_,length2,T = ConnexionGen(network,45,plot=False,saveName=None)
PB,length3 = CreateLine20kV(network,45,39,plot=False)
PB,length4 = CreateLine20kV(network,7,110,plot=True)
print(f'Length = {length}km and {length2} km and {length3}km and {length4}km; Pb_bus={PB[1]} ; Pb_line={PB[2]} ; Pb_trafo={PB[3]} ; Transfo installed? = {T}' )
maxg = MaxPow(network,p_max=60); print(f'Max gen power = {maxg}')


#Calculation of big excel files
    #Try to connect the generator to node1 close to the generator and all other nodes -> gives the combinaition with power>30
# SUP30=[]
# for i in MaxGen['Bus connected'][MaxGen.Length <=9.63].array: #bus close to generator: 35
#     MAX = MaxPowerWithoutProbDouble(network,i,p_max=10)
#     print(i)
#     for maxi in [MAX[i] for i in range(0,179) if MAX[i][1]>=30]:
#         SUP30.append([i,maxi])

#MAX = MaxPowWithoutProb(network)
#MaxGen = pd.read_excel('Max_gen_oberrhein.xlsx') #once the excel file is created, just import the data
#print([MAX[i] for i in range(0,179) if MAX[i][1]>20])
#MaxGen = pd.DataFrame(data = MAX)
#MaxGen.columns = ['Bus connected','MaxPower','Length','Trafo added']
#MaxGen.to_excel('Max_gen_oberrhein_20kV.xlsx') #save the datas
