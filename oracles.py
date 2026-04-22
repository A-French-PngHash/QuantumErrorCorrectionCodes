from qiskit import QuantumCircuit,transpile
import matplotlib.pyplot as plt
from itertools import combinations,permutations
import graycode
import math

__oracle_style = {
    "backgroundcolor": "#ffffff",      # fond blanc pur
    "textcolor": "#111111",            # texte sombre
    "fontsize": 12,                    # texte lisible
    "figwidth": 15,
    "dpi": 120,
    "linecolor": "#888888",            # contours des gates
    "gatefacecolor": "#e0f2fe",        # remplissage gates
    "displaycolor": {
        "h": "#38bdf8",                # portes H
        "x": "#f87171",                # portes X
        "cx": "#34d399",               # portes CX
        "measure": "#fbbf24",          # mesure
        "barrier": "#9ca3af"
    },
}



#---------- Oracle pour arrête ----------



#O(n+m)
def oracle_edge(graph,composed_mcx=True):
	"""
	Génére un oracle pour un graphe donné sous forme de liste d'adjacence.
	La gestion des qubits de l'oracle est la suivante (dans l'ordre):
	- registre 1 : sommet 1 (bit_n qubits) (plus petit bit en haut)
	- registre 2 : sommet 2 (bit_n qubits) (plus petit bit en haut)
	- registre de travail (2*bit_n - 2 qubits)
	- qubit résultat (1 qubits)
	
	On l'utilise avec:
	circuit.compose(oracle, qubits=[entré_registre_1]+[entré_registre_2]+[qubits_de_travail]+[qubit_résultat], inplace=True)

	:param graph: liste d'adjacence du graphe
	:param composed_mcx: si True, utilise des portes MCX sont décomposé en porte CCX (inactif ici)
	:return: oracle pour les arrêtes du graphe, bit_n, bit de travail
	"""
	#---------- Variables globales au programme ----------
	n = len(graph)
	bit_n = math.ceil(math.log2(n))
	oracle = QuantumCircuit((2*bit_n + 2*bit_n - 1) if composed_mcx else 2*bit_n+1, name="oracle") # registre 1 (bit_n) + registre 2 (bit_n) + qubit de travail avec le qubit de résultat à la fin(2*bit_n - 1) (résultat en dernier)
	last_number_register_2 = (1 << bit_n)-1 # registre qui garde le dernier nombre mis dans le second registre afin de pouvoir correctement y inscrire le nombre suivant
	
	#---------- Fonction pour un sommet ----------
	# Fonction interne pour ajouter les arêtes d'un sommet i en parcourant sa liste d'adjacence
	def ajoute_arrete_sommet_i(i):
		# Prérequis : le registre 1 contient le sommet i
		nonlocal last_number_register_2

		#On initialise les qubits de travail correspondant au registre 1 car celui-là de va pas changer
		if composed_mcx:
			if bit_n>1:
				oracle.ccx(0,1,2*bit_n)
			for k in range(bit_n-2):
				oracle.ccx(2+k,2*bit_n+k, 2*bit_n+1+k)

		for j in graph[i]:
			#On doit ajouter l'arête (i,j)

			# repègre les bits du registe 2 à changer
			bit_to_change= j ^ last_number_register_2
			bit_string = "{0:b}".format(bit_to_change)
			# Met le sommet i dans le registre 2 (pour qu'il soit reconnu)
			for k in range(len(bit_string)):
				if bit_string[-(k+1)] == '1':
					oracle.x(bit_n + k)
			last_number_register_2 = j
			# Mets le résultat à 1 pour les sommets correspondants aux deux registres
			# On utilise uniquement des portes de toffoli pour pouvoir appliquer les erreurs et les codes correcteurs d'erreurs
			# Equivalent à : oracle.mcx([k for k in range(2*bit_n)], 2*bit_n)
			# Voir schéma p.184 du livre "Quantum Computation and Quantum Information" de Nielsen et Chuang
			if composed_mcx:
				if bit_n==1: # Cas particulier
					oracle.ccx(0,1,2*bit_n)
					continue

				for k in range(bit_n-2,2*bit_n-2):
					oracle.ccx(2+k,2*bit_n+k, 2*bit_n+1+k)
				# On nettoie les qubits de travail sans nettoyer le résultat
				for k in range(bit_n-2,2*bit_n-3)[::-1]:
					oracle.ccx(2+k,2*bit_n+k, 2*bit_n+1+k)
				#Si bit_n == 1, 2*bit_n est le bit de résultat, il ne faut donc pas le nettoyer
				#oracle.ccx([0,1],2*bit_n)
			else:
				oracle.mcx([k for k in range(2*bit_n)], 2*bit_n)
		if composed_mcx:
			#On nettoie les qubits de travail du registre 1
			for k in range(bit_n-2)[::-1]:
				oracle.ccx(2+k,2*bit_n+k, 2*bit_n+1+k)
			if bit_n>1: #Si bit_n == 1, 2*bit_n est le bit de résultat, il ne faut donc pas le nettoyer
				oracle.ccx(0,1,2*bit_n)

	# ---------- Début programme ----------
	
	# Gray coding pour minimiser le nombre de portes utilisées
	counting = graycode.gen_gray_codes(bit_n)

	#Cas particulier du sommet 0
	if counting[0] == 0:
		oracle.x([i for i in range(bit_n)])
	else:
		raise Exception("Cas non prévu")
	ajoute_arrete_sommet_i(0)
	
	#On s'occupe des autres sommets
	for i in range(len(counting)-1):
		#Met couting[i+1] dans le registre 1 (sachant qu'avant il y avait counting[i])
		oracle.barrier()
		oracle.x(int(math.log2(abs(counting[i+1] - counting[i]))))
		#Lie le sommet i dans toutes ces arrêtes
		ajoute_arrete_sommet_i(counting[i+1])
	
	# Met le registre 1 à la valeur de départ
	last_counting = counting[-1] # Last gray code value
	i = 1
	for count in range(bit_n):
		if i & last_counting == 0: # If bit `i` is not in `last_counting` then we need to apply a X gate
			oracle.x(count)
		i = i << 1
	
	# Met le registre 2 à la valuer de départ
	i = 1
	for count in range(bit_n):
		if i & last_number_register_2 == 0: # If bit `i` is not in `last_counting` then we need to apply a X gate
			oracle.x(bit_n+count)
		i = i << 1

	return oracle,bit_n, (2*bit_n-2 if composed_mcx else 0)

def __test_oracle_edge_aux(oracle_graph,bit_n,registre1,registre2,resultat_attendu):
	circuit = QuantumCircuit(2*bit_n+2*bit_n-1,1)
	#Initialise registre 1
	i=1
	for count in range(bit_n):
		if i & registre1 != 0:
			circuit.x(count)
		i = i << 1
	#Initialise registre 2
	i=1
	for count in range(bit_n):
		if i & registre2 != 0:
			circuit.x(bit_n+count)
		i = i << 1
	circuit.barrier()
	circuit.compose(oracle_graph,qubits=range(4*bit_n-1),inplace=True)
	circuit.measure(4*bit_n-2,0)
	# Run and print results et then circuit
	simulator = AerSimulator()
	circuit = transpile(circuit, simulator)
	result = simulator.run(circuit).result()
	counts = result.get_counts(circuit)
	reussit = counts[str(resultat_attendu)] == 1024 if str(resultat_attendu) in counts else False
	print("✅" if reussit else "❌","Arrête : (",registre1,",",registre2,")","Attendu: ",resultat_attendu," Obtenue: ",counts)
	if not reussit:
		circuit.draw("mpl")
		plt.show()

def __test_oracle_edge():
	graph_0 = [[],[]]
	test_0 = [[0,1,0],[1,0,0]]
	graph_1 = [[1],[0]]
	test_1 = [[0,1,1],[1,0,1],[0,0,0],[1,1,0]]

	graph_2 = [[],[],[],[]]
	test_2 = [[i,j,0] for i in range(4) for j in range(4)]

	graph_3 = [[1],[0,2,3],[1,3],[1,2]]
	"""
	0   3
	| / |
	1 - 2
	"""
	test_3 = [[0,1,1],[1,3,1],[1,2,1],[1,1,0],[1,0,1],[2,0,0],[0,2,0],[3,1,1],[0,3,0]]

	tests = [[graph_0,test_0,"Vide"],[graph_1,test_1,"Complet"],[graph_2,test_2,"Vide"],[graph_3,test_3,"1 triangle"]]
	for [graph,list_values,name] in tests:
		oracle,bit_n,_ = oracle_edge(graph)
		print(name," - Bit_n:",bit_n)
		for i in range(len(graph)):
			for j in range(len(graph)):
				__test_oracle_edge_aux(oracle,bit_n,i,j,int(j in graph[i]))





#---------- Oracle pour triangle naïf (O(n^3)) ----------




@DeprecationWarning
def oracle_triangle_naif(N, edges):
    """
    Generates an oracle that detects if 3 vertices form a triangle.
    
    Args:
        N (int): Number of vertices in the graph (vertices are 0 to N-1)
        edges: List of edges, each as (u,v)
        
    Returns:
        QuantumCircuit: Oracle with 3*n_bits input qubits + 1 output qubit
                       Output qubit flips to |1⟩ if the 3 vertices form a triangle
    """
    
    n_bits = math.ceil(math.log2(N)) if N > 1 else 1
    
    #3 vertex registers + 1 output
    oracle = QuantumCircuit(3 * n_bits + 1, name="triangle_oracle")
    
    # Build undirected edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v))) 
    
    # Define qubit registers
    a_qubits = list(range(n_bits))
    b_qubits = list(range(n_bits, 2*n_bits))
    c_qubits = list(range(2*n_bits, 3*n_bits))
    output_qubit = 3*n_bits
    
    # Find all triangles in the graph (all 6 orderings)
    triangles = []
    for a, b, c in combinations(range(N), 3):
        # Check if all three edges exist
        if (min(a,b), max(a,b)) in edge_set and \
           (min(b,c), max(b,c)) in edge_set and \
           (min(a,c), max(a,c)) in edge_set:
            print(f"Triangle found: {min(a,b,c)}, {a+b+c-min(a,b,c)-max(a,b,c)}, {max(a,b,c)}")
            for perm in permutations([a, b, c]):
                triangles.append(perm)

    
    print(f"Found {len(triangles)} triangles in the graph")
    
    # Encode each triangle into the circuit
    for a, b, c in triangles:
        a_bin = format(a, f'0{n_bits}b')
        b_bin = format(b, f'0{n_bits}b')
        c_bin = format(c, f'0{n_bits}b')
        
        # apply X to qubits to make them |1⟩)
        for i, bit in enumerate(a_bin):
            if bit == '0':
                oracle.x(a_qubits[i])
        for i, bit in enumerate(b_bin):
            if bit == '0':
                oracle.x(b_qubits[i])
        for i, bit in enumerate(c_bin):
            if bit == '0':
                oracle.x(c_qubits[i])
        
        # multi-controlled NOT to output
        control_qubits = a_qubits + b_qubits + c_qubits
        oracle.mcx(control_qubits, output_qubit)
        
        # reverse the X gates
        for i, bit in enumerate(a_bin):
            if bit == '0':
                oracle.x(a_qubits[i])
        for i, bit in enumerate(b_bin):
            if bit == '0':
                oracle.x(b_qubits[i])
        for i, bit in enumerate(c_bin):
            if bit == '0':
                oracle.x(c_qubits[i])
    
    return oracle

def __test_oracle_triangle_naif():
	# 8 vertices: 0..7
    N = 8
    
    # Example edges forming a few triangles
    edges = {
        (0,1), (1,2), (0,2),      # triangle 0-1-2
        (3,4), (4,5), (3,5),      # triangle 3-4-5
        (6,7), (0,6), (0,7),      # triangle 0-6-7
        (2,3), (2,5)               # extra edges, not forming triangles
    }
    
    # Generate the oracle
    oracle = oracle_triangle_naif(N, edges)
    
    print(f"\nOracle circuit has {oracle.num_qubits} qubits")
    print(f"Circuit depth: {oracle.depth()}")
    print(f"Gate count: {oracle.size()}")
    
    # Visualize (text-based diagram)
    print("\nCircuit diagram:")
    print(oracle.draw(output='text'))





#---------- Oracle pour triangle (O(m+n)) ----------




def oracle_triangle(edge_oracle,bit_n,bit_travail_edge,composed_mcx=True):
	"""
	Génére un oracle pour triangle à partir d'un oracle pour arrête.
	La gestion des qubits de l'oracle est la suivante (dans l'ordre):
	- registre 1 : sommet 1 (bit_n qubits) (plus petit bit en haut)
	- registre 2 : sommet 2 (bit_n qubits) (plus petit bit en haut)
	- registre 3 : sommet 3 (bit_n qubits) (plus petit bit en haut)
	- registre de travail (2*bit_n + 1 qubits) (edge + triangle)
	- qubit résultat (1 qubits)
	
	On l'utilise avec:
	circuit.compose(oracle, qubits=[entré_registre_1]+[entré_registre_2]+[entré_registre_3]+[qubits_de_travail]+[qubit_résultat], inplace=True)

	:param edge_oracle: oracle pour les arrêtes du graphe
	:param bit_n: nombre de qubits nécessaires pour représenter les sommets du graphe
	:param bit_travail_edge: nombre de qubits de travail utilisés par edge_oracle
	:param composed_mcx: si True, utilise des portes MCX décomposées en CCX
	:return: oracle pour les arrêtes du graphe, bit_travail
	"""
	#Circuit avec : les 3 registres de sommets (3*bit_n), les 2 registres de travail de edge_oracle (2*bit_n-2),le registre de travail de triangle_oracle (3), le qubit de sortie (1)
	triangle_oracle = QuantumCircuit(3*bit_n+(bit_travail_edge)+3+1, name="Triangle Oracle")
		
	v_1= list(range(0,bit_n)) #Registre du sommet 1
	v_2= list(range(bit_n,2*bit_n)) #Registre du sommet 2
	v_3= list(range(2*bit_n,3*bit_n)) #Registre du sommet 3
	q_work_edge= list(range(3*bit_n,3*bit_n+bit_travail_edge)) #Registres de travail pour les appels à edge_oracle
	q_work_1= [3*bit_n+bit_travail_edge] #Registre de travail 1 pour triangle_oracle
	q_work_2= [3*bit_n+bit_travail_edge+1] #Registre de travail 2 pour triangle_oracle
	q_work_3= [3*bit_n+bit_travail_edge+2]   #Registre de travail 3 pour triangle_oracle
	q_out= [3*bit_n+bit_travail_edge+3] #Qubit de sortie pour triangle_oracle

	#Vérifie "e(v_1,v_2) et e(v_2,v_3)"", mets le résultat dans q_work_3 puis nettoie q_work_1
	triangle_oracle.compose(edge_oracle, v_1 + v_2 + q_work_edge + q_work_1,inplace=True)
	triangle_oracle.compose(edge_oracle, v_2 + v_3 + q_work_edge + q_work_2,inplace=True)
	triangle_oracle.ccx(q_work_1, q_work_2, q_work_3)
	triangle_oracle.compose(edge_oracle, v_1 + v_2 + q_work_edge + q_work_1,inplace=True)

	#Vérifie "e(v_1,v_3) et q_work_3", mets le résultat dans q_out puis nettoie q_work_1
	triangle_oracle.compose(edge_oracle, v_1 + v_3 + q_work_edge + q_work_1,inplace=True)
	triangle_oracle.ccx(q_work_1, q_work_3, q_out)
	triangle_oracle.compose(edge_oracle, v_1 + v_3 + q_work_edge + q_work_1,inplace=True)

	# Nettoie q_work_3 puis q_work_2
	triangle_oracle.compose(edge_oracle, v_1 + v_2 + q_work_edge + q_work_1,inplace=True)
	triangle_oracle.ccx(q_work_1, q_work_2, q_work_3)
	triangle_oracle.compose(edge_oracle, v_1 + v_2 + q_work_edge + q_work_1,inplace=True)
	triangle_oracle.compose(edge_oracle, v_2 + v_3 + q_work_edge + q_work_2,inplace=True)

	return triangle_oracle,bit_travail_edge+3



#---------- Tests ----------

if __name__=="__main__":
	#---------- Test oracle_edge ----------
	__test_oracle_edge()
	#---------- Test oracle_triangle_naif ----------
	__test_oracle_triangle_naif()