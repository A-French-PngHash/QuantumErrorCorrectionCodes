import random
import math
from matplotlib import pyplot as plt

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit,transpile

from oracles import oracle_edge, oracle_triangle
from tools import bitstring_to_triangle,run_circuit,is_triangle, triangle_counts_from_bitstrings_counts

import sys
from pathlib import Path

# Add parent directory to Python path
code_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, code_dir)

from amplitude_amplification import build_mcx


#---------- Amplitude amplification circuits ----------


def generate_q(A:QuantumCircuit,oracle:QuantumCircuit,bit_entree:int,bit_reserve_oracle:int,bit_travail_oracle:int,bit_travail_a:int,composed_mcx:bool)->QuantumCircuit:
	"""
	Génere la porte Q = A S_0 A^-1 U_f pour l'amplification d'un algorithme A avec un oracle U_f, en fonction du nombre de bits d'entrée et de travail de chacun
	
	:param A: Algorithme to amplify
	:type A: QuantumCircuit
	:param oracle: Oracle U_f
	:type oracle: QuantumCircuit
	:param bit_entree: Nombre de bit de la donnée à amplifier
	:type bit_entree: int
	:param bit_reserve_oracle: Nombre de bits de réserve de l'oracle (notamment pour triangle_finder_two_gate)
	:type bit_reserve_oracle: int
	:param bit_travail_oracle: Nombre de bits de travail de l'oracle
	:type bit_travail_oracle: int
	:param bit_travail_a: Nombre de bits de travail de l'algorithme A
	:type bit_travail_a: int
	:param composed_mcx: si True, utilise des portes MCX décomposées en CCX
	:type composed_mcx: bool
	:return: La porte Q = A S_0 A^-1 U_f, ses bits d'entrée, ses bits de travail
	:rtype: QuantumCircuit,int,int

	Dans l'ordre : bit_entree, bit_reserve_oracle, bit_travail (max(bit_travail_oracle,bit_travail_a,bit_travail_toffoli)), 1 bit pour l'amplification (|->)
	"""
	bit_oracle = bit_entree + bit_reserve_oracle + max(bit_travail_oracle,bit_travail_a,(bit_entree-1-2 if composed_mcx else 0))  #On a besoin de bit_entre-3 bits de travail pour le mcx (-1 car mcz et [:bit_entree-1] avec pour résultat bit_entree-1 et -2 car besoins de la pyramide de mcx)
	qubits_a = list(range(bit_entree))+list(range(bit_entree+bit_reserve_oracle,bit_entree+bit_reserve_oracle+bit_travail_a))
	Q=QuantumCircuit(bit_oracle+1,name="QAA") # +1 pour le qubit |-> pour l'inversion de phase
	inv_a=A.inverse()

	#On applique l'oracle (bit_entree, les bits de travail et le bit |->)
	Q.compose(oracle,qubits=list(range(bit_entree+bit_reserve_oracle+bit_travail_oracle))+[bit_oracle],inplace=True)
		
	Q.barrier()
	
	#On applique l'amplificateur A^-1 sur les bits d'entrée
	Q.compose(inv_a,qubits=qubits_a,inplace=True)
	
	Q.barrier()

	#S_0
	Q.x(list(range(bit_entree)))
	# mcz (car Z = HXH)
	Q.h(bit_entree-1)

	#mcx_circuit = build_mcx(list(range(bit_entree-1)), bit_entree-1, list(range(bit_entree,bit_oracle)))
	#Q.compose(mcx_circuit, qubits=range(bit_oracle), inplace=True)
	if composed_mcx:
		bit_debut_travail = bit_entree + bit_reserve_oracle
		if(bit_entree>=4):
			Q.ccx(0,1,bit_debut_travail)
			for i in range(2,bit_entree-2):
				Q.ccx(i,bit_debut_travail+i-2,bit_debut_travail+i-1)
			Q.ccx(bit_entree-2,bit_debut_travail+bit_entree-3-1,bit_entree-1) #bit_entree+bit_entree-2-1 -> dernier bit de travail utilisé
			#On défait la pyramide de mcx
			for i in range(2,bit_entree-2)[::-1]:
				Q.ccx(i,bit_debut_travail+i-2,bit_debut_travail+i-1)
			Q.ccx(0,1,bit_debut_travail)
		elif bit_entree==3:
			Q.ccx(0,1,bit_entree-1)
		elif bit_entree==2:
			Q.cx(0,bit_entree-1)
		elif bit_entree==1:
			Q.x(0)
	else:
		Q.mcx(list(range(bit_entree-1)), bit_entree-1)


	Q.h(bit_entree-1)

	Q.x(list(range(bit_entree)))

	Q.barrier()

	Q.compose(A,qubits=qubits_a,inplace=True)

	return Q,bit_entree,bit_oracle-(bit_entree+bit_reserve_oracle)

def h_gate(bit_entree:int)->QuantumCircuit:
	H=QuantumCircuit(bit_entree,name="H") 
	H.h(list(range(bit_entree)))
	return H

def iteration_from_a(a:float):
	if a>1:
		print("Warning: a should be <=1, but got a=",a)
		return 0
	theta_a = math.asin(math.sqrt(a))
	#theta_a = math.sqrt(a) #pour a petit, asin(sqrt(a)) ~ sqrt(a)
	return math.floor(math.pi /(4*theta_a))

def amplitude_amplification_with_known_iteration(A:QuantumCircuit,oracle:QuantumCircuit,bit_entree:int,bit_reserve_oracle:int,bit_travail_oracle:int,bit_travail_a:int,n:int,composed_mcx):
	Q,bit_entree,bit_travail = generate_q(A,oracle,bit_entree,bit_reserve_oracle,bit_travail_oracle,bit_travail_a,composed_mcx)
	Q.barrier()
	iteration = n
	bit_oracle: int = bit_entree + bit_reserve_oracle + bit_travail
	qc = QuantumCircuit(bit_oracle + 1)  # grover qubit + output bit

	#On applique A sur les bits d'entrée
	qc.compose(A,qubits=list(range(bit_entree))+list(range(bit_entree+bit_reserve_oracle,bit_entree+bit_reserve_oracle+bit_travail_a)),inplace=True)
	
	qc.barrier()
	#On initialise le qubit |-> 
	qc.x(bit_oracle)
	qc.h(bit_oracle)
	for _ in range(iteration):
		qc.compose(Q, qubits=range(bit_oracle + 1), inplace=True)
	#On nettoie le bit de travail |->
	qc.h(bit_oracle)
	qc.x(bit_oracle)

	return qc,bit_oracle+1

def amplitude_amplification_with_known_a(A:QuantumCircuit,oracle:QuantumCircuit,bit_entree:int,bit_reserve_oracle:int,bit_travail_oracle:int,bit_travail_a:int,a:float,composed_mcx:bool=True):
	return amplitude_amplification_with_known_iteration(A,oracle,bit_entree,bit_reserve_oracle,bit_travail_oracle,bit_travail_a,iteration_from_a(a),composed_mcx)


def amplitude_amplification_with_known_iteration_measure(A,oracle,bit_entree:int,bit_reserve_oracle:int,bit_travail_oracle:int,bit_travail_a:int,n:int,composed_mcx:bool=True):
	QAA,bit_circuit = amplitude_amplification_with_known_iteration(A,oracle,bit_entree,bit_reserve_oracle,bit_travail_oracle,bit_travail_a,n,composed_mcx)
	qc = QuantumCircuit(bit_circuit,bit_entree)  # grover qubit + output bit
	qc.compose(QAA,range(bit_circuit),inplace=True)
	qc.measure(list(range(bit_entree)), list(range(bit_entree)))  # measure all input qubits
	return qc

def amplitude_amplification_with_known_a_measure(A,oracle,bit_entree:int,bit_reserve_oracle:int,bit_travail_oracle:int,bit_travail_a:int,a:float,composed_mcx:bool=True):
	return amplitude_amplification_with_known_iteration_measure(A,oracle,bit_entree,bit_reserve_oracle,bit_travail_oracle,bit_travail_a,iteration_from_a(a),composed_mcx)

def amplitude_amplification_with_unknown_a_measure(A,oracle, bit_entree, bit_reserve_oracle, bit_travail_oracle,bit_travail_a,f, c:float=1.5,composed_mcx:bool=True,debug:bool=True):
	def get_result(qc):
		counts = run_circuit(qc,shots=1)
		assert len(counts)==1
		r = list(counts.keys())[0]
		return r
	
	l=0

	#Optimisation, on pré-transpile les circuits
	simulator = AerSimulator()
	A = transpile(A, simulator)
	oracle = transpile(oracle, simulator)

	#Pour obtenir le circuit A|0>
	qc1 = QuantumCircuit(bit_entree+bit_travail_a,bit_entree)
	qc1.compose(A,qubits=range(bit_entree+bit_travail_a),inplace=True)
	qc1.measure(range(bit_entree), range(bit_entree))

	while True:
		l+=1
		M = math.ceil(c**l)
		j = random.randint(0, M)

		if debug:
			print(f"Iteration {l}, M={M}, j={j}", flush=True)
		result1 = get_result(qc1)
		if f(result1):
			if debug:
				print("Found triangle with 0 iterations", flush=True)
			return result1
		
		qc2_grov, bit_circuit= amplitude_amplification_with_known_iteration(A,oracle, bit_entree, bit_reserve_oracle, bit_travail_oracle,bit_travail_a,j,composed_mcx)
		qc2 = QuantumCircuit(bit_circuit,bit_entree)
		qc2.compose(qc2_grov,range(bit_circuit),inplace=True)
		qc2.measure(range(bit_entree), range(bit_entree))

		result2 = get_result(qc2)
		if f(result2):
			if debug:
				print(f"Found triangle with {j} iterations", flush=True)
			return result2


#---------- Triangle finder circuits ----------


def triangle_finder_one_gate(graph,composed_mcx=True)->QuantumCircuit:
	"""
	
	—a (bit_n)————————————┌———┐—
	—b (bit_n)————————————| 1 |—
	—travail (2*bit_n-2)——|   |—
	-travail (grover) (1)—└———┘—

	:param graph: Graphe en liste d'adjacence
	:param composed_mcx: si True, utilise des portes MCX décomposées en CCX
	:return: Circuit quantique cherchant une arête dans le graphe, bit_travail

	"""
	#Nombre de sommets
	n = len(graph)
	#Nombre d'arrêtes
	m=0
	for v in graph:
		m+=len(v)
	m=m/2

	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph,composed_mcx)

	#Grover pour trouver une arête (sur les 2 premiers registres)

	grover_find_arete = amplitude_amplification_with_known_a(h_gate(2*bit_n),oracle_arrete,2*bit_n,0,bit_travail_arrete,0,2*m/(2**(2*bit_n)),composed_mcx)
	return grover_find_arete

def triangle_finder_two_gate(graph,nb_triangle:int,composed_mcx:bool=True)->QuantumCircuit:
	"""
	—a (bit_n)———————————┌———┐—
	—b (bit_n)———————————|   |—
	—c (bit_n)———————————| 2 |—
	—travail (2*bit_n+1)—|   |—
	-travail (grover)(1)—└———┘—
	
	:param graph: Graphe en liste d'adjacence
	:param nb_triangle: Nombre de triangles par arête espéré
	
	:return: Circuit quantique cherchant un sommet a formant un triangle dans le graphe avec b et c

	"""

	#Nombre de sommets
	n = len(graph)
	#Nombre d'arrêtes
	m=0
	for v in graph:
		m+=len(v)
	m=m/2

	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph,composed_mcx)
	oracle,bit_travail_triangle = oracle_triangle(oracle_arrete,bit_n,bit_travail_arrete,composed_mcx)

	#Grover pour trouver une arête (sur les 2 premiers registres)

	#Porte H uniquement sur a

	grover_find_triangle_from_arete = amplitude_amplification_with_known_a(h_gate(bit_n),oracle,bit_n,2*bit_n,bit_travail_triangle,0,nb_triangle/(2**(bit_n)),composed_mcx) #On travaille sur a (bit_n), l'oracle réserve 2*bit_n (b et c), puis bit_travail_triangle en bits de travail, et on a pas de bit de travail pour l'amplification

	return grover_find_triangle_from_arete

def triangle_finder_one_and_two(graph,nb_triangle:int,composed_mcx:bool=True)->QuantumCircuit:
	"""
	—a (bit_n)————————————————┌—┐—\n
	—b (bit_n)———————————┌—┐——| |—\n
	—c (bit_n)———————————|1|——|2|—\n
	—travail (2*bit_n+1)—| |——| |—\n
	-travail (grover)(1)—└—┘——└—┘—\n
	
	:param graph: Graphe en liste d'adjacence
	:param nb_triangle: Nombre de fois à appliquer l'amplification pour la porte 2 (nombre de triangles par arête)
	
	:return: Circuit quantique cherchant un triangle dans le graphe

	"""

	#Nombre de sommets
	n = len(graph)
	#Nombre d'arrêtes
	m=0
	for v in graph:
		m+=len(v)
	m=m/2

	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph,composed_mcx)
	oracle,bit_travail_triangle = oracle_triangle(oracle_arrete,bit_n,bit_travail_arrete,composed_mcx)

	tf_one_gate,bit_circuit_tf_one = triangle_finder_one_gate(graph,composed_mcx)
	tf_two_gate,bit_circuit_tf_two = triangle_finder_two_gate(graph,nb_triangle,composed_mcx)

	bit_circuit_tot = max(bit_n+bit_circuit_tf_one,bit_circuit_tf_two)

	A = QuantumCircuit(bit_circuit_tot,name="A")
	#On trouve l'arrête
	A.compose(tf_one_gate,qubits=list(range(bit_n,bit_n+bit_circuit_tf_one)),inplace=True)
	#On trouve le troisième sommet formant un triangle avec cette arrête
	A.compose(tf_two_gate,qubits=list(range(bit_circuit_tf_two)),inplace=True) 
	#On amplifie l'ensemble
	return A,bit_circuit_tot

#def triangle_finder_one_and_two_measure(graph,nb_triangle:int)->QuantumCircuit:


def triangle_finder_fixed_a_for_gate_two(graph,nb_triangle,c,composed_mcx:bool=True):
	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph,composed_mcx)
	oracle,bit_travail_triangle = oracle_triangle(oracle_arrete,bit_n,bit_travail_arrete,composed_mcx)


	A,bit_circuit_a = triangle_finder_one_and_two(graph,nb_triangle,composed_mcx)

	def f(bitstring):
		return is_triangle(bitstring_to_triangle(bitstring,False),graph)

	return amplitude_amplification_with_unknown_a_measure(A,oracle,3*bit_n,0,bit_travail_triangle,bit_circuit_a-3*bit_n,f,c,composed_mcx)


#---------- Base ----------


def groover_gate(oracle:QuantumCircuit, bit_entree:int,bit_travail:int,composed_mcx=True)->QuantumCircuit:
	return generate_q(h_gate(bit_entree),oracle,bit_entree,0,bit_travail,0,composed_mcx)

def triangle_finder_naive_gate_measure(graph,composed_mcx=True):
	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph,composed_mcx)
	oracle,bit_travail_triangle = oracle_triangle(oracle_arrete,bit_n,bit_travail_arrete,composed_mcx)

	return amplitude_amplification_with_known_iteration_measure(h_gate(3*bit_n),oracle,3*bit_n,0,bit_travail_triangle,0,1,composed_mcx)

def triangle_finder_naive(graph,c,composed_mcx=True):
	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph,composed_mcx)
	oracle,bit_travail_triangle = oracle_triangle(oracle_arrete,bit_n,bit_travail_arrete,composed_mcx)

	def f(bitstring):
		return is_triangle(bitstring_to_triangle(bitstring,False),graph)
	
	A = h_gate(3*bit_n)

	return amplitude_amplification_with_unknown_a_measure(A,oracle,3*bit_n,0,bit_travail_triangle,0,f,c,composed_mcx)


#---------- Tets ----------



def __test_amplitude_amplification_with_known_a():
	from tools import run_circuit,triangle_counts_from_bitstrings_counts,graph_3_triangle_345
	import qiskit.visualization as qv

	oracle_arrete,bit_n = oracle_edge(graph_3_triangle_345)
	oracle = oracle_triangle(oracle_arrete,bit_n)

	a = QuantumCircuit(3*bit_n,name="A")
	a.h(list(range(3*bit_n)))

	qc = amplitude_amplification_with_known_a_measure(a,oracle, 3*bit_n,2*bit_n+1,0,6/(2**(3*bit_n)))
	print("Running circuit...", flush=True)
	result = run_circuit(qc)
	qv.plot_histogram(result)
	#plt.savefig("grover_result.png")
	triangle_counts = triangle_counts_from_bitstrings_counts(result,False)
	qv.plot_histogram(triangle_counts)
	plt.show()

def __test_triangle_finder():
	from tools import graph_2_triangle,graph_3_triangle_345,graph_2_complet,graph_3_complet,graph_4_345_067
	import qiskit.visualization as qv

	def aux(graph,triangle_suppose):
		print("Rapide:")
		triangle = triangle_finder_fixed_a_for_gate_two(graph,1,1.5,False)
		print("Naif:")
		_  = triangle_finder_naive(graph,1.5)
		print(f"Triangle {triangle_suppose}:", bitstring_to_triangle(triangle,False))
		print("-------\n")
	
	aux(graph_2_complet,"2 complet")
	aux(graph_2_triangle,"123")
	aux(graph_3_complet,"3 complet")
	aux(graph_3_triangle_345,"345")
	aux(graph_3_triangle_345_067,"345 et 067")
	aux(graph_4_345_067,"345 et 067")

	


def __graph_proba_triangle_finder_fixed_a_for_gate_two(graph, nb_triangle):
	oracle_arrete,bit_n,bit_travail_arrete = oracle_edge(graph)
	oracle,bit_travail_triangle = oracle_triangle(oracle_arrete,bit_n,bit_travail_arrete)


	A = triangle_finder_one_and_two(graph,nb_triangle)
	bit_travail_A = bit_travail_triangle + 1

	outputs = []

	for i in range(1,12):
		circuit = amplitude_amplification_with_known_iteration_measure(A,oracle,3*bit_n,bit_travail_triangle,bit_travail_A,i)
		counts = run_circuit(circuit,shots=1024)
		proba = sum(counts[k] for k in counts if is_triangle(bitstring_to_triangle(k,False),graph))/1024
		outputs.append((i,proba))
		print(f"i={i}, proba={proba}", flush=True)
	plt.plot([x[0] for x in outputs],[x[1] for x in outputs])
	plt.xlabel("Nombre d'itérations")
	plt.ylabel("Probabilité de trouver un triangle")
	plt.title(f"nb_triangle={nb_triangle}")

def __graph_proba_A_triangle_finder_gate_one_and_two(graph, nb_triangle):
	import qiskit.visualization as qv
	from tools import run_circuit,triangle_counts_from_bitstrings_counts
	

	n = len(graph)
	bit_n = math.ceil(math.log2(n))
	A,bit_circuit = triangle_finder_one_and_two(graph,nb_triangle)

	circuit = QuantumCircuit(bit_circuit,3*bit_n)
	circuit.compose(A,range(bit_circuit),inplace=True)
	circuit.measure(range(3*bit_n), range(3*bit_n))
	counts = run_circuit(circuit,shots=10024)
	tri_counts = triangle_counts_from_bitstrings_counts(counts,False)

	qv.plot_histogram(tri_counts)
	plt.title(f"Distribution des triangles trouvés par A (nb_triangle={nb_triangle})")
	plt.show()


def __graph_tri_proba_A_triangle_finder_gate_one_and_two(graph):
	import qiskit.visualization as qv
	from tools import run_circuit,triangle_counts_from_bitstrings_counts
	

	n = len(graph)
	bit_n = math.ceil(math.log2(n))
	outputs = []
	temoin = triangle_finder_naive_gate_measure(graph)
	counts_temoin = run_circuit(temoin,shots=2048)
	proba_temoin = sum(counts_temoin[k] for k in counts_temoin if is_triangle(bitstring_to_triangle(k,False),graph))/2048
	print(f"Proba de trouver un triangle avec une itération groover naif : {proba_temoin}", flush=True)

	for i in range(1,15):
		A,_ = triangle_finder_one_and_two(graph,i)

		circuit = QuantumCircuit(A.num_qubits,3*bit_n)
		circuit.compose(A,range(A.num_qubits),inplace=True)
		circuit.measure(range(3*bit_n), range(3*bit_n))
		counts = run_circuit(circuit,shots=1024)
		tri_counts = triangle_counts_from_bitstrings_counts(counts,False)
		proba = sum(counts[k] for k in counts if is_triangle(bitstring_to_triangle(k,False),graph))/1024
		outputs.append((i,proba))
		print(f"nb_tri={i}, proba={proba}", flush=True)
	plt.plot([x[0] for x in outputs],[x[1] for x in outputs],[proba_temoin for _ in outputs])
	plt.xlabel("nb_triangle")
	plt.ylabel("Probabilité de trouver un triangle")
	plt.title(f"Distribution des triangles trouvés par A en fonction de nb_triangle")
	plt.show()
def __test__triangle_finder_fixed_a_for_gate_two(graph,a,b,nb_triangle):
	from tools import vertex_to_bitstring,run_circuit,triangle_counts_from_bitstrings_counts
	import qiskit.visualization as qv
	bit_n=math.ceil(math.log2(len(graph)))
	circuit_test = QuantumCircuit(3*bit_n+2*bit_n+1+1,3*bit_n)

	#préparation de l'état |0>|a>|b>
	bit_a = vertex_to_bitstring(a,bit_n,True)
	bit_b = vertex_to_bitstring(b,bit_n,True)
	for i in range(bit_n):
		if bit_a[i]=="1":
			circuit_test.x(bit_n+i)
		if bit_b[i]=="1":
			circuit_test.x(2*bit_n+i)

	tri,bit_circuit =triangle_finder_two_gate(graph,nb_triangle)
	circuit_test.compose(tri,range(bit_circuit),inplace=True)
	circuit_test.measure(range(3*bit_n), range(3*bit_n))

	counts = run_circuit(circuit_test,shots=1024)
	tri_counts = triangle_counts_from_bitstrings_counts(counts,False)
	qv.plot_histogram(tri_counts)
	plt.title(f"Distribution des triangles trouvés par la porte 2 à partir de l'arrête ({a},{b}) (nb_triangle={nb_triangle})")
	plt.show()


def __test_gate_one(graph):
	import qiskit.visualization as qv
	from tools import run_circuit,triangle_counts_from_bitstrings_counts
	

	n = len(graph)
	bit_n = math.ceil(math.log2(n))
	outputs = []
	gate_one,_ = triangle_finder_one_gate(graph,True)
	circuit_temoin = QuantumCircuit(gate_one.num_qubits,3*bit_n)
	circuit_temoin.compose(gate_one,range(gate_one.num_qubits),inplace=True)
	circuit_temoin.measure(range(3*bit_n), range(3*bit_n))
	counts_temoin = run_circuit(circuit_temoin,shots=2048)
	triangle_counts = triangle_counts_from_bitstrings_counts(counts_temoin,False,False)
	qv.plot_histogram(triangle_counts)
	plt.show()


	



if __name__ == "__main__":
	from tools import graph_3_triangle_345,graph_2_complet,graph_2_triangle,graph_3_complet,graph_3_triangle_345_067,graph_4_345_067
	import qiskit.visualization as qv
	#__graph_proba_A_triangle_finder_gate_one_and_two(graph_3_triangle_345,4)
	#__graph_proba_A_triangle_finder_gate_one_and_two(graph_3_triangle_345,5)

	#__test__triangle_finder_fixed_a_for_gate_two(graph_3_triangle_345,3,4,1)
	__test_triangle_finder()
	gate = triangle_finder_naive_gate_measure(graph_2_complet,False)
	counts = run_circuit(gate,shots=2048)
	counts = triangle_counts_from_bitstrings_counts(counts,False)
	qv.plot_histogram(counts)
	plt.show()
	# exit(0)
	# graphs = [graph_2_complet,graph_2_triangle,graph_3_complet,graph_3_triangle_345,graph_3_triangle_345_067]
	# graphs_names = ["graph_2_complet","graph_2_triangle","graph_3_complet","graph_3_triangle_345","graph_3_triangle_345_067"]
	# for i in range(len(graphs)):
	# 	print(f"Testing {graphs_names[i]}")
	# 	__graph_tri_proba_A_triangle_finder_gate_one_and_two(graphs[i])

	#__test_triangle_finder()

	#Note: __graph_tri_proba_A_triangle_finder_gate_one_and_two(graph_4_345_067) semble s'être accéléer pour nb_tri \in [10,15] (très faible probabilité)