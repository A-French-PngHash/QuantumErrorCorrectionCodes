from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
import qiskit.visualization as qv

# Convention de nommage: graph_[n_bit]_[nom ou id]

graph_1_vide = [[],[]]
graph_1_complet = [[1],[0]]

graph_2_vide = [[],[],[],[]]
graph_2_triangle = [[1],[0,2,3],[1,3],[1,2]]
"""
0   3
| / |
1 - 2
"""
graph_2_complet = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
"""
0 - 3
| X |
1 - 2
"""

graph_3_triangle_345 = [
	[2,6,7],#0
	[2],	#1
	[0,1,5],#2
	[4,5],	#3
	[3,5],	#4
	[2,3,4],#5
	[0],	#6
	[0]		#7
]
graph_3_triangle_345_067 = [
	[2,6,7],#0
	[2],	#1
	[0,1,5],#2
	[4,5],	#3
	[3,5],	#4
	[2,3,4],#5
	[0,7],	#6
	[0,6]	#7
]
graph_3_complet = [
	[1,2,3,4,5,6,7],#0
	[0,2,3,4,5,6,7],#1
	[0,1,3,4,5,6,7],#2
	[0,1,2,4,5,6,7],#3
	[0,1,2,3,5,6,7],#4
	[0,1,2,3,4,6,7],#5
	[0,1,2,3,4,5,7],#6
	[0,1,2,3,4,5,6] #7
]

graph_4_345_067 =[
	[2,6,7],	#0
	[2,8,10],	#1
	[0,1,5],	#2
	[4,5],		#3
	[3,5],		#4
	[2,3,4],	#5
	[0,7,11],	#6
	[0,6,12],	#7
	[1,9],		#8
	[8,10,15],	#9
	[9,1],		#10
	[6,12,13],	#11
	[7,11,14,15],#12
	[11,14],	#13
	[12,13],	#14
	[12,9]		#15
]

style = {
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


def run_circuit(circuit,shots=1024):
	"""
	Exécute un circuit Qiskit sur un simulateur Aer et retourne les counts

	:param circuit: circuit Qiskit à exécuter
	:return: résultat sous forme de counts Qiskit
	"""
	simulator = AerSimulator()
	circuit = transpile(circuit, simulator)

	result = simulator.run(circuit,shots=shots).result()
	counts = result.get_counts(circuit)

	return counts

def bitstring_to_triangle(bitstring,little_endian=True,ordered=True):
		n = len(bitstring) // 3
		if little_endian:
			a = int(bitstring[n-1::-1], 2)
			b = int(bitstring[2*n-1:n-1:-1], 2)
			c = int(bitstring[:2*n-1:-1], 2)
		else:
			a = int(bitstring[:n], 2)
			b = int(bitstring[n:2*n], 2)
			c = int(bitstring[2*n:], 2)
		if not ordered:
			return str(a)+","+str(b)+","+str(c)
		x=min(a,b,c)
		z=max(a,b,c)
		y=a+b+c - x - z
		return str(x)+","+str(y)+","+str(z)
def vertex_to_bitstring(vertex,bit_n,little_endian=True):
	if little_endian:
		return format(vertex, '0{}b'.format(bit_n))[::-1]
	else:
		return format(vertex, '0{}b'.format(bit_n))

def is_triangle(str,graph):
		sommet = str.split(",")
		a=int(sommet[0])
		b=int(sommet[1])
		c=int(sommet[2])
		return (a in graph[b]) and (a in graph[c]) and (b in graph[c])

def triangle_counts_from_bitstrings_counts(counts, big_endian=True,ordered=True):
	"""
	Mets en forme le résultat d'un circuit grover cherchant des triangles
	L'entier est interprété depuis le binaire en fonction de la convention (paramètre big_endian) choisie

	:param counts: liste du résultat d'un circuit grover cherchant des triangles
	:param big_endian: convention de numérotation des sommets,True ou False en fonction de Little Endian ou Big Endian
	:return: résultat où les triangles sont explicites sous forme de chaînes "a,b,c" (a<b<c)
	"""
	triangle_counts = {}

	

	#Le qubit qui ne sert à rien est en position 0
	for bitstring, count in counts.items():
		triangle= bitstring_to_triangle(bitstring, big_endian,ordered)
		if triangle not in triangle_counts:
			triangle_counts[triangle] = 0
		triangle_counts[triangle] += count  # exclude output qubit

	return triangle_counts