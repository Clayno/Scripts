#!/usr/bin/python3

import argparse

def print_poly(f):
	poly = ''
	for i in range(0,len(f)):
		if i == 0:
			poly += "1"
		elif f[int(i)] == 1:
			poly += ' + '+'X^'+str(i)
	return poly


def bm(s):
	n = len(s)
	g = [0]*n
	f = [0]*n
	# g(X) = (g(i)X^i, i in (0, n))
	g[0] = 1
	# f(X) = (f(i)X^i, i in (0, n))
	f[0] = 1

	L = 0
	m = -1
	N = 0

	while N < n :

		d = sum( int(s[N-i], 2) & f[i] for i in range(0, L+1)) % 2
		
		if d == 1:
			t = f
			# f(X) = f(X) + g(X)X^(N-m)
			if N-m > 0:
				f = f + [0]*(N-m)
			for i in range(N-m, N-m+n) :
				f[i] = f[i]^g[i-N+m]
			if 2*L <= N:
				L = N+1-L
				m = N
				g = t
		if args.verbose != None :
			poly = print_poly(f)
			print("N=", N, "sn=", s[N], " d=", d, " L=", L, " m=", m, "f=", poly, " g=", print_poly(g))
		N += 1

	poly = print_poly(f)
	fun = [i for i in f[1:L+1]]
	return (fun, poly, L)
 

def test(seq, fun):
	seed = [int(i) for i in reversed(seq)][len(seq)-L:]
	assert(len(seed) == len(fun))
	output = ''
	for i in range(len(seq)):
	    print("Tour ", i, ' seed : ', seed)
	    print('Output : ', seed[-1], ' seq: ', seq[i], end=' ')
	    if seed[-1] == int(seq[i]):
	    	print("\033[92mOK\033[0m")
	    else:
	    	print("\033[91mNOT OK\033[0m")
	    feedback = 0
	    output += str(seed[-1])
	    feedback = sum(seed[i] * fun[i] + feedback for i in range(len(seed))) % 2
	    seed = [feedback] + seed[:-1]

	result = bin(int(output,2))[2:]
	return result

if __name__ == '__main__':
	#seq = '1100101011111110011011110110101101001101001110010111100111001010'
	parser  = argparse.ArgumentParser(description="Script implementing the Berlekamp-Massey algorithm to calculate the polynom of minimal length calculating the sequence entered.")
	parser.add_argument('sequence', action='store', help="Sequence of bits generated by the LFSR you want to find the polynom of. (Example: '11001'")
	parser.add_argument('-v', '--verbose', action='count', help='Be verbose during the polynom generation.')
	parser.add_argument('-t', '--test', action='count', help='Test the generated polynom.')
	args = parser.parse_args() 
	seq = args.sequence
	(fun, poly, L) = bm(seq)
	print("Minimal length: ", L)
	print("Poynôme: ", poly)
	print("Feedback function: ", fun)
	if args.test != None:
		print("Test: \n")
		print("Result: ", test(seq, fun))

