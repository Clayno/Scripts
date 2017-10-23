import requests
import re
import base64
import binascii
import urllib.parse
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("url", help="Url de l'oracle", type=str)
parser.add_argument("cipher", help="Cipher a dechiffrer", type=str)
parser.add_argument("block_size", help="Taille de bloc utilise", type=int)
parser.add_argument("type", help="Type d'encodage utilise. 0 = Base64, 1 = Hexadecimal", type=int)
parser.add_argument("error", help="Message d'erreur", type=str)
parser.parse_args()
args = parser.parse_args()

print(args)
block_size = 16
url = args.url
flagToNormalize = args.cipher


def normalizeFlag(flag, type) :
	if type == 0 : # base64
		flagB64 = urllib.parse.unquote(flagQB64)
		bFlagB64 = flagB64.encode()
		bFlagX = base64.b64decode(bFlagB64)
		bFlag = binascii.hexlify(bFlagX)
		return bFlag.decode()
	elif type == 1 : # Hexadecimal
		return flag

def oracle(flag, error):
	req = requests.get(url + flag)
	search = re.search(error, req.text)
	if search == None :
		print("Trouve !")
		return True
	else :
		return False

def get_blocks(data, block_size):
    return [data[i*(block_size*2):(i+1)*(block_size*2)] for i in range(len(data)//(block_size*2))]

def decrypt(flag, block_size, error):
	BlockList = get_blocks(flag, block_size)
	print(BlockList)
	BlockString = ["".join(BlockList[i]) for i in range(len(BlockList))]
	BlockFound = list((0,)*block_size)
	ourBlock = list(('00',)*block_size)
	Plaintext = ""
	for block in reversed(range(1, len(BlockList))) :
		for i in reversed(range(0, block_size)) :
			print("Byte " + str(i))
			for k in range(0, 256) :
				# Incrementation du bloc controle
				if k < 16 :
					ourBlock[i] = '0' + format(k, 'x')
				else :
					ourBlock[i] = format(k, 'x')
				toSend = ''.join(ourBlock) + BlockString[block]
				
				if oracle(toSend, error) :
					# I = Etat intermediaire
					# C = Ciphertext
					# P = Plaintext
					# C' = Ciphertext controle
					# P' = Plaintext dont l'oracle verifie le padding
					# I[n] = C[n-1]^P[n] et P'[n] = P[n]^C[n-1]^C'
					# Ici : P[n][k] = C[n-1][k]^C'[k]^j, avec j de 1 a 15
					new = int(BlockList[block - 1][i], 16) ^ int(ourBlock[i], 16) ^ (block_size-i)
					print(new)
					BlockFound[i] = new
					Plaintext = chr(new) + Plaintext
					print(Plaintext)
					break
				# Si on y est, pas de resultats
				if k == 255 :	
					print("Aucun resultat...")
					exit()

			# Incrementation du bloc controle
			# On veut trouver les bits suivants
			# C'[k] = P[n][k]^C[n-1][k]^2
			for j in range(i, block_size) :
				tmp = BlockFound[j] ^ int(BlockList[block - 1][j], 16) ^ (block_size+1-i)
				if tmp < 16 :
					ourBlock[j] = '0' + format(tmp, 'x')
				else :
					ourBlock[j] = format(tmp, 'x')

		print("Plaintext : " + Plaintext)

flag = normalizeFlag(flagToNormalize, args.type)
decrypt(flag, block_size, args.error)